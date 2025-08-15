# smc_backtest_v2.4_api.py  (v2.5.1 semi-agressive, FINAL)
# TwelveData fetch (LTF+HTF) -> CSV -> SMC backtest
# Auto-bias, Confluence≥2/3, Mitigation (+ fallback optionnel), CT encadrée
# Sanity SL/TP, micro-stop filter, expectancy correcte (R basé sur risque à l'entrée)
# Journal + Résultats + Equity curve

import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ============================== SECTION 1 : FETCH ==============================

def fetch_twelvedata(api_key: str, symbol: str, interval: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "format": "JSON",
      # "dp": 6,  # si ton plan le permet, garde par défaut
        "timezone": "UTC",
        "order": "ASC"
    }
    if start: params["start_date"] = start
    if end:   params["end_date"] = end

    print(f"📡 Fetch {symbol} {interval} {start} → {end} ...")
    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Fallback si pas de values
    if not isinstance(data, dict) or "values" not in data or not data["values"]:
        if start and end:
            try:
                end_dt = datetime.fromisoformat(end) + timedelta(days=1)
                params["end_date"] = end_dt.strftime("%Y-%m-%d")
                r2 = requests.get(base_url, params=params, timeout=60)
                r2.raise_for_status()
                data2 = r2.json()
                if isinstance(data2, dict) and "values" in data2 and data2["values"]:
                    data = data2
                else:
                    raise RuntimeError(f"Erreur API TwelveData: {data2}")
            except Exception:
                raise RuntimeError(f"Erreur API TwelveData: {data}")
        else:
            raise RuntimeError(f"Erreur API TwelveData: {data}")

    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for c in ("open","high","low","close"):
        df[c] = df[c].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time","open","high","low","close"]]

# ============================ SECTION 2 : UTILS/SMC ===========================

def parse_tf(s: str) -> pd.Timedelta:
    s = s.strip().lower()
    if s.endswith("min"): return pd.Timedelta(minutes=int(s[:-3]))
    if s.endswith("h"):   return pd.Timedelta(hours=int(s[:-1]))
    if s.endswith("d"):   return pd.Timedelta(days=int(s[:-1]))
    raise ValueError(f"Unsupported timeframe: {s}")

def to_pips(symbol: str) -> float:
    s = symbol.replace("-", "").replace("_", "").replace("/", "").upper()
    # JPY pairs: 1 pip = 0.01
    jpy = ("USDJPY","EURJPY","GBPJPY","AUDJPY","NZDJPY","CADJPY","CHFJPY")
    if s in jpy or s.endswith("JPY"):
        return 1e-2
    # XAU/XAG: souvent 0.1 (selon broker), prends 0.1 comme "pip"
    if "XAU" in s or "XAG" in s:
        return 1e-1
    # Crypto (si jamais) – 1 pip arbitraire plus fin
    if "BTC" in s or "ETH" in s:
        return 1e-2
    # Par défaut (majors/FX 5 décimales): 1 pip = 0.0001
    return 1e-4


def commission_cost(notional: float, commission_per_million: float) -> float:
    return (notional / 1_000_000.0) * commission_per_million

def atr(df: pd.DataFrame, period: int = 14, alpha: Optional[float] = None) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    if alpha is not None:
        return tr.ewm(alpha=alpha, adjust=False).mean()
    return tr.rolling(window=period, min_periods=1).mean()

def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    highs = df["high"].values
    lows  = df["low"].values
    sh = np.zeros(len(df), dtype=bool)
    sl = np.zeros(len(df), dtype=bool)
    for i in range(left, len(df)-right):
        if highs[i] == max(highs[i-left:i+right+1]): sh[i] = True
        if lows[i]  == min(lows[i-left:i+right+1]):  sl[i] = True
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"]  = sl
    return out

def last_bos_bias_series(htf: pd.DataFrame) -> pd.DataFrame:
    htf_sw = find_swings(htf, 2, 2).reset_index(drop=True)
    last_high = None
    last_low  = None
    bias = "neutral"
    blist: List[str] = []
    for i in range(len(htf_sw)):
        if htf_sw.at[i,"swing_high"]: last_high = htf_sw.at[i,"high"]
        if htf_sw.at[i,"swing_low"]:  last_low  = htf_sw.at[i,"low"]
        c = htf_sw.at[i,"close"]
        if last_high is not None and c > last_high: bias = "bull"
        if last_low  is not None and c < last_low:  bias = "bear"
        blist.append(bias)
    out = htf_sw.copy()
    out["bias"] = blist
    return out[["time","open","high","low","close","bias"]]

# Bear signals
def had_bear_sweep(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    lo = max(0, i-lookback)
    if i <= lo: return None
    recent_high = df["high"].iloc[lo:i].max()
    if df["high"].iat[i] > recent_high and df["close"].iat[i] < recent_high:
        return recent_high
    return None

def bear_bos(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    lo = max(0, i-lookback)
    if i <= lo: return None
    recent_low = df["low"].iloc[lo:i].min()
    return recent_low if df["close"].iat[i] < recent_low else None

def bear_fvg(df: pd.DataFrame, i: int) -> Optional[Tuple[float,float]]:
    if i < 1 or i+1 >= len(df): return None
    hi_next = df["high"].iat[i+1]
    lo_prev = df["low"].iat[i-1]
    return (hi_next, lo_prev) if hi_next < lo_prev else None  # gap [hi_next, lo_prev]

# Bull signals
def had_bull_sweep(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    lo = max(0, i-lookback)
    if i <= lo: return None
    recent_low = df["low"].iloc[lo:i].min()
    if df["low"].iat[i] < recent_low and df["close"].iat[i] > recent_low:
        return recent_low
    return None

def bull_bos(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    lo = max(0, i-lookback)
    if i <= lo: return None
    recent_high = df["high"].iloc[lo:i].max()
    return recent_high if df["close"].iat[i] > recent_high else None

def bull_fvg(df: pd.DataFrame, i: int) -> Optional[Tuple[float,float]]:
    if i < 1 or i+1 >= len(df): return None
    lo_next = df["low"].iat[i+1]
    hi_prev = df["high"].iat[i-1]
    return (hi_prev, lo_next) if lo_next > hi_prev else None  # gap [hi_prev, lo_next]

# Confluence/momentum/news helpers
def count_confluence_bear(df, i) -> Tuple[int, Tuple[Optional[float],Optional[float],Optional[Tuple[float,float]]]]:
    sw = had_bear_sweep(df, i-1)
    bos = bear_bos(df, i)
    fvg = bear_fvg(df, i-1)
    return int(sw is not None) + int(bos is not None) + int(fvg is not None), (sw, bos, fvg)

def count_confluence_bull(df, i) -> Tuple[int, Tuple[Optional[float],Optional[float],Optional[Tuple[float,float]]]]:
    sw = had_bull_sweep(df, i-1)
    bos = bull_bos(df, i)
    fvg = bull_fvg(df, i-1)
    return int(sw is not None) + int(bos is not None) + int(fvg is not None), (sw, bos, fvg)

def momentum_ok(df, i, atr_series, min_ratio: float) -> bool:
    if min_ratio <= 0: return True
    body = abs(df.at[i,"close"] - df.at[i,"open"])
    a = float(atr_series.iat[i]) if i < len(atr_series) else float(atr_series.iloc[-1])
    return a > 0 and (body / a) >= min_ratio

def is_near_macro(ts: pd.Timestamp, window_min: int) -> bool:
    if window_min <= 0: return False
    macros = [(12,0),(13,30),(14,0),(14,30),(18,0)]  # UTC slots simples
    for h,m in macros:
        t0 = ts.replace(hour=h, minute=m, second=0, microsecond=0)
        if abs((ts - t0).total_seconds()) <= window_min*60:
            return True
    return False

# ============================== SECTION 3 : ENGINE =============================

@dataclass
class Trade:
    time_enter: pd.Timestamp
    side: str
    entry: float
    sl: float
    tp: float
    size: int
    equity_before: float
    equity_after: float
    exit_time: pd.Timestamp
    exit_price: float
    reason: str
    r_multiple: float

def backtest_smc(
    df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
    symbol: str = "EUR/USD",
    ltf: str = "15min", htf: str = "4h",
    capital: float = 100000.0, risk_per_trade: float = 0.005,
    leverage_max: int = 30, lot_min: int = 1000, lot_step: int = 1000,
    spread_pips: float = 0.2, slip_entry_pips: float = 0.1, slip_exit_pips: float = 0.1,
    commission_per_million: float = 7.0,
    forbidden_hours_utc: Tuple[int,...] = (0,2,3,4,16,17,23),
    atr_period: int = 14, atr_alpha: float = 0.4, atr_min_pips: float = 1.0,
    min_stop_pips: float = 1.6, require_confluence: bool = True,
    buffer_sl_pips: float = 0.4,
    rr_target: float = 2.0, rr_target_alt: float = 1.3, atr_rr_switch_pips: float = 1.1,
    use_partials: bool = True, partial_take_r: float = 1.5, move_be_r: float = 1.0,
    trail_struct_window: int = 20,
    auto_bias: bool = True, force_bias: Optional[str] = None,
    # v2.5 options
    confluence_min: int = 2,
    pullback_bars: int = 3, pullback_optional: bool = True, fallback_market_bars: int = 2,
    allow_countertrend: bool = True, ct_risk_factor: float = 0.5, ct_rr_cap: float = 1.2,
    momentum_min_body_atr: float = 0.25, news_free_minutes: int = 30
):
    pips = to_pips(symbol)

    # LTF prep
    df = df_ltf.copy().reset_index(drop=True)
    df["ATR"] = atr(df, period=atr_period, alpha=atr_alpha)
    df_sw = find_swings(df, 2, 2)

    # HTF bias series & align to LTF
    htf_bias_df = last_bos_bias_series(df_htf.copy())
    bias_map = pd.merge_asof(df[["time"]], htf_bias_df[["time","bias"]], on="time", direction="backward")
    df["HTF_bias"] = bias_map["bias"].ffill().fillna("neutral")

    equity = capital
    trades: List[Trade] = []

    # Position state
    position = None
    entry_idx = None
    entry_price = None
    sl_price = None
    tp_price = None
    size_units = 0
    be_armed = False
    partial_done = False
    entry_fees = 0.0
    init_sl_price = None
    risk_amount_open = 0.0  # <— nécessaire pour nonlocal

    # helper: nearest swing after entry
    def nearest_swing_after(entry_index: int, window: int, side: str) -> Optional[float]:
        lo = max(0, entry_index)
        hi = min(len(df_sw)-1, entry_index + window)
        sub = df_sw.iloc[lo:hi+1]
        if side == "SHORT":
            hs = sub[sub["swing_high"]]
            if hs.empty: return None
            return float(hs["high"].iloc[0])
        else:
            ls = sub[sub["swing_low"]]
            if ls.empty: return None
            return float(ls["low"].iloc[0])

    locked_bias = None

    # === inner entry funcs (corrigées) ===
    def try_short_entry(use_ct=False) -> bool:
        nonlocal position, entry_idx, entry_price, sl_price, tp_price, size_units, be_armed, partial_done, init_sl_price, entry_fees, equity, risk_amount_open
        conf, (sw, bos, fvg) = count_confluence_bear(df, i)
        if conf < max(1, confluence_min) and not use_ct:
            return False

        fvg_low, fvg_high = fvg if fvg else (None, None)
        entered = False
        waited = 0
        target_rr = min(rr_use, ct_rr_cap) if use_ct else rr_use
        risk_factor = (ct_risk_factor if use_ct else 1.0)

        # Attente mitigation (pas de fallback ici: si tu en veux, active plus bas)
        for j in range(1, pullback_bars+1):
            if i+j >= len(df): break
            if fvg_high is not None and df.at[i+j,"high"] >= fvg_high:
                entry_exec = fvg_high - (spread_pips + slip_entry_pips) * pips
                waited = j; entered = True; break

        # fallback optionnel
        if not entered and pullback_optional:
            fb = fallback_market_bars
            if fb > 0 and i+fb < len(df):
                entry_exec = df.at[i+fb,"open"] - (spread_pips + slip_entry_pips) * pips
                waited = fb; entered = True

        if not entered:
            return False

        # Structure & SL/TP init
        struct_high = max((sw or df.at[i-1,"high"]), df.at[i-1,"high"])
        raw_sl = struct_high + buffer_sl_pips * pips
        stop_dist = raw_sl - entry_exec
        stop_pips_val = stop_dist / pips
        if stop_dist <= 0:
            return False

        tp_px_calc = entry_exec - target_rr * stop_dist  # calculé avant sanity

        # --- SANITY SHORT ---
        min_margin = (spread_pips + slip_entry_pips + slip_exit_pips) * pips

        # SL au-dessus de l'entrée
        if raw_sl <= entry_exec + min_margin:
            raw_sl = entry_exec + max(min_margin, buffer_sl_pips * pips)
            stop_dist = raw_sl - entry_exec
            stop_pips_val = stop_dist / pips
            if stop_dist <= 0:
                return False

        # TP sous l'entrée
        if tp_px_calc >= entry_exec - min_margin:
            tp_px_calc = entry_exec - max(min_margin, rr_target_alt * min_margin)

        # micro-stop + min_stop_pips
        if stop_pips_val < max(min_stop_pips, atr_min_pips * 0.6):
            return False

        # sens final
        if not (raw_sl > entry_exec and tp_px_calc < entry_exec):
            return False

        # Sizing
        risk_amount_at_entry = equity * risk_per_trade * risk_factor
        units_calc = risk_amount_at_entry / stop_dist
        units = int(max(lot_min, ((units_calc // lot_step) * lot_step)))

        # Leverage cap
        notional = units * entry_exec
        max_notional = equity * leverage_max
        if max_notional > 0 and notional > max_notional:
            scale = max_notional / notional
            units = int(max(lot_min, ((units * scale) // lot_step) * lot_step))
        if units <= 0:
            return False

        entry_notional = units * entry_exec
        fees = commission_cost(entry_notional, commission_per_million)
        equity -= fees

        # Commit
        position = "SHORT"; entry_idx = i+waited; entry_price = entry_exec
        sl_price = raw_sl; tp_price = tp_px_calc; size_units = units
        be_armed = False; partial_done = False; init_sl_price = raw_sl; entry_fees = fees
        risk_amount_open = risk_amount_at_entry
        return True

    def try_long_entry(use_ct=False) -> bool:
        nonlocal position, entry_idx, entry_price, sl_price, tp_price, size_units, be_armed, partial_done, init_sl_price, entry_fees, equity, risk_amount_open
        conf, (sw, bos, fvg) = count_confluence_bull(df, i)
        if conf < max(1, confluence_min) and not use_ct:
            return False

        fvg_lo, fvg_hi = fvg if fvg else (None, None)
        entered = False
        waited = 0
        target_rr = min(rr_use, ct_rr_cap) if use_ct else rr_use
        risk_factor = (ct_risk_factor if use_ct else 1.0)

        # Attente mitigation
        for j in range(1, pullback_bars+1):
            if i+j >= len(df): break
            if fvg_lo is not None and df.at[i+j,"low"] <= fvg_lo:
                entry_exec = fvg_lo + (spread_pips + slip_entry_pips) * pips
                waited = j; entered = True; break

        # fallback optionnel
        if not entered and pullback_optional:
            fb = fallback_market_bars
            if fb > 0 and i+fb < len(df):
                entry_exec = df.at[i+fb,"open"] + (spread_pips + slip_entry_pips) * pips
                waited = fb; entered = True

        if not entered:
            return False

        # Structure & SL/TP init
        struct_low = min((sw or df.at[i-1,"low"]), df.at[i-1,"low"])
        raw_sl = struct_low - buffer_sl_pips * pips
        stop_dist = entry_exec - raw_sl
        stop_pips_val = stop_dist / pips
        if stop_dist <= 0:
            return False

        tp_px_calc = entry_exec + target_rr * stop_dist  # calculé avant sanity

        # --- SANITY LONG ---
        min_margin = (spread_pips + slip_entry_pips + slip_exit_pips) * pips

        # SL en-dessous de l'entrée
        if raw_sl >= entry_exec - min_margin:
            raw_sl = entry_exec - max(min_margin, buffer_sl_pips * pips)
            stop_dist = entry_exec - raw_sl
            stop_pips_val = stop_dist / pips
            if stop_dist <= 0:
                return False

        # TP au-dessus de l'entrée
        if tp_px_calc <= entry_exec + min_margin:
            tp_px_calc = entry_exec + max(min_margin, rr_target_alt * min_margin)

        # micro-stop + min_stop_pips
        if stop_pips_val < max(min_stop_pips, atr_min_pips * 0.6):
            return False

        # sens final
        if not (raw_sl < entry_exec and tp_px_calc > entry_exec):
            return False

        # Sizing
        risk_amount_at_entry = equity * risk_per_trade * risk_factor
        units_calc = risk_amount_at_entry / stop_dist
        units = int(max(lot_min, ((units_calc // lot_step) * lot_step)))

        # Leverage cap
        notional = units * entry_exec
        max_notional = equity * leverage_max
        if max_notional > 0 and notional > max_notional:
            scale = max_notional / notional
            units = int(max(lot_min, ((units * scale) // lot_step) * lot_step))
        if units <= 0:
            return False

        entry_notional = units * entry_exec
        fees = commission_cost(entry_notional, commission_per_million)
        equity -= fees

        # Commit
        position = "LONG"; entry_idx = i+waited; entry_price = entry_exec
        sl_price = raw_sl; tp_price = tp_px_calc; size_units = units
        be_armed = False; partial_done = False; init_sl_price = raw_sl; entry_fees = fees
        risk_amount_open = risk_amount_at_entry
        return True

    # === main loop ===
    for i in range(30, len(df)-1):  # ensure i+1 exists
        t = df.at[i, "time"]
        if t.hour in forbidden_hours_utc:
            continue
        if (df.at[i, "ATR"] / pips) < atr_min_pips:
            continue
        if is_near_macro(t, news_free_minutes):
            continue

        # Bias
        bias_here = df.at[i, "HTF_bias"]
        if force_bias in ("bull","bear"):
            bias_here = force_bias
        elif not auto_bias:
            if locked_bias is None:
                locked_bias = bias_here
            bias_here = locked_bias

        # manage open position
        if position is not None:
            # move BE at 1R
            if not be_armed and init_sl_price is not None:
                R = abs(entry_price - init_sl_price)
                if R > 0:
                    if position == "SHORT":
                        fav = entry_price - df.at[i, "low"]
                        if fav >= move_be_r * R:
                            sl_price = entry_price - (spread_pips + slip_exit_pips) * pips
                            be_armed = True
                    else:
                        fav = df.at[i, "high"] - entry_price
                        if fav >= move_be_r * R:
                            sl_price = entry_price + (spread_pips + slip_exit_pips) * pips
                            be_armed = True

            # partials
            if use_partials and not partial_done and init_sl_price is not None:
                R = abs(entry_price - init_sl_price)
                if R > 0:
                    if position == "SHORT" and (entry_price - df.at[i,"low"]) >= partial_take_r * R:
                        exit_px = max(df.at[i,"low"], entry_price - partial_take_r*R) - slip_exit_pips*pips
                        notional_half = (size_units*0.5) * exit_px
                        fees_half = commission_cost(notional_half, commission_per_million)
                        pnl_half = (entry_price - exit_px) * (size_units*0.5) - fees_half
                        equity += pnl_half
                        size_units = int(size_units*0.5)
                        partial_done = True
                    elif position == "LONG" and (df.at[i,"high"] - entry_price) >= partial_take_r * R:
                        exit_px = min(df.at[i,"high"], entry_price + partial_take_r*R) + slip_exit_pips*pips
                        notional_half = (size_units*0.5) * exit_px
                        fees_half = commission_cost(notional_half, commission_per_million)
                        pnl_half = (exit_px - entry_price) * (size_units*0.5) - fees_half
                        equity += pnl_half
                        size_units = int(size_units*0.5)
                        partial_done = True

            # trailing by structure
            sw_level = nearest_swing_after(entry_idx, trail_struct_window, position)
            if sw_level is not None:
                if position == "SHORT":
                    new_sl = sw_level + buffer_sl_pips * pips
                    if new_sl < sl_price: sl_price = new_sl
                else:
                    new_sl = sw_level - buffer_sl_pips * pips
                    if new_sl > sl_price: sl_price = new_sl

            # exits
            hit_sl = (df.at[i,"high"] >= sl_price) if position == "SHORT" else (df.at[i,"low"] <= sl_price)
            hit_tp = (df.at[i,"low"] <= tp_price) if position == "SHORT" else (df.at[i,"high"] >= tp_price)

            if hit_sl or hit_tp:
                reason = "SL" if hit_sl and not hit_tp else ("TP" if hit_tp and not hit_sl else "SL")
                if reason == "SL":
                    exit_px = sl_price + (slip_exit_pips*pips if position == "SHORT" else -slip_exit_pips*pips)
                else:
                    exit_px = tp_price - (slip_exit_pips*pips if position == "SHORT" else -slip_exit_pips*pips)

                exit_notional = size_units * exit_px
                exit_fees = commission_cost(exit_notional, commission_per_million)
                pnl = ((entry_price - exit_px) if position=="SHORT" else (exit_px - entry_price)) * size_units - entry_fees - exit_fees
                equity_after = equity + pnl
                r_mult = (pnl / risk_amount_open) if risk_amount_open > 0 else 0.0

                trades.append(Trade(
                    time_enter=df.at[entry_idx,"time"],
                    side=position, entry=entry_price, sl=sl_price, tp=tp_price, size=size_units,
                    equity_before=equity, equity_after=equity_after,
                    exit_time=df.at[i,"time"], exit_price=exit_px, reason=reason, r_multiple=r_mult
                ))
                equity = equity_after

                # reset
                position = None; entry_idx = None; entry_price = None
                sl_price = None; tp_price = None; size_units = 0
                be_armed = False; partial_done = False; entry_fees = 0.0; init_sl_price = None
                risk_amount_open = 0.0
                continue

        # no open position -> look for entries (with momentum filter)
        if position is None:
            if not momentum_ok(df, i, df["ATR"], momentum_min_body_atr):
                continue

            atr_pips_here = df.at[i,"ATR"] / pips
            rr_use = rr_target if atr_pips_here >= atr_rr_switch_pips else rr_target_alt

            # route principale
            if bias_here == "bear":
                entered_main = try_short_entry(use_ct=False)
                if not entered_main and allow_countertrend:
                    try_long_entry(use_ct=True)
            elif bias_here == "bull":
                entered_main = try_long_entry(use_ct=False)
                if not entered_main and allow_countertrend:
                    try_short_entry(use_ct=True)

    # Force close if still open at the end
    if position is not None:
        last_i = len(df)-1
        exit_px = df.at[last_i, "close"]
        exit_notional = size_units * exit_px
        exit_fees = commission_cost(exit_notional, commission_per_million)
        pnl = ((entry_price - exit_px) if position=="SHORT" else (exit_px - entry_price)) * size_units - entry_fees - exit_fees
        equity_after = equity + pnl
        r_mult = (pnl / risk_amount_open) if risk_amount_open > 0 else 0.0
        trades.append(Trade(
            time_enter=df.at[entry_idx,"time"],
            side=position, entry=entry_price, sl=sl_price, tp=tp_price, size=size_units,
            equity_before=equity, equity_after=equity_after,
            exit_time=df.at[last_i,"time"], exit_price=exit_px, reason="Timeout", r_multiple=r_mult
        ))
        equity = equity_after
        risk_amount_open = 0.0

    # Metrics
    eq_curve = [capital] + [t.equity_after for t in trades]
    peak = eq_curve[0]; mdd = 0.0
    for v in eq_curve:
        if v > peak: peak = v
        draw = (peak - v) / peak * 100.0
        if draw > mdd: mdd = draw
    wins = [t for t in trades if t.equity_after > t.equity_before]
    avg_R = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0
    winrate = (len(wins)/len(trades)*100.0) if trades else 0.0

    results = {
        "final_equity": equity,
        "return_pct": (equity / capital - 1.0)*100.0,
        "trades": len(trades),
        "winrate_pct": winrate,
        "avg_R": avg_R,
        "max_drawdown_pct": mdd
    }
    return results, trades

# ============================== SECTION 4 : MAIN ==============================

def main():
    ap = argparse.ArgumentParser()
    # Source & TF
    ap.add_argument("--api-key", type=str, required=True, help="TwelveData API key")
    ap.add_argument("--symbol", type=str, default="EUR/USD")
    ap.add_argument("--ltf", type=str, default="15min")
    ap.add_argument("--htf", type=str, default="4h")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)

    # Risk / sizing / leverage
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--risk_per_trade", type=float, default=0.005)
    ap.add_argument("--leverage_max", type=int, default=30)
    ap.add_argument("--lot_min", type=int, default=1000)
    ap.add_argument("--lot_step", type=int, default=1000)

    # Costs
    ap.add_argument("--spread_pips", type=float, default=0.2)
    ap.add_argument("--slip_entry_pips", type=float, default=0.1)
    ap.add_argument("--slip_exit_pips", type=float, default=0.1)
    ap.add_argument("--commission_per_million", type=float, default=7.0)

    # Filters (defaults v2.5)
    ap.add_argument("--forbidden_hours_utc", type=str, default="0,2,3,4,16,17,23")
    ap.add_argument("--atr_period", type=int, default=14)
    ap.add_argument("--atr_alpha", type=float, default=0.4)
    ap.add_argument("--atr_min_pips", type=float, default=1.0)
    ap.add_argument("--min_stop_pips", type=float, default=1.6)
    ap.add_argument("--require_confluence", action="store_true")

    # Management (defaults v2.5)
    ap.add_argument("--buffer_sl_pips", type=float, default=0.4)
    ap.add_argument("--rr_target", type=float, default=2.0)
    ap.add_argument("--rr_target_alt", type=float, default=1.3)
    ap.add_argument("--atr_rr_switch_pips", type=float, default=1.1)
    ap.add_argument("--use_partials", action="store_true")
    ap.add_argument("--partial_take_r", type=float, default=1.5)
    ap.add_argument("--move_be_r", type=float, default=1.0)
    ap.add_argument("--trail_struct_window", type=int, default=20)

    # Bias
    ap.add_argument("--auto-bias", action="store_true", help="Re-evaluate HTF bias dynamically")
    ap.add_argument("--force-bias", type=str, default=None, choices=[None,"bull","bear"])

    # v2.5 – nouveaux switchs
    ap.add_argument("--confluence_min", type=int, default=2, help="Min signals among Sweep/BOS/FVG")
    ap.add_argument("--pullback_bars", type=int, default=3, help="Bars to wait for mitigation")
    ap.add_argument("--pullback_optional", action="store_true", help="If no mitigation, fallback market")
    ap.add_argument("--fallback_market_bars", type=int, default=2, help="Bars before fallback market")
    ap.add_argument("--allow_countertrend", action="store_true", help="Allow countertrend scalp")
    ap.add_argument("--ct_risk_factor", type=float, default=0.5, help="Risk fraction for CT trades")
    ap.add_argument("--ct_rr_cap", type=float, default=1.2, help="Max RR for CT trades")
    ap.add_argument("--momentum_min_body_atr", type=float, default=0.25, help="|close-open|/ATR min (anti-doji)")
    ap.add_argument("--news-free-minutes", type=int, default=30, help="Block entries ±N min around macro times")

    ap.add_argument("--pending_ttl_bars", type=int, default=24, help="Durée de vie des ordres en barres LTF")
    ap.add_argument("--session_start_utc", type=int, default=6, help="Heure UTC de début de session")
    ap.add_argument("--session_end_utc", type=int, default=20, help="Heure UTC de fin de session")

    args = ap.parse_args()

    # Fenêtre par défaut robuste
    if not args.start and not args.end:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
        args.start = start_dt.strftime("%Y-%m-%d")
        args.end   = end_dt.strftime("%Y-%m-%d")
    elif args.start and not args.end:
        start_dt = datetime.fromisoformat(args.start)
        args.end = (start_dt + timedelta(days=30)).strftime("%Y-%m-%d")
    elif args.end and not args.start:
        end_dt = datetime.fromisoformat(args.end)
        args.start = (end_dt - timedelta(days=30)).strftime("%Y-%m-%d")
    # Pad fin d’1 jour pour inclure la journée partielle
    sdt = datetime.fromisoformat(args.start)
    edt = datetime.fromisoformat(args.end) + timedelta(days=1)
    args.start = sdt.strftime("%Y-%m-%d")
    args.end   = edt.strftime("%Y-%m-%d")
    print(f"📅 Période: {args.start} → {args.end}")

    # Fetch LTF + HTF and save CSVs in current folder
    df_ltf = fetch_twelvedata(args.api_key, args.symbol, args.ltf, args.start, args.end)
    ltf_csv = f"{args.symbol.replace('/','')}_{args.ltf}.csv"
    df_ltf.to_csv(ltf_csv, index=False)
    print(f"💾 LTF saved: {ltf_csv}")

    df_htf = fetch_twelvedata(args.api_key, args.symbol, args.htf, args.start, args.end)
    htf_csv = f"{args.symbol.replace('/','')}_{args.htf}.csv"
    df_htf.to_csv(htf_csv, index=False)
    print(f"💾 HTF saved: {htf_csv}")

    forbidden_hours = tuple(int(x) for x in args.forbidden_hours_utc.split(",") if x != "")

    results, trades = backtest_smc(
        df_ltf=df_ltf, df_htf=df_htf,
        symbol=args.symbol, ltf=args.ltf, htf=args.htf,
        capital=args.capital, risk_per_trade=args.risk_per_trade,
        leverage_max=args.leverage_max, lot_min=args.lot_min, lot_step=args.lot_step,
        spread_pips=args.spread_pips, slip_entry_pips=args.slip_entry_pips, slip_exit_pips=args.slip_exit_pips,
        commission_per_million=args.commission_per_million,
        forbidden_hours_utc=forbidden_hours,
        atr_period=args.atr_period, atr_alpha=args.atr_alpha, atr_min_pips=args.atr_min_pips, min_stop_pips=args.min_stop_pips,
        require_confluence=args.require_confluence,
        buffer_sl_pips=args.buffer_sl_pips,
        rr_target=args.rr_target, rr_target_alt=args.rr_target_alt, atr_rr_switch_pips=args.atr_rr_switch_pips,
        use_partials=args.use_partials, partial_take_r=args.partial_take_r, move_be_r=args.move_be_r,
        trail_struct_window=args.trail_struct_window,
        auto_bias=args.__dict__.get("auto_bias", False),
        force_bias=args.__dict__.get("force_bias", None),
        # v2.5
        confluence_min=args.confluence_min,
        pullback_bars=args.pullback_bars, pullback_optional=args.pullback_optional, fallback_market_bars=args.fallback_market_bars,
        allow_countertrend=args.allow_countertrend, ct_risk_factor=args.ct_risk_factor, ct_rr_cap=args.ct_rr_cap,
        momentum_min_body_atr=args.momentum_min_body_atr, news_free_minutes=args.news_free_minutes
    )

    # Journal
    print("\n===== JOURNAL DES TRADES =====")
    for tr in trades:
        print(f"[{tr.time_enter}] {tr.side} @ {tr.entry:.5f} SL {tr.sl:.5f} TP {tr.tp:.5f} size {tr.size:,} "
              f"Exit {tr.reason} @ {tr.exit_price:.5f} | Equity: {tr.equity_after:.2f}")

    # Résumé
    print("\n===== RÉSULTATS =====")
    for k, v in results.items():
        if isinstance(v, float):
            if "pct" in k:
                print(f"{k}: {v:.2f}%")
            else:
                print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")

    # Equity curve
    if trades:
        eq = [args.capital] + [t.equity_after for t in trades]
        xs = list(range(len(eq)))
        plt.figure(figsize=(11,6))
        plt.plot(xs, eq, label="Equity")
        # Mark wins/losses
        win_idx = [i+1 for i, t in enumerate(trades) if t.equity_after > t.equity_before]
        loss_idx = [i+1 for i, t in enumerate(trades) if t.equity_after < t.equity_before]
        win_eq = [eq[i] for i in win_idx]
        loss_eq = [eq[i] for i in loss_idx]
        plt.scatter(win_idx, win_eq, label="Wins", c="green")
        plt.scatter(loss_idx, loss_eq, label="Losses", c="red")
        plt.title("Equity Curve")
        plt.xlabel("Trade #")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("equity_curve.png")
        print("🖼️  equity_curve.png sauvegardé.")
        plt.show()
    else:
        print("Aucun trade -> pas de courbe d'equity.")

if __name__ == "__main__":
    main()


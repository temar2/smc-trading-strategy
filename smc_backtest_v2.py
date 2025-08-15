# smc_backtest_v2.py
# V2 SMC Backtest autonome (pandas/numpy)
# Entrée: CSV LTF: columns=time,open,high,low,close (UTC ISO8601)
# Sortie: résumé perf + journal des trades
# Features:
# - Bias HTF via swings & BOS
# - Signals LTF: Sweep + BOS + FVG (triple confluence)
# - ATR filter
# - SL beyond structure + buffer
# - BE @ 1R, partials 50% @ 1.5R, trailing stop by structure
# - Time session filters
# - Fees: spread, slippage, commission

import argparse, math
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


# ---------- helpers timeframes ----------

def parse_tf(s: str) -> pd.Timedelta:
    s = s.strip().lower()
    if s.endswith("min"):
        return pd.Timedelta(minutes=int(s[:-3]))
    if s.endswith("h"):
        return pd.Timedelta(hours=int(s[:-1]))
    if s.endswith("d"):
        return pd.Timedelta(days=int(s[:-1]))
    raise ValueError(f"Unsupported timeframe: {s}")

def to_pips(sym: str) -> float:
    # simplifié pour majors/FX à 5 décimales (EURUSD…)
    return 1e-4

# ---------- swings & structure ----------

def find_swings(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    # ajoute cols swing_high/swing_low bool sur base fractale
    highs = df["high"].values
    lows  = df["low"].values
    sh = np.zeros(len(df), dtype=bool)
    sl = np.zeros(len(df), dtype=bool)
    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i-left:i+right+1]):
            sh[i] = True
        if lows[i]  == min(lows[i-left:i+right+1]):
            sl[i] = True
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"]  = sl
    return out

def last_bos_bias(htf: pd.DataFrame) -> str:
    # très simple: regarde dernier break de swing high/low confirmé en clôture
    htf = htf.copy()
    htf = find_swings(htf, 2, 2)
    last_high = None
    last_low  = None
    bias = "neutral"
    for i in range(len(htf)):
        if htf["swing_high"].iat[i]:
            last_high = htf["high"].iat[i]
        if htf["swing_low"].iat[i]:
            last_low = htf["low"].iat[i]
        # BOS up si close > last_high
        if last_high is not None and htf["close"].iat[i] > last_high:
            bias = "bull"
        # BOS down si close < last_low
        if last_low is not None and htf["close"].iat[i] < last_low:
            bias = "bear"
    return bias

# ---------- ATR ----------

def atr(df: pd.DataFrame, period: int = 14, alpha: Optional[float] = None) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    if alpha:
        return tr.ewm(alpha=alpha, adjust=False).mean()
    return tr.rolling(window=period, min_periods=1).mean()

# ---------- SMC patterns (LTF) ----------

def had_bear_sweep(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    # Sweep (short): la bougie i a pris la liquidité au-dessus d'un swing récent puis a réintégré (close < swept high)
    lo = max(0, i - lookback)
    recent_high = df["high"].iloc[lo:i].max() if i > lo else None
    if recent_high is None or np.isnan(recent_high):
        return None
    took_out = df["high"].iat[i] > recent_high
    reenter  = df["close"].iat[i] < recent_high
    return recent_high if took_out and reenter else None

def bear_bos(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    # BOS down: close i < plus bas des swing lows récents
    lo = max(0, i - lookback)
    recent_low = df["low"].iloc[lo:i].min() if i > lo else None
    if recent_low is None or np.isnan(recent_low):
        return None
    return recent_low if df["close"].iat[i] < recent_low else None

def bear_fvg(df: pd.DataFrame, i: int) -> Optional[Tuple[float, float]]:
    # Bearish FVG: high(i+1) < low(i-1) → gap [high(i+1), low(i-1)]
    if i < 1 or i+1 >= len(df):
        return None
    hi_next = df["high"].iat[i+1]
    lo_prev = df["low"].iat[i-1]
    if hi_next < lo_prev:
        return (hi_next, lo_prev)  # bas du FVG, haut du FVG
    return None

def nearest_swing_high_after(df: pd.DataFrame, i_entry: int, window: int = 30) -> Optional[float]:
    sub = df.iloc[i_entry - 1 : min(len(df), i_entry + window)]
    sub = find_swings(sub, 2, 2)
    highs = sub.index[sub["swing_high"]]
    if len(highs) == 0:
        return None
    # premier swing high formé après l'entrée (≥ i_entry)
    highs_after = [idx for idx in highs if idx >= sub.index[1]]
    if not highs_after:
        return None
    # prendre la plus proche
    idx = highs_after[0]
    return sub.loc[idx, "high"]

# ---------- fees ----------

def commission_cost(notional: float, commission_per_million: float) -> float:
    return (notional / 1_000_000.0) * commission_per_million

# ---------- dataclasses ----------

@dataclass
class Trade:
    time_enter: pd.Timestamp
    side: str
    entry: float
    sl: float
    tp: float
    size: float
    equity_before: float
    equity_after: float
    exit_time: pd.Timestamp
    exit_price: float
    reason: str  # "SL" / "TP" / "Trail" / "Timeout" / "BE_Partial"
    r_multiple: float

# ---------- core backtest ----------

def backtest_smc(
    df_ltf: pd.DataFrame,
    symbol: str = "EUR/USD",
    ltf: str = "15min",
    htf: str = "4h",
    start: Optional[str] = None,
    end: Optional[str] = None,
    capital: float = 100_000.0,
    risk_per_trade: float = 0.005,   # 0.5%
    spread_pips: float = 0.2,
    slip_entry_pips: float = 0.1,
    slip_exit_pips: float = 0.1,
    commission_per_million: float = 7.0,
    leverage_max: int = 30,
    lot_min: int = 1000,
    lot_step: int = 1000,
    min_stop_pips: float = 1.2,
    atr_period: int = 14,
    atr_alpha: float = 0.4,
    atr_min_pips: float = 0.8,       # ATR LTF minimal en pips (évite ranges)
    require_confluence: bool = True,
    use_partials: bool = True,
    buffer_sl_pips: float = 0.2,
    rr_target: float = 2.0,
    partial_take_r: float = 1.5,
    move_be_r: float = 1.0,
    trail_struct_window: int = 20,
    pending_ttl_bars: int = 24,
    forbidden_hours_utc: Tuple[int, ...] = (2, 3, 4, 16, 17),  # pré-Londres & creux NY
):
    df = df_ltf.copy()
    # time bounds
    if start:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["time"] <= pd.Timestamp(end, tz="UTC")]
    df = df.sort_values("time").reset_index(drop=True).copy()

    # ATR
    df["ATR"] = atr(df, period=atr_period, alpha=atr_alpha)

    # HTF resample
    tf_ltf = parse_tf(ltf)
    tf_htf = parse_tf(htf)
    df = df.set_index("time")
    rule = f"{int(tf_htf / pd.Timedelta(minutes=1))}min"
    htf_df = df.resample(rule).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    htf_df = htf_df.reset_index().rename(columns={"index":"time"})
    bias = last_bos_bias(htf_df)

    # on travaille SHORT only quand bias = bear (tu pourras ouvrir LONG plus tard)
    take_shorts = (bias == "bear")

    pips = to_pips(symbol)

    equity = capital
    trades: List[Trade] = []
    position = None  # pas de position en cours
    entry_idx = None
    entry_price = None
    sl_price = None
    tp_price = None
    size_units = 0
    risk_amount = 0.0
    be_armed = False
    partial_done = False
    highest_since_entry = None  # pour trail (short: on trail via swing highs)
    entry_notional = 0.0
    entry_fees = 0.0

    # remettre l'index time pour itérer
    df = df.reset_index()
    # pré-calcul swings pour trailing plus tard
    df_sw = find_swings(df, 2, 2)

    for i in range(len(df)):
        t = df.loc[i, "time"]
        o = df.loc[i, "open"]
        h = df.loc[i, "high"]
        l = df.loc[i, "low"]
        c = df.loc[i, "close"]
        atr_val = df.loc[i, "ATR"]

        hour = t.hour
        if hour in forbidden_hours_utc:
            pass_for_time = True
        else:
            pass_for_time = False

        # trailing stop / management si position ouverte
        if position == "SHORT":
            # update plus haut depuis l'entrée (utile si on voulait trailing simple)
            if highest_since_entry is None or h > highest_since_entry:
                highest_since_entry = h

            # BE trigger @ 1R
            if not be_armed:
                r_dist = (entry_price - sl_price)
                if r_dist > 0 and (entry_price - c) >= (move_be_r * r_dist):
                    # move SL à BE (entry - spread - slippage sécurité)
                    sl_price = entry_price - (spread_pips + slip_exit_pips) * pips
                    be_armed = True

            # Partial 50% @ 1.5R
            if use_partials and not partial_done:
                r_dist = (entry_price - sl_price)
                if r_dist > 0 and (entry_price - c) >= (partial_take_r * r_dist):
                    # on exécute un TP partiel à 50%
                    exit_px = c - slip_exit_pips * pips
                    notional_half = (size_units * 0.5) * entry_price
                    fees_half = commission_cost(notional_half, commission_per_million)
                    pnl_half = (entry_price - exit_px) * (size_units * 0.5) - fees_half
                    equity += pnl_half
                    size_units *= 0.5
                    partial_done = True
                    # on ne ferme pas le trade; on continue avec trailing

            # trailing par structure: si un swing high se forme après l'entrée, trail SL juste au-dessus (- buffer)
            sh_price = nearest_swing_high_after(df, entry_idx, window=trail_struct_window)
            if sh_price is not None:
                trail = sh_price + (buffer_sl_pips * pips)
                # on trail vers le bas (plus serré) uniquement si cela rapproche pour un short
                if trail < sl_price:
                    sl_price = trail

            # vérifier SL/TP/Trail hits pendant cette bougie
            hit_sl = (h >= sl_price)
            hit_tp = (l <= tp_price)

            exit_now = False
            reason = None
            exit_price_exec = None

            if hit_sl and hit_tp:
                # ambigu: quelle est touchée en premier? on choisit conservateur (SL)
                exit_now = True
                reason = "SL"
                exit_price_exec = sl_price + slip_exit_pips * pips
            elif hit_sl:
                exit_now = True
                reason = "SL"
                exit_price_exec = sl_price + slip_exit_pips * pips
            elif hit_tp:
                exit_now = True
                reason = "TP"
                exit_price_exec = tp_price - slip_exit_pips * pips

            # expiration du pending/TTL (si rien ne se passe trop longtemps) — ici non utilisée en position ouverte

            if exit_now:
                # frais de sortie
                exit_notional = size_units * exit_price_exec
                exit_fees = commission_cost(exit_notional, commission_per_million)
                pnl = (entry_price - exit_price_exec) * size_units - entry_fees - exit_fees
                equity_after = equity + pnl
                r_mult = (entry_price - exit_price_exec) / (entry_price - sl_price) if (entry_price - sl_price) != 0 else 0.0

                trades.append(Trade(
                    time_enter=df.loc[entry_idx, "time"],
                    side="SHORT",
                    entry=entry_price,
                    sl=sl_price,
                    tp=tp_price,
                    size=size_units,
                    equity_before=equity,
                    equity_after=equity_after,
                    exit_time=t,
                    exit_price=exit_price_exec,
                    reason=reason,
                    r_multiple=r_mult
                ))

                equity = equity_after
                # reset position
                position = None
                entry_idx = None
                entry_price = None
                sl_price = None
                tp_price = None
                size_units = 0
                risk_amount = 0.0
                be_armed = False
                partial_done = False
                highest_since_entry = None
                entry_notional = 0.0
                entry_fees = 0.0

                continue  # next bar

        # si pas de position, chercher signal
        if position is None and take_shorts and not pass_for_time and i > 30:
            # ATR filter
            if (atr_val / pips) < atr_min_pips:
                continue

            # triple confluence
            sw = had_bear_sweep(df, i-1, lookback=30)  # sweep juste avant
            bos = bear_bos(df, i, lookback=30)
            fvg = bear_fvg(df, i-1)  # FVG autour du signal

            if require_confluence and (sw is None or bos is None or fvg is None):
                continue

            # Entry logique: SHORT sur pullback vers le haut du FVG (ou à l'ouverture suivante si indispo)
            entry_px = min(df.loc[i+1, "open"], fvg[1]) if i+1 < len(df) else fvg[1]
            # SL: au-dessus de la mèche structurelle (max du swing high sweep) + buffer
            struct_high = max(sw, df.loc[i-1, "high"]) if sw is not None else df.loc[i-1, "high"]
            raw_sl = struct_high + (buffer_sl_pips * pips)
            stop_pips = max((entry_px - raw_sl) / pips * -1.0, 0)  # distance SL en pips (positive)
            if stop_pips < min_stop_pips:
                # SL trop proche → ignore
                continue

            # TP à RR target (2R par défaut) basé sur distance SL
            tp_px = entry_px - rr_target * (entry_px - raw_sl)

            # risque et taille
            risk_amount = equity * risk_per_trade
            money_per_pip = risk_amount / (stop_pips * pips)
            units_float = money_per_pip  # pour EURUSD: 1 pip par unité ~ 0.0001 * units en $; approximation
            # contrainte levier
            notional = units_float * entry_px
            max_notional = equity * leverage_max
            if notional > max_notional:
                scale = max_notional / notional
                units_float *= scale

            # arrondi lot
            units = int(max(lot_min, (units_float // lot_step) * lot_step))
            if units <= 0:
                continue

            # appliquer spread/slippage à l'entrée (short → on vend au bid ≈ open - spread)
            entry_exec = entry_px - (spread_pips + slip_entry_pips) * pips
            entry_notional = units * entry_exec
            entry_fees = commission_cost(entry_notional, commission_per_million)

            # set position
            position = "SHORT"
            entry_idx = i+1 if i+1 < len(df) else i
            entry_price = entry_exec
            sl_price = raw_sl
            tp_price = tp_px
            size_units = units
            be_armed = False
            partial_done = False
            highest_since_entry = entry_exec

    # Fin: si position ouverte, on force une sortie à la dernière close
    if position == "SHORT":
        t = df.loc[len(df)-1, "time"]
        exit_price_exec = df.loc[len(df)-1, "close"] + slip_exit_pips * pips
        exit_notional = size_units * exit_price_exec
        exit_fees = commission_cost(exit_notional, commission_per_million)
        pnl = (entry_price - exit_price_exec) * size_units - entry_fees - exit_fees
        equity_after = equity + pnl
        r_mult = (entry_price - exit_price_exec) / (entry_price - sl_price) if (entry_price - sl_price) != 0 else 0.0
        trades.append(Trade(
            time_enter=df.loc[entry_idx, "time"], side="SHORT",
            entry=entry_price, sl=sl_price, tp=tp_price, size=size_units,
            equity_before=equity, equity_after=equity_after,
            exit_time=t, exit_price=exit_price_exec, reason="Timeout", r_multiple=r_mult
        ))
        equity = equity_after

    # métriques
    wins = [t for t in trades if t.reason in ("TP", "Trail")]
    losses = [t for t in trades if t.reason == "SL"]
    n = len(trades)
    winrate = (len(wins) / n * 100.0) if n else 0.0
    avg_R = np.mean([t.r_multiple for t in trades]) if trades else 0.0

    # max drawdown approximatif via equity steps
    eq_curve = [capital]
    for t in trades:
        eq_curve.append(t.equity_after)
    dd = 0.0
    peak = eq_curve[0] if eq_curve else capital
    for v in eq_curve:
        if v > peak:
            peak = v
        dd = max(dd, (peak - v) / peak * 100.0)
    results = {
        "Biais HTF": "bear" if take_shorts else bias,
        "final_equity": equity,
        "return_pct": (equity / capital - 1.0) * 100.0,
        "trades": n,
        "winrate_pct": winrate,
        "avg_R": avg_R,
        "max_drawdown_pct": dd,
    }
    return results, trades


def load_csv_ltf(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # columns: time,open,high,low,close
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV LTF: time,open,high,low,close (UTC)")
    ap.add_argument("--symbol", default="EUR/USD")
    ap.add_argument("--ltf", default="15min")
    ap.add_argument("--htf", default="4h")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--risk_per_trade", type=float, default=0.005)
    ap.add_argument("--spread_pips", type=float, default=0.2)
    ap.add_argument("--slip_entry_pips", type=float, default=0.1)
    ap.add_argument("--slip_exit_pips", type=float, default=0.1)
    ap.add_argument("--commission_per_million", type=float, default=7.0)
    ap.add_argument("--leverage_max", type=int, default=30)
    ap.add_argument("--lot_min", type=int, default=1000)
    ap.add_argument("--lot_step", type=int, default=1000)
    ap.add_argument("--min_stop_pips", type=float, default=1.2)
    ap.add_argument("--atr_period", type=int, default=14)
    ap.add_argument("--atr_alpha", type=float, default=0.4)
    ap.add_argument("--atr_min_pips", type=float, default=0.8)
    ap.add_argument("--require_confluence", action="store_true")
    ap.add_argument("--use_partials", action="store_true")
    ap.add_argument("--buffer_sl_pips", type=float, default=0.2)
    ap.add_argument("--rr_target", type=float, default=2.0)
    ap.add_argument("--partial_take_r", type=float, default=1.5)
    ap.add_argument("--move_be_r", type=float, default=1.0)
    ap.add_argument("--trail_struct_window", type=int, default=20)
    ap.add_argument("--pending_ttl_bars", type=int, default=24)
    ap.add_argument("--forbidden_hours_utc", default="2,3,4,16,17")
    args = ap.parse_args()

    df = load_csv_ltf(args.csv)

    forbidden_hours = tuple(int(x) for x in str(args.forbidden_hours_utc).split(",") if x != "")

    results, trades = backtest_smc(
        df_ltf=df,
        symbol=args.symbol,
        ltf=args.ltf,
        htf=args.htf,
        start=args.start,
        end=args.end,
        capital=args.capital,
        risk_per_trade=args.risk_per_trade,
        spread_pips=args.spread_pips,
        slip_entry_pips=args.slip_entry_pips,
        slip_exit_pips=args.slip_exit_pips,
        commission_per_million=args.commission_per_million,
        leverage_max=args.leverage_max,
        lot_min=args.lot_min,
        lot_step=args.lot_step,
        min_stop_pips=args.min_stop_pips,
        atr_period=args.atr_period,
        atr_alpha=args.atr_alpha,
        atr_min_pips=args.atr_min_pips,
        require_confluence=args.require_confluence,
        use_partials=args.use_partials,
        buffer_sl_pips=args.buffer_sl_pips,
        rr_target=args.rr_target,
        partial_take_r=args.partial_take_r,
        move_be_r=args.move_be_r,
        trail_struct_window=args.trail_struct_window,
        pending_ttl_bars=args.pending_ttl_bars,
        forbidden_hours_utc=forbidden_hours,
    )

    # print trades
    for tr in trades:
        print(f"[{tr.time_enter}] Enter {tr.side} @ {tr.entry:.5f} SL {tr.sl:.5f} TP {tr.tp:.5f} size {int(tr.size):,}")
        print(f"[{tr.exit_time}] Exit {tr.reason} @ {tr.exit_price:.5f} | Equity: {tr.equity_after:.2f}")

    print("\n===== RÉSULTATS V2 =====")
    for k, v in results.items():
        if isinstance(v, float):
            if "pct" in k:
                print(f"{k}: {v:.2f}%")
            else:
                print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()

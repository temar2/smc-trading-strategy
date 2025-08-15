## smc_backtest_v2.5.py  (v2.5.2 stable)
# TwelveData fetch (LTF+HTF) -> CSV -> SMC backtest
# Ajouts: --max_lot (cap lots), --hard_min_stop_pips (SL min absolu)
# + support --pending_ttl_bars (optionnel) et fenêtrage --session_start_utc/--session_end_utc
# Risk sizing propre, levier, commission, spread+slippage, partials, BE, trailing structure

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
        "timezone": "UTC",
        "order": "ASC",
    }
    if start: params["start_date"] = start
    if end:   params["end_date"] = end

    print(f"📡 Fetch {symbol} {interval} {start} → {end} ...")
    r = requests.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Fallback: décale end d'1 jour si vide
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
    # JPY: 1 pip = 0.01
    if s.endswith("JPY"):
        return 1e-2
    # Métaux (approx broker): 0.1
    if "XAU" in s or "XAG" in s:
        return 1e-1
    # Crypto: on met 0.01 par "pip" pour éviter des tailles absurdes
    if "BTC" in s or "ETH" in s:
        return 1e-2
    # FX classique 5 décimales: 0.0001
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
    """Détecte un liquidity sweep bearish avec validation rejection"""
    lo = max(0, i-lookback)
    if i <= lo: return None

    recent_high = df["high"].iloc[lo:i].max()
    current_high = df["high"].iat[i]
    current_close = df["close"].iat[i]
    current_open = df["open"].iat[i]

    # Sweep: high dépasse recent_high mais close rejette en dessous
    if current_high > recent_high and current_close < recent_high:
        # Validation rejection: wick significatif (>30% de la range)
        candle_range = current_high - min(current_open, current_close)
        upper_wick = current_high - max(current_open, current_close)
        if candle_range > 0 and (upper_wick / candle_range) > 0.3:
            return recent_high
    return None

def bear_bos(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    """Détecte une cassure de structure bearish avec validation momentum"""
    lo = max(0, i-lookback)
    if i <= lo: return None

    recent_low = df["low"].iloc[lo:i].min()
    current_low = df["low"].iat[i]
    current_close = df["close"].iat[i]

    # Cassure confirmée si: low ET close cassent la structure
    if current_low < recent_low and current_close < recent_low:
        # Validation momentum: bougie bearish
        if df["close"].iat[i] < df["open"].iat[i]:
            return recent_low
    return None

def bear_fvg(df: pd.DataFrame, i: int, min_gap_pips: float = 0.5) -> Optional[Tuple[float,float]]:
    """Détecte un FVG bearish: 3-candle pattern SANS look-ahead bias"""
    if i < 2: return None  # Besoin de 2 bougies précédentes minimum

    # Pattern correct: candle[i-2] high < candle[i] low (gap avec candle[i-1] au milieu)
    high_before = df["high"].iat[i-2]
    low_current = df["low"].iat[i]

    # Vérifier que la bougie du milieu ne comble pas le gap
    middle_high = df["high"].iat[i-1]
    middle_low = df["low"].iat[i-1]

    if high_before < low_current and middle_high < low_current and middle_low > high_before:
        gap_size = (low_current - high_before) * 10000  # en pips
        if gap_size >= min_gap_pips:
            return (high_before, low_current)  # zone FVG [high_before, low_current]
    return None

# Bull signals
def had_bull_sweep(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    """Détecte un liquidity sweep bullish avec validation rejection"""
    lo = max(0, i-lookback)
    if i <= lo: return None

    recent_low = df["low"].iloc[lo:i].min()
    current_low = df["low"].iat[i]
    current_close = df["close"].iat[i]
    current_open = df["open"].iat[i]

    # Sweep: low dépasse recent_low mais close rejette au dessus
    if current_low < recent_low and current_close > recent_low:
        # Validation rejection: wick significatif (>30% de la range)
        candle_range = max(current_open, current_close) - current_low
        lower_wick = min(current_open, current_close) - current_low
        if candle_range > 0 and (lower_wick / candle_range) > 0.3:
            return recent_low
    return None

def bull_bos(df: pd.DataFrame, i: int, lookback: int = 20) -> Optional[float]:
    """Détecte une cassure de structure bullish avec validation momentum"""
    lo = max(0, i-lookback)
    if i <= lo: return None

    recent_high = df["high"].iloc[lo:i].max()
    current_high = df["high"].iat[i]
    current_close = df["close"].iat[i]

    # Cassure confirmée si: high ET close cassent la structure
    if current_high > recent_high and current_close > recent_high:
        # Validation momentum: bougie bullish
        if df["close"].iat[i] > df["open"].iat[i]:
            return recent_high
    return None

def bull_fvg(df: pd.DataFrame, i: int, min_gap_pips: float = 0.5) -> Optional[Tuple[float,float]]:
    """Détecte un FVG bullish: 3-candle pattern SANS look-ahead bias"""
    if i < 2: return None  # Besoin de 2 bougies précédentes minimum

    # Pattern correct: candle[i-2] low > candle[i] high (gap avec candle[i-1] au milieu)
    low_before = df["low"].iat[i-2]
    high_current = df["high"].iat[i]

    # Vérifier que la bougie du milieu ne comble pas le gap
    middle_high = df["high"].iat[i-1]
    middle_low = df["low"].iat[i-1]

    if low_before > high_current and middle_low > high_current and middle_high < low_before:
        gap_size = (low_before - high_current) * 10000  # en pips
        if gap_size >= min_gap_pips:
            return (high_current, low_before)  # zone FVG [high_current, low_before]
    return None

# Order Blocks Detection
def bear_order_block(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[Tuple[float,float]]:
    """Détecte un order block bearish: dernière bougie bullish avant cassure bearish"""
    if i < lookback: return None

    # Chercher la dernière bougie bullish significative avant une chute
    for j in range(i-1, max(0, i-lookback), -1):
        if df["close"].iat[j] > df["open"].iat[j]:  # bougie bullish
            # Vérifier qu'il y a eu une chute après
            if df["close"].iat[i] < df["low"].iat[j]:
                body_size = abs(df["close"].iat[j] - df["open"].iat[j])
                if body_size > 0:  # bougie avec du corps
                    return (df["low"].iat[j], df["high"].iat[j])  # zone OB
    return None

def bull_order_block(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[Tuple[float,float]]:
    """Détecte un order block bullish: dernière bougie bearish avant cassure bullish"""
    if i < lookback: return None

    # Chercher la dernière bougie bearish significative avant une montée
    for j in range(i-1, max(0, i-lookback), -1):
        if df["close"].iat[j] < df["open"].iat[j]:  # bougie bearish
            # Vérifier qu'il y a eu une montée après
            if df["close"].iat[i] > df["high"].iat[j]:
                body_size = abs(df["close"].iat[j] - df["open"].iat[j])
                if body_size > 0:  # bougie avec du corps
                    return (df["low"].iat[j], df["high"].iat[j])  # zone OB
    return None

def count_confluence_bear(df, i) -> Tuple[int, float, Tuple[Optional[float],Optional[float],Optional[Tuple[float,float]],Optional[Tuple[float,float]]]]:
    """Retourne confluence count + quality score + signaux"""
    sw = had_bear_sweep(df, i-1)
    bos = bear_bos(df, i)
    fvg = bear_fvg(df, i-1)
    ob = bear_order_block(df, i)

    # Calcul du score de qualité (0.0 à 1.0)
    quality_score = 0.0
    count = 0

    if sw is not None:
        count += 1
        quality_score += 0.3  # Sweep = signal fort
    if bos is not None:
        count += 1
        quality_score += 0.4  # BOS = signal très fort
    if fvg is not None:
        count += 1
        quality_score += 0.2  # FVG = signal moyen
    if ob is not None:
        count += 1
        quality_score += 0.1  # OB = signal faible

    # Bonus pour confluence multiple
    if count >= 3:
        quality_score += 0.2
    elif count >= 2:
        quality_score += 0.1

    quality_score = min(1.0, quality_score)  # Cap à 1.0

    return count, quality_score, (sw, bos, fvg, ob)

def count_confluence_bull(df, i) -> Tuple[int, float, Tuple[Optional[float],Optional[float],Optional[Tuple[float,float]],Optional[Tuple[float,float]]]]:
    """Retourne confluence count + quality score + signaux"""
    sw = had_bull_sweep(df, i-1)
    bos = bull_bos(df, i)
    fvg = bull_fvg(df, i-1)
    ob = bull_order_block(df, i)

    # Calcul du score de qualité (0.0 à 1.0)
    quality_score = 0.0
    count = 0

    if sw is not None:
        count += 1
        quality_score += 0.3  # Sweep = signal fort
    if bos is not None:
        count += 1
        quality_score += 0.4  # BOS = signal très fort
    if fvg is not None:
        count += 1
        quality_score += 0.2  # FVG = signal moyen
    if ob is not None:
        count += 1
        quality_score += 0.1  # OB = signal faible

    # Bonus pour confluence multiple
    if count >= 3:
        quality_score += 0.2
    elif count >= 2:
        quality_score += 0.1

    quality_score = min(1.0, quality_score)  # Cap à 1.0

    return count, quality_score, (sw, bos, fvg, ob)

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
    max_lot: float = 2.0,  # NEW: cap lots/trade
    spread_pips: float = 0.2, slip_entry_pips: float = 0.1, slip_exit_pips: float = 0.1,
    commission_per_million: float = 7.0,
    forbidden_hours_utc: Tuple[int,...] = (0,2,3,4,16,17,23),
    session_start_utc: Optional[int] = None, session_end_utc: Optional[int] = None,  # NEW: fenêtre
    atr_period: int = 14, atr_alpha: float = 0.4, atr_min_pips: float = 1.0,
    min_stop_pips: float = 1.6, hard_min_stop_pips: float = 2.0,  # NEW: SL min absolu
    require_confluence: bool = True,
    buffer_sl_pips: float = 0.4,
    rr_target: float = 2.0, rr_target_alt: float = 1.3, atr_rr_switch_pips: float = 1.1,
    use_partials: bool = True, partial_take_r: float = 1.5, move_be_r: float = 1.0,
    trail_struct_window: int = 20,
    auto_bias: bool = True, force_bias: Optional[str] = None,
    confluence_min: int = 2,
    pullback_bars: int = 3, pullback_optional: bool = True, fallback_market_bars: int = 2,
    allow_countertrend: bool = True, ct_risk_factor: float = 0.5, ct_rr_cap: float = 1.2,
    momentum_min_body_atr: float = 0.25, news_free_minutes: int = 30,
    pending_ttl_bars: Optional[int] = None,  # accepté, non utilisé ici (placeholder)
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
    risk_amount_open = 0.0

    # Historique des trades pour Kelly Criterion (en % de return)
    trade_returns_history = []

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

    # --- sizing helpers ---
    def apply_caps_and_fees(units: int, entry_exec: float) -> Tuple[int, float]:
        # Cap lots
        max_units_by_lot = int(max_lot * 100_000) if max_lot and max_lot > 0 else None
        if max_units_by_lot:
            units = min(units, max_units_by_lot)
        # Arrondi par pas
        units = int(max(lot_min, ((units // lot_step) * lot_step)))
        if units <= 0:
            return 0, 0.0
        # Fees (à l’entrée)
        entry_notional = units * entry_exec
        fees = commission_cost(entry_notional, commission_per_million)
        return units, fees

    def calculate_kelly_fraction(trade_history: list, lookback: int = 20) -> float:
        """Calcule la fraction Kelly optimale basée sur l'historique récent"""
        if len(trade_history) < 5:
            return 1.0  # Pas assez d'historique, utiliser allocation normale

        # Prendre les derniers trades
        recent_trades = trade_history[-lookback:] if len(trade_history) >= lookback else trade_history

        # Calculer win rate et avg win/loss
        wins = [t for t in recent_trades if t > 0]
        losses = [abs(t) for t in recent_trades if t < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 1.0

        win_rate = len(wins) / len(recent_trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return 1.0

        # Kelly Fraction = (bp - q) / b
        # où b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly_fraction = (b * p - q) / b

        # Limiter Kelly entre 0.1 et 2.0 pour éviter les extrêmes
        kelly_fraction = max(0.1, min(2.0, kelly_fraction))

        return kelly_fraction

    def detect_volatility_breakout(i: int, lookback: int = 20) -> float:
        """Détecte les breakouts de volatilité et retourne un multiplicateur"""
        if i < lookback:
            return 1.0

        # ATR actuel vs moyenne des 20 dernières périodes
        current_atr = df.at[i, "ATR"]
        avg_atr = df["ATR"].iloc[i-lookback:i].mean()

        if avg_atr == 0:
            return 1.0

        volatility_ratio = current_atr / avg_atr

        # Multiplicateur basé sur la volatilité
        if volatility_ratio > 1.5:  # Haute volatilité
            return 1.5  # Augmenter le risque
        elif volatility_ratio > 1.2:  # Volatilité modérée
            return 1.2
        elif volatility_ratio < 0.7:  # Faible volatilité
            return 0.7  # Réduire le risque
        else:
            return 1.0  # Volatilité normale

    def detect_market_regime(i: int, lookback: int = 50) -> str:
        """Détecte le régime de marché: TREND, RANGE, BREAKOUT"""
        if i < lookback:
            return "NEUTRAL"

        # Calculer la range des 50 dernières bougies
        high_max = df["high"].iloc[i-lookback:i].max()
        low_min = df["low"].iloc[i-lookback:i].min()
        range_size = high_max - low_min

        # Prix actuel par rapport à la range
        current_price = df.at[i, "close"]
        range_position = (current_price - low_min) / range_size if range_size > 0 else 0.5

        # ATR pour mesurer la volatilité
        current_atr = df.at[i, "ATR"]
        avg_atr = df["ATR"].iloc[i-lookback:i].mean()
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Classification du régime
        if volatility_ratio > 1.3:
            return "BREAKOUT"  # Haute volatilité = breakout
        elif range_position > 0.8 or range_position < 0.2:
            return "TREND"  # Prix aux extrêmes = trend
        else:
            return "RANGE"  # Prix au milieu = range

    def is_high_impact_session(i: int) -> bool:
        """Vérifie si nous sommes dans une session à fort impact"""
        current_time = df.at[i, "time"]
        hour_utc = current_time.hour

        # Sessions à fort impact (UTC)
        london_open = 8 <= hour_utc <= 10  # London Open
        ny_open = 13 <= hour_utc <= 15     # NY Open
        london_ny_overlap = 13 <= hour_utc <= 16  # Overlap London/NY

        return london_open or ny_open or london_ny_overlap

    def get_session_multiplier(i: int) -> float:
        """Retourne un multiplicateur basé sur la session de trading"""
        current_time = df.at[i, "time"]
        hour_utc = current_time.hour

        # Sessions et leurs multiplicateurs
        if 13 <= hour_utc <= 16:  # London/NY Overlap - Le plus actif
            return 1.4
        elif 8 <= hour_utc <= 12:  # London Session
            return 1.2
        elif 13 <= hour_utc <= 17:  # NY Session
            return 1.2
        elif 0 <= hour_utc <= 3:   # Sydney Session
            return 0.9
        elif 23 <= hour_utc <= 24 or 0 <= hour_utc <= 8:  # Asian Session
            return 0.8
        else:  # Autres heures
            return 0.7

    def get_htf_trend_strength(i: int) -> float:
        """Calcule la force du trend HTF (0.0 à 1.0)"""
        current_time = df.at[i, "time"]

        # Trouver la bougie HTF correspondante
        htf_idx = None
        for idx, htf_time in enumerate(df_htf["time"]):
            if htf_time <= current_time:
                htf_idx = idx
            else:
                break

        if htf_idx is None or htf_idx < 10:
            return 0.5  # Neutre si pas assez de données

        # Calculer la pente de la MA sur HTF
        htf_closes = df_htf["close"].iloc[htf_idx-9:htf_idx+1]  # 10 dernières bougies HTF

        if len(htf_closes) < 10:
            return 0.5

        # Régression linéaire simple pour la pente
        x = list(range(len(htf_closes)))
        y = htf_closes.values

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.5

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # Normaliser la pente en force de trend (0.0 à 1.0)
        avg_price = sum_y / n
        normalized_slope = abs(slope) / (avg_price * 0.001)  # Normaliser par rapport au prix

        trend_strength = min(1.0, normalized_slope)

        return trend_strength

    def htf_confirms_direction(i: int, direction: str) -> bool:
        """Vérifie si le HTF confirme la direction du trade"""
        htf_bias = df.at[i, "HTF_bias"]
        trend_strength = get_htf_trend_strength(i)

        # Confirmation forte si bias aligné ET trend fort
        if direction == "SHORT":
            return htf_bias == "bearish" and trend_strength > 0.6
        elif direction == "LONG":
            return htf_bias == "bullish" and trend_strength > 0.6

        return False

    # === inner entry funcs ===
    def try_short_entry(i: int, rr_use: float, use_ct=False) -> bool:
        nonlocal position, entry_idx, entry_price, sl_price, tp_price, size_units, be_armed, partial_done, init_sl_price, entry_fees, equity, risk_amount_open
        conf, quality_score, (sw, bos, fvg, ob) = count_confluence_bear(df, i)
        if conf < max(1, confluence_min) and not use_ct:
            return False

        fvg_low, fvg_high = fvg if fvg else (None, None)
        ob_low, ob_high = ob if ob else (None, None)
        entered = False
        waited = 0
        target_rr = min(rr_use, ct_rr_cap) if use_ct else rr_use

        # ALLOCATION ADAPTATIVE basée sur qualité + Kelly + Volatilité + Régime
        base_risk_factor = (ct_risk_factor if use_ct else 1.0)
        quality_multiplier = 0.5 + (quality_score * 1.5)  # Range: 0.5x à 2.0x
        kelly_multiplier = calculate_kelly_fraction(trade_returns_history)
        volatility_multiplier = detect_volatility_breakout(i)

        # Ajustement selon le régime de marché
        market_regime = detect_market_regime(i)
        if market_regime == "BREAKOUT":
            regime_multiplier = 1.3  # Plus agressif en breakout
        elif market_regime == "TREND":
            regime_multiplier = 1.1  # Légèrement plus agressif en trend
        elif market_regime == "RANGE":
            regime_multiplier = 0.8  # Plus conservateur en range
        else:
            regime_multiplier = 1.0

        # Multiplicateur de session
        session_multiplier = get_session_multiplier(i)

        # Multiplicateur HTF confluence
        htf_multiplier = 1.0
        if htf_confirms_direction(i, "SHORT"):
            htf_multiplier = 1.3  # Bonus pour confluence HTF

        risk_factor = base_risk_factor * quality_multiplier * kelly_multiplier * volatility_multiplier * regime_multiplier * session_multiplier * htf_multiplier

        # Attente mitigation (FVG prioritaire, puis Order Block)
        for j in range(1, pullback_bars+1):
            if i+j >= len(df): break

            # Priorité 1: FVG mitigation avec confirmation de rejection
            if fvg_high is not None and df.at[i+j,"high"] >= fvg_high:
                # Confirmation: close doit rejeter en dessous du FVG + bougie bearish
                if (df.at[i+j,"close"] < fvg_high and
                    df.at[i+j,"close"] < df.at[i+j,"open"]):  # bougie bearish
                    entry_exec = fvg_high - (spread_pips + slip_entry_pips) * pips
                    waited = j; entered = True; break

            # Priorité 2: Order Block mitigation avec confirmation
            elif ob_high is not None and df.at[i+j,"high"] >= ob_high and df.at[i+j,"low"] <= ob_low:
                # Confirmation: bougie bearish dans l'OB
                if df.at[i+j,"close"] < df.at[i+j,"open"]:
                    entry_exec = (ob_high + ob_low) / 2 - (spread_pips + slip_entry_pips) * pips
                    waited = j; entered = True; break

        # fallback optionnel
        if not entered and pullback_optional:
            fb = fallback_market_bars
            if fb > 0 and i+fb < len(df):
                entry_exec = df.at[i+fb,"open"] - (spread_pips + slip_entry_pips) * pips
                waited = fb; entered = True

        if not entered:
            return False

        # Structure & SL/TP init avec plus de marge
        struct_high = max((sw or df.at[i-1,"high"]), df.at[i-1,"high"])
        raw_sl = struct_high + (buffer_sl_pips + 2.0) * pips  # +2 pips de marge anti-stop hunt
        stop_dist = raw_sl - entry_exec
        stop_pips_val = stop_dist / pips
        if stop_dist <= 0:
            return False

        # --- Hard min stop pips (priorité) ---
        if stop_pips_val < hard_min_stop_pips:
            return False
        # Et filtre min_stop / atr_min ensuite
        if stop_pips_val < max(min_stop_pips, atr_min_pips * 0.6):
            return False

        # TP initial
        tp_px_calc = entry_exec - target_rr * stop_dist

        # Sanity spread/slip margin
        min_margin = (spread_pips + slip_entry_pips + slip_exit_pips) * pips
        if raw_sl <= entry_exec + min_margin:
            raw_sl = entry_exec + max(min_margin, buffer_sl_pips * pips)
            stop_dist = raw_sl - entry_exec
            stop_pips_val = stop_dist / pips
            if stop_dist <= 0: return False
            if stop_pips_val < hard_min_stop_pips: return False

        if tp_px_calc >= entry_exec - min_margin:
            tp_px_calc = entry_exec - max(min_margin, rr_target_alt * min_margin)

        if not (raw_sl > entry_exec and tp_px_calc < entry_exec):
            return False

        # Sizing
        risk_amount_at_entry = equity * risk_per_trade * risk_factor
        if stop_dist <= 0: return False
        units_calc = risk_amount_at_entry / stop_dist
        # Leverage cap
        notional = units_calc * entry_exec
        max_notional = equity * leverage_max
        if max_notional > 0 and notional > max_notional:
            scale = max_notional / notional
            units_calc *= max(0.0, scale)
        # Arrondir au lieu de tronquer pour préserver le sizing
        units = round(units_calc)
        # caps + fees
        units, fees = apply_caps_and_fees(units, entry_exec)
        if units <= 0: return False
        equity -= fees

        # Commit
        position = "SHORT"; entry_idx = i+waited; entry_price = entry_exec
        sl_price = raw_sl; tp_price = tp_px_calc; size_units = units
        be_armed = False; partial_done = False; init_sl_price = raw_sl; entry_fees = fees
        risk_amount_open = risk_amount_at_entry
        return True

    def try_long_entry(i: int, rr_use: float, use_ct=False) -> bool:
        nonlocal position, entry_idx, entry_price, sl_price, tp_price, size_units, be_armed, partial_done, init_sl_price, entry_fees, equity, risk_amount_open
        conf, quality_score, (sw, bos, fvg, ob) = count_confluence_bull(df, i)
        if conf < max(1, confluence_min) and not use_ct:
            return False

        fvg_lo, fvg_hi = fvg if fvg else (None, None)
        ob_low, ob_high = ob if ob else (None, None)
        entered = False
        waited = 0
        target_rr = min(rr_use, ct_rr_cap) if use_ct else rr_use

        # ALLOCATION ADAPTATIVE basée sur qualité + Kelly + Volatilité + Régime
        base_risk_factor = (ct_risk_factor if use_ct else 1.0)
        quality_multiplier = 0.5 + (quality_score * 1.5)  # Range: 0.5x à 2.0x
        kelly_multiplier = calculate_kelly_fraction(trade_returns_history)
        volatility_multiplier = detect_volatility_breakout(i)

        # Ajustement selon le régime de marché
        market_regime = detect_market_regime(i)
        if market_regime == "BREAKOUT":
            regime_multiplier = 1.3  # Plus agressif en breakout
        elif market_regime == "TREND":
            regime_multiplier = 1.1  # Légèrement plus agressif en trend
        elif market_regime == "RANGE":
            regime_multiplier = 0.8  # Plus conservateur en range
        else:
            regime_multiplier = 1.0

        # Multiplicateur de session
        session_multiplier = get_session_multiplier(i)

        # Multiplicateur HTF confluence
        htf_multiplier = 1.0
        if htf_confirms_direction(i, "LONG"):
            htf_multiplier = 1.3  # Bonus pour confluence HTF

        risk_factor = base_risk_factor * quality_multiplier * kelly_multiplier * volatility_multiplier * regime_multiplier * session_multiplier * htf_multiplier

        # Attente mitigation (FVG prioritaire, puis Order Block)
        for j in range(1, pullback_bars+1):
            if i+j >= len(df): break

            # Priorité 1: FVG mitigation avec confirmation de rejection
            if fvg_lo is not None and df.at[i+j,"low"] <= fvg_lo:
                # Confirmation: close doit rejeter au dessus du FVG + bougie bullish
                if (df.at[i+j,"close"] > fvg_lo and
                    df.at[i+j,"close"] > df.at[i+j,"open"]):  # bougie bullish
                    entry_exec = fvg_lo + (spread_pips + slip_entry_pips) * pips
                    waited = j; entered = True; break

            # Priorité 2: Order Block mitigation avec confirmation
            elif ob_low is not None and df.at[i+j,"low"] <= ob_low and df.at[i+j,"high"] >= ob_high:
                # Confirmation: bougie bullish dans l'OB
                if df.at[i+j,"close"] > df.at[i+j,"open"]:
                    entry_exec = (ob_high + ob_low) / 2 + (spread_pips + slip_entry_pips) * pips
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

        # --- Hard min stop pips (priorité) ---
        if stop_pips_val < hard_min_stop_pips:
            return False
        # Et filtre min_stop / atr_min ensuite
        if stop_pips_val < max(min_stop_pips, atr_min_pips * 0.6):
            return False

        tp_px_calc = entry_exec + target_rr * stop_dist

        # Sanity spread/slip margin
        min_margin = (spread_pips + slip_entry_pips + slip_exit_pips) * pips
        if raw_sl >= entry_exec - min_margin:
            raw_sl = entry_exec - max(min_margin, buffer_sl_pips * pips)
            stop_dist = entry_exec - raw_sl
            stop_pips_val = stop_dist / pips
            if stop_dist <= 0: return False
            if stop_pips_val < hard_min_stop_pips: return False

        if tp_px_calc <= entry_exec + min_margin:
            tp_px_calc = entry_exec + max(min_margin, rr_target_alt * min_margin)

        if not (raw_sl < entry_exec and tp_px_calc > entry_exec):
            return False

        # Sizing
        risk_amount_at_entry = equity * risk_per_trade * risk_factor
        if stop_dist <= 0: return False
        units_calc = risk_amount_at_entry / stop_dist
        # Leverage cap
        notional = units_calc * entry_exec
        max_notional = equity * leverage_max
        if max_notional > 0 and notional > max_notional:
            scale = max_notional / notional
            units_calc *= max(0.0, scale)
        # Arrondir au lieu de tronquer pour préserver le sizing
        units = round(units_calc)
        # caps + fees
        units, fees = apply_caps_and_fees(units, entry_exec)
        if units <= 0: return False
        equity -= fees

        # Commit
        position = "LONG"; entry_idx = i+waited; entry_price = entry_exec
        sl_price = raw_sl; tp_price = tp_px_calc; size_units = units
        be_armed = False; partial_done = False; init_sl_price = raw_sl; entry_fees = fees
        risk_amount_open = risk_amount_at_entry
        return True

    # === main loop ===
    for i in range(30, len(df)-1):  # i+1 needed
        t = df.at[i, "time"]
        # Fenêtre de session (si fournie)
        if session_start_utc is not None and session_end_utc is not None:
            h = t.hour
            if session_start_utc <= session_end_utc:
                if not (session_start_utc <= h <= session_end_utc):
                    continue
            else:
                # fenêtre qui passe minuit (ex: 22→5)
                if not (h >= session_start_utc or h <= session_end_utc):
                    continue

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
            if 'locked_bias' not in locals() or locked_bias is None:
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

                # Enregistrer le return pour Kelly Criterion
                trade_return_pct = (equity_after - equity) / equity * 100  # en %
                trade_returns_history.append(trade_return_pct)
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

            if bias_here == "bear":
                entered_main = try_short_entry(i, rr_use, use_ct=False)
                if not entered_main and allow_countertrend:
                    try_long_entry(i, rr_use, use_ct=True)
            elif bias_here == "bull":
                entered_main = try_long_entry(i, rr_use, use_ct=False)
                if not entered_main and allow_countertrend:
                    try_short_entry(i, rr_use, use_ct=True)

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

        # Enregistrer le return pour Kelly Criterion
        trade_return_pct = (equity_after - equity) / equity * 100  # en %
        trade_returns_history.append(trade_return_pct)
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
    ap.add_argument("--max_lot", type=float, default=2.0, help="Max lots per trade (100k units per lot)")

    # Costs
    ap.add_argument("--spread_pips", type=float, default=0.2)
    ap.add_argument("--slip_entry_pips", type=float, default=0.1)
    ap.add_argument("--slip_exit_pips", type=float, default=0.1)
    ap.add_argument("--commission_per_million", type=float, default=7.0)

    # Filters
    ap.add_argument("--forbidden_hours_utc", type=str, default="0,2,3,4,16,17,23")
    ap.add_argument("--session_start_utc", type=int, default=None)
    ap.add_argument("--session_end_utc", type=int, default=None)
    ap.add_argument("--atr_period", type=int, default=14)
    ap.add_argument("--atr_alpha", type=float, default=0.4)
    ap.add_argument("--atr_min_pips", type=float, default=1.0)
    ap.add_argument("--min_stop_pips", type=float, default=1.6)
    ap.add_argument("--hard_min_stop_pips", type=float, default=2.0, help="Absolute minimum SL in pips")
    ap.add_argument("--require_confluence", action="store_true")

    # Management
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

    # v2.5 – switchs
    ap.add_argument("--confluence_min", type=int, default=2, help="Min among Sweep/BOS/FVG")
    ap.add_argument("--pullback_bars", type=int, default=3, help="Bars to wait for mitigation")
    ap.add_argument("--pullback_optional", action="store_true", help="If no mitigation, fallback market")
    ap.add_argument("--fallback_market_bars", type=int, default=2, help="Bars before fallback market")
    ap.add_argument("--allow_countertrend", action="store_true", help="Allow countertrend scalp")
    ap.add_argument("--ct_risk_factor", type=float, default=0.5, help="Risk fraction for CT trades")
    ap.add_argument("--ct_rr_cap", type=float, default=1.2, help="Max RR for CT trades")
    ap.add_argument("--momentum_min_body_atr", type=float, default=0.25, help="|close-open|/ATR min (anti-doji)")
    ap.add_argument("--news-free-minutes", type=int, default=30, help="Block entries ±N min around macro times")

    # Accept but unused here (compat)
    ap.add_argument("--pending_ttl_bars", type=int, default=None)

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
        max_lot=args.max_lot,
        spread_pips=args.spread_pips, slip_entry_pips=args.slip_entry_pips, slip_exit_pips=args.slip_exit_pips,
        commission_per_million=args.commission_per_million,
        forbidden_hours_utc=forbidden_hours,
        session_start_utc=args.session_start_utc, session_end_utc=args.session_end_utc,
        atr_period=args.atr_period, atr_alpha=args.atr_alpha, atr_min_pips=args.atr_min_pips,
        min_stop_pips=args.min_stop_pips, hard_min_stop_pips=args.hard_min_stop_pips,
        require_confluence=args.require_confluence,
        buffer_sl_pips=args.buffer_sl_pips,
        rr_target=args.rr_target, rr_target_alt=args.rr_target_alt, atr_rr_switch_pips=args.atr_rr_switch_pips,
        use_partials=args.use_partials, partial_take_r=args.partial_take_r, move_be_r=args.move_be_r,
        trail_struct_window=args.trail_struct_window,
        auto_bias=args.__dict__.get("auto_bias", False), force_bias=args.__dict__.get("force_bias", None),
        confluence_min=args.confluence_min,
        pullback_bars=args.pullback_bars, pullback_optional=args.pullback_optional, fallback_market_bars=args.fallback_market_bars,
        allow_countertrend=args.allow_countertrend, ct_risk_factor=args.ct_risk_factor, ct_rr_cap=args.ct_rr_cap,
        momentum_min_body_atr=args.momentum_min_body_atr, news_free_minutes=args.news_free_minutes,
        pending_ttl_bars=args.pending_ttl_bars
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

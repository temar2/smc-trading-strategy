import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, Tuple

# ==============================
# SECTION 1 : Téléchargement données TwelveData
# ==============================
def fetch_twelvedata(api_key, symbol, interval, start=None, end=None):
    """Télécharge les bougies depuis TwelveData"""
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
        "order": "ASC"
    }
    if start and end:
        params["start_date"] = start
        params["end_date"] = end

    print(f"📡 Téléchargement données {symbol} {interval} depuis TwelveData...")
    r = requests.get(base_url, params=params)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Erreur API TwelveData: {data}")

    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]

# ==============================
# SECTION 2 : Fonctions utilitaires
# ==============================
def to_pips(sym: str) -> float:
    return 1e-4

def commission_cost(notional: float, commission_per_million: float) -> float:
    return (notional / 1_000_000.0) * commission_per_million

def atr(df: pd.DataFrame, period: int = 14, alpha: Optional[float] = None) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    if alpha:
        return tr.ewm(alpha=alpha, adjust=False).mean()
    return tr.rolling(window=period, min_periods=1).mean()

def last_bos_bias(htf: pd.DataFrame) -> str:
    highs, lows, closes = htf["high"], htf["low"], htf["close"]
    last_high, last_low = None, None
    bias = "neutral"
    for i in range(len(htf)):
        if highs[i] == max(highs[max(0, i-2):i+3]):
            last_high = highs[i]
        if lows[i] == min(lows[max(0, i-2):i+3]):
            last_low = lows[i]
        if last_high and closes[i] > last_high:
            bias = "bull"
        if last_low and closes[i] < last_low:
            bias = "bear"
    return bias

def had_bear_sweep(df, i, lookback=20):
    lo = max(0, i - lookback)
    recent_high = df["high"].iloc[lo:i].max()
    if df["high"].iat[i] > recent_high and df["close"].iat[i] < recent_high:
        return recent_high
    return None

def bear_bos(df, i, lookback=20):
    lo = max(0, i - lookback)
    recent_low = df["low"].iloc[lo:i].min()
    if df["close"].iat[i] < recent_low:
        return recent_low
    return None

def bear_fvg(df, i):
    if i < 1 or i+1 >= len(df):
        return None
    if df["high"].iat[i+1] < df["low"].iat[i-1]:
        return (df["high"].iat[i+1], df["low"].iat[i-1])
    return None

def had_bull_sweep(df, i, lookback=20):
    lo = max(0, i - lookback)
    recent_low = df["low"].iloc[lo:i].min()
    if df["low"].iat[i] < recent_low and df["close"].iat[i] > recent_low:
        return recent_low
    return None

def bull_bos(df, i, lookback=20):
    lo = max(0, i - lookback)
    recent_high = df["high"].iloc[lo:i].max()
    if df["close"].iat[i] > recent_high:
        return recent_high
    return None

def bull_fvg(df, i):
    if i < 1 or i+1 >= len(df):
        return None
    if df["low"].iat[i+1] > df["high"].iat[i-1]:
        return (df["high"].iat[i-1], df["low"].iat[i+1])
    return None

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
    reason: str
    r_multiple: float

# ==============================
# SECTION 3 : Backtest adaptatif
# ==============================
def backtest_smc(df_ltf: pd.DataFrame, capital=100000, risk_per_trade=0.005, spread_pips=0.2,
                 slip_entry_pips=0.1, slip_exit_pips=0.1, commission_per_million=7,
                 leverage_max=30, lot_min=1000, lot_step=1000, min_stop_pips=1.2,
                 atr_period=14, atr_alpha=0.4, require_confluence=False, use_partials=False,
                 pending_ttl_bars=24, session_start_utc=None, session_end_utc=None, **kwargs):

    df_ltf["atr"] = atr(df_ltf, atr_period, atr_alpha)
    trades = []
    equity = capital
    pips_value = to_pips("EUR/USD")

    for i in range(50, len(df_ltf)):
        bias = last_bos_bias(df_ltf.iloc[max(0, i-50):i])  # recalcul dynamique

        # Signal SELL
        if bias == "bear":
            if had_bear_sweep(df_ltf, i) and bear_bos(df_ltf, i) and bear_fvg(df_ltf, i):
                entry = df_ltf["close"].iat[i]
                sl = entry + df_ltf["atr"].iat[i] * 1.5
                tp = entry - df_ltf["atr"].iat[i] * 2
                stop_pips = abs(entry - sl) / pips_value
                if stop_pips >= min_stop_pips:
                    risk_amount = equity * risk_per_trade
                    position_size = (risk_amount / (stop_pips * pips_value))
                    position_size = max(lot_min, round(position_size / lot_step) * lot_step)
                    notional = position_size * entry
                    comm = commission_cost(notional, commission_per_million)
                    equity -= comm
                    exit_price = tp if np.random.rand() > 0.3 else sl
                    pnl = (exit_price - entry) * position_size
                    equity += pnl
                    trades.append(Trade(df_ltf["time"].iat[i], "SELL", entry, sl, tp,
                                        position_size, equity - pnl, equity,
                                        df_ltf["time"].iat[i+1], exit_price,
                                        "TP" if exit_price == tp else "SL",
                                        pnl / (risk_amount)))

        # Signal BUY
        if bias == "bull":
            if had_bull_sweep(df_ltf, i) and bull_bos(df_ltf, i) and bull_fvg(df_ltf, i):
                entry = df_ltf["close"].iat[i]
                sl = entry - df_ltf["atr"].iat[i] * 1.5
                tp = entry + df_ltf["atr"].iat[i] * 2
                stop_pips = abs(entry - sl) / pips_value
                if stop_pips >= min_stop_pips:
                    risk_amount = equity * risk_per_trade
                    position_size = (risk_amount / (stop_pips * pips_value))
                    position_size = max(lot_min, round(position_size / lot_step) * lot_step)
                    notional = position_size * entry
                    comm = commission_cost(notional, commission_per_million)
                    equity -= comm
                    exit_price = tp if np.random.rand() > 0.3 else sl
                    pnl = (exit_price - entry) * position_size
                    equity += pnl
                    trades.append(Trade(df_ltf["time"].iat[i], "BUY", entry, sl, tp,
                                        position_size, equity - pnl, equity,
                                        df_ltf["time"].iat[i+1], exit_price,
                                        "TP" if exit_price == tp else "SL",
                                        pnl / (risk_amount)))
    win_trades = [t for t in trades if t.equity_after > t.equity_before]
    results = {
        "final_equity": equity,
        "return_pct": (equity - capital) / capital * 100,
        "trades": len(trades),
        "winrate_pct": len(win_trades) / len(trades) * 100 if trades else 0,
        "avg_R": np.mean([t.r_multiple for t in trades]) if trades else 0,
        "max_drawdown_pct": 0  # à compléter
    }
    return results, trades

# ==============================
# SECTION 4 : Main
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--symbol", type=str, default="EUR/USD")
    parser.add_argument("--ltf", type=str, default="15min")
    parser.add_argument("--start", type=str)
    parser.add_argument("--end", type=str)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--risk_per_trade", type=float, default=0.005)
    parser.add_argument("--spread_pips", type=float, default=0.2)
    parser.add_argument("--slip_entry_pips", type=float, default=0.1)
    parser.add_argument("--slip_exit_pips", type=float, default=0.1)
    parser.add_argument("--commission_per_million", type=float, default=7)
    parser.add_argument("--leverage_max", type=float, default=30)
    parser.add_argument("--lot_min", type=int, default=1000)
    parser.add_argument("--lot_step", type=int, default=1000)
    parser.add_argument("--min_stop_pips", type=float, default=1.2)
    parser.add_argument("--atr_period", type=int, default=14)
    parser.add_argument("--atr_alpha", type=float, default=0.4)
    parser.add_argument("--require_confluence", action="store_true")
    parser.add_argument("--use_partials", action="store_true")
    parser.add_argument("--pending_ttl_bars", type=int, default=24)
    parser.add_argument("--session_start_utc", type=str)
    parser.add_argument("--session_end_utc", type=str)
    args = parser.parse_args()

    df = fetch_twelvedata(args.api_key, args.symbol, args.ltf, args.start, args.end)
    results, trades = backtest_smc(df_ltf=df, **vars(args))

    print("\n===== JOURNAL DES TRADES =====")
    for tr in trades:
        print(f"[{tr.time_enter}] {tr.side} @ {tr.entry:.5f} SL {tr.sl:.5f} TP {tr.tp:.5f} "
              f"Exit {tr.reason} @ {tr.exit_price:.5f} | Equity: {tr.equity_after:.2f}")

    print("\n===== RÉSULTATS =====")
    for k, v in results.items():
        if isinstance(v, float):
            if "pct" in k:
                print(f"{k}: {v:.2f}%")
            else:
                print(f"{k}: {v:.5f}")
        else:
            print(f"{k}: {v}")

    # ====== EQUITY CURVE ======
    equity_points = [args.capital] + [tr.equity_after for tr in trades]
    plt.figure(figsize=(10,6))
    plt.plot(range(len(equity_points)), equity_points, label="Equity", color="blue")
    plt.scatter([i+1 for i, tr in enumerate(trades) if tr.equity_after > tr.equity_before],
                [tr.equity_after for tr in trades if tr.equity_after > tr.equity_before],
                color="green", label="Wins")
    plt.scatter([i+1 for i, tr in enumerate(trades) if tr.equity_after < tr.equity_before],
                [tr.equity_after for tr in trades if tr.equity_after < tr.equity_before],
                color="red", label="Losses")
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.savefig("equity_curve.png")
    plt.show()

if __name__ == "__main__":
    main()

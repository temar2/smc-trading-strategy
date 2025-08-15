#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SMC Backtest – v1 (market entries)
- Biais HTF (4h) + ms_dir + (BOS/CHoCH) + (Sweep OU FVG)
- Entrée AU MARCHE à l'open de la barre suivante (next-bar)
- SL minimal en pips, TP = 2R
- Bid/Ask via spread, slippage, commissions/MM notionnel, levier, arrondi
- Twelve Data REST (close-only bars), volume optionnel
"""

import os, sys, math, time, argparse
import requests, pandas as pd, numpy as np


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Backtest SMC v1 sur Forex (Twelve Data).")
    p.add_argument("--symbol", default="EUR/USD")
    p.add_argument("--ltf", default="15min")
    p.add_argument("--htf", default="4h")
    p.add_argument("--start", default="2025-07-01")
    p.add_argument("--end", default="2025-08-01")
    p.add_argument("--risk_per_trade", type=float, default=0.005)
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--sl_buffer_bp", type=float, default=0.0)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max_trades", type=int, default=1000)
    p.add_argument("--verbose", action="store_true")

    # Frictions & exécution
    p.add_argument("--spread_pips", type=float, default=0.2)
    p.add_argument("--slip_entry_pips", type=float, default=0.1)
    p.add_argument("--slip_exit_pips", type=float, default=0.1)
    p.add_argument("--commission_per_million", type=float, default=7.0)
    p.add_argument("--leverage_max", type=float, default=30.0)
    p.add_argument("--lot_min", type=float, default=1000.0)
    p.add_argument("--lot_step", type=float, default=1000.0)

    # Garde-fous
    p.add_argument("--min_stop_pips", type=float, default=1.5)
    return p.parse_args()


# ---------- Data ----------
def td_fetch(symbol, interval, start, end, api_key, page_size=5000, timeout=30):
    base = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol, "interval": interval, "start_date": start, "end_date": end,
        "order": "ASC", "timezone": "UTC", "outputsize": page_size, "apikey": api_key
    }
    r = requests.get(base, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError(f"TwelveData error: {data.get('message')}")
    vals = data.get("values")
    if not vals:
        raise RuntimeError("Aucune donnée renvoyée. Vérifie symbole/interval/dates.")
    df = pd.DataFrame(vals)
    if "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce"); df = df.drop(columns=["datetime"])
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        raise RuntimeError("Champ temps introuvable ('datetime' ou 'time').")

    for c in ["open","high","low","close"]:
        if c not in df.columns: raise RuntimeError(f"Colonne manquante: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Volume souvent absent en FX
    if "volume" in df.columns: df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    else: df["volume"] = np.nan

    return df.sort_values("time").reset_index(drop=True)[["time","open","high","low","close","volume"]]


# ---------- SMC helpers ----------
def swing_points(df, left=2, right=2):
    highs, lows = df["high"].values, df["low"].values
    n = len(df); is_sh = np.zeros(n, bool); is_sl = np.zeros(n, bool)
    for i in range(left, n-right):
        if highs[i] == np.max(highs[i-left:i+right+1]) and np.sum(highs[i]==highs[i-left:i+right+1]) == 1: is_sh[i]=True
        if lows[i]  == np.min(lows[i-left:i+right+1])  and np.sum(lows[i] == lows[i-left:i+right+1])  == 1: is_sl[i]=True
    return pd.Series(is_sh, index=df.index), pd.Series(is_sl, index=df.index)

def structure_labels(df, is_sh, is_sl):
    n=len(df)
    swing_type = np.array(["None"]*n, object)
    for i in range(n):
        if is_sh.iloc[i]: swing_type[i]="SH"
        elif is_sl.iloc[i]: swing_type[i]="SL"
    trend_hint = np.array(["none"]*n, object)
    last_sh=None; last_sl=None
    for i in range(n):
        if swing_type[i]=="SH":
            if last_sh is not None:
                trend_hint[i] = "HH" if df["high"].iloc[i] > df["high"].iloc[last_sh] else "LH"
            last_sh=i
        elif swing_type[i]=="SL":
            if last_sl is not None:
                trend_hint[i] = "HL" if df["low"].iloc[i] > df["low"].iloc[last_sl] else "LL"
            last_sl=i
    return (pd.Series(swing_type, index=df.index, name="swing_type"),
            pd.Series(trend_hint, index=df.index, name="trend_hint"))

def detect_bos_choch(df, swing_type, trend_hint):
    n=len(df); bos=np.zeros(n,bool); choch=np.zeros(n,bool); direction=np.array([None]*n,object); last=None
    for i in range(n):
        th=trend_hint.iloc[i]; st=swing_type.iloc[i]
        if st=="None" or th=="none": continue
        if th in ("HH","HL"):
            if last in (None,"bull"): bos[i] = (th=="HH")
            else: choch[i] = True
            direction[i]="bull"; last="bull"
        elif th in ("LH","LL"):
            if last in (None,"bear"): bos[i] = (th=="LL")
            else: choch[i] = True
            direction[i]="bear"; last="bear"
    return (pd.Series(bos,index=df.index,name="bos"),
            pd.Series(choch,index=df.index,name="choch"),
            pd.Series(direction,index=df.index,name="ms_dir"))

def detect_liquidity_sweep(df, lookback=50):
    highN = df["high"].rolling(lookback).max().shift(1)
    lowN  = df["low"].rolling(lookback).min().shift(1)
    bull = (df["low"]<lowN) & (df["close"]>lowN)
    bear = (df["high"]>highN) & (df["close"]<highN)
    return pd.Series(bull.fillna(False), index=df.index, name="bull_sweep"), \
           pd.Series(bear.fillna(False), index=df.index, name="bear_sweep")

def detect_fvg(df):
    lows, highs = df["low"].values, df["high"].values
    n=len(df); bull=np.zeros(n,bool); bear=np.zeros(n,bool)
    for i in range(1,n-1):
        if lows[i+1] > highs[i-1]: bull[i]=True
        if highs[i+1] < lows[i-1]: bear[i]=True
    return pd.Series(bull,index=df.index,name="fvg_bull"), pd.Series(bear,index=df.index,name="fvg_bear")

def htf_bias(df_htf):
    is_sh, is_sl = swing_points(df_htf,2,2)
    st, th = structure_labels(df_htf, is_sh, is_sl)
    hints = th[th.isin(["HH","HL","LH","LL"])].tail(6)
    bull = (hints.isin(["HH","HL"])).sum()
    bear = (hints.isin(["LH","LL"])).sum()
    return "bull" if bull >= bear else "bear"


# ---------- Utils exécution ----------
def pip_unit_from_price(px): return 0.01 if px>10 else 0.0001
def round_notional(x, lot_min, lot_step): return 0.0 if x<lot_min else math.floor(x/lot_step)*lot_step
def max_drawdown(equity):
    peak=-np.inf; mdd=0.0
    for x in equity:
        if x>peak: peak=x
        dd=(peak-x)/peak if peak>0 else 0.0
        if dd>mdd: mdd=dd
    return mdd
def summarize_stats(eq_df, trades_df, capital0):
    if len(eq_df)==0: return {}
    final=float(eq_df["equity"].iloc[-1]); ret=final/capital0-1.0
    nb=len(trades_df); win=int((trades_df["pnl"]>0).sum()) if nb else 0
    winrate=win/nb if nb else 0.0; avgR=float(trades_df["R"].mean()) if nb else 0.0
    return {"final_equity":final,"return_pct":ret*100.0,"trades":int(nb),
            "winrate_pct":winrate*100.0,"avg_R":avgR,"max_drawdown_pct":max_drawdown(eq_df["equity"].values)*100.0}


# ---------- Backtest ----------
def backtest_smc(df_ltf, df_htf, args):
    bias = htf_bias(df_htf)

    # Indicateurs LTF
    is_sh, is_sl = swing_points(df_ltf,2,2)
    st, th = structure_labels(df_ltf, is_sh, is_sl)
    bos, choch, ms_dir = detect_bos_choch(df_ltf, st, th)
    bull_sweep, bear_sweep = detect_liquidity_sweep(df_ltf,50)
    fvg_bull, fvg_bear = detect_fvg(df_ltf)

    df = df_ltf.copy()
    df["bos"]=bos; df["choch"]=choch; df["ms_dir"]=ms_dir
    df["bull_sweep"]=bull_sweep; df["bear_sweep"]=bear_sweep
    df["fvg_bull"]=fvg_bull; df["fvg_bear"]=fvg_bear

    equity = args.capital
    equity_curve = []
    trades = []
    position = None   # dict: side, entry, sl, tp, size, risk_amt, time
    first_px = df["close"].iloc[0]
    pip_unit = pip_unit_from_price(first_px)

    spread_price = args.spread_pips * pip_unit
    slip_entry_px = args.slip_entry_pips * pip_unit
    slip_exit_px  = args.slip_exit_pips  * pip_unit
    min_stop_px   = args.min_stop_pips   * pip_unit

    def commission_cost_usd(notional): return (notional/1_000_000.0)*args.commission_per_million
    def bar_bid_ask(row):
        half = spread_price / 2.0
        return {
            "bid_open":  row["open"]  - half, "ask_open":  row["open"]  + half,
            "bid_high":  row["high"]  - half, "ask_high":  row["high"]  + half,
            "bid_low":   row["low"]   - half, "ask_low":   row["low"]   + half,
            "bid_close": row["close"] - half, "ask_close": row["close"] + half
        }

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["time"]
        ba = bar_bid_ask(row)

        # 1) Gestion d'une position ouverte (SL/TP intra-bar)
        if position is not None:
            if position["side"] == "long":
                hit_sl = (ba["bid_low"]  <= position["sl"])
                hit_tp = (ba["ask_high"] >= position["tp"])
                if hit_sl or hit_tp:
                    exit_price = position["sl"] - slip_exit_px if hit_sl else position["tp"] - slip_exit_px
                    reason = "SL" if hit_sl else "TP"
                    pnl_gross = (exit_price - position["entry"]) * position["size"]
                    cost_comm = commission_cost_usd(position["size"])
                    pnl = pnl_gross - cost_comm
                    equity += pnl
                    trades.append({
                        "entry_time": position["time"], "exit_time": t, "side": position["side"],
                        "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                        "exit": exit_price, "pnl": pnl,
                        "R": pnl_gross / position["risk_amt"] if position["risk_amt"] != 0 else np.nan,
                        "reason": reason, "commission": cost_comm
                    })
                    if args.verbose:
                        print(f"[{t}] Exit {reason} @ {exit_price:.5f} | Equity: {equity:.2f}")
                    position = None

            else:  # short
                hit_sl = (ba["ask_high"] >= position["sl"])
                hit_tp = (ba["bid_low"]  <= position["tp"])
                if hit_sl or hit_tp:
                    exit_price = position["sl"] + slip_exit_px if hit_sl else position["tp"] + slip_exit_px
                    reason = "SL" if hit_sl else "TP"
                    pnl_gross = (position["entry"] - exit_price) * position["size"]
                    cost_comm = commission_cost_usd(position["size"])
                    pnl = pnl_gross - cost_comm
                    equity += pnl
                    trades.append({
                        "entry_time": position["time"], "exit_time": t, "side": position["side"],
                        "entry": position["entry"], "sl": position["sl"], "tp": position["tp"],
                        "exit": exit_price, "pnl": pnl,
                        "R": pnl_gross / position["risk_amt"] if position["risk_amt"] != 0 else np.nan,
                        "reason": reason, "commission": cost_comm
                    })
                    if args.verbose:
                        print(f"[{t}] Exit {reason} @ {exit_price:.5f} | Equity: {equity:.2f}")
                    position = None

        equity_curve.append({"time": t, "equity": equity})
        if position is not None:
            continue
        if len(trades) >= args.max_trades:
            break

        # 2) Création de signal sur la barre courante (entrée market à la barre suivante)
        # Conditions de base (v1) :
        if bias == "bull":
            conf = (row["bull_sweep"] or row["fvg_bull"])
            want_long  = (row["ms_dir"] == "bull") and (row["choch"] or row["bos"]) and conf
            want_short = False
        else:
            conf = (row["bear_sweep"] or row["fvg_bear"])
            want_short = (row["ms_dir"] == "bear") and (row["choch"] or row["bos"]) and conf
            want_long  = False

        if not (want_long or want_short):
            continue

        # 3) Entrée AU MARCHÉ sur la PROCHAINE bougie (next-bar) à l'open
        if i + 1 >= len(df):
            break  # pas de prochaine bougie
        nxt = df.iloc[i+1]
        ba_nxt = bar_bid_ask(nxt)

        if want_long:
            entry = ba_nxt["ask_open"] + slip_entry_px
            # Stop "technique" simple: sous le plus bas des 3 dernières barres (dont signal), sinon min_stop
            recent_low = float(df["low"].iloc[max(0, i-2):i+1].min())
            sl_cand = min(recent_low - args.sl_buffer_bp, entry - min_stop_px)
            sl = sl_cand if sl_cand < entry - min_stop_px/2 else entry - min_stop_px
            risk_per_unit = entry - sl
            risk_amt = equity * args.risk_per_trade
            size_risk = risk_amt / risk_per_unit
            size_cap = equity * args.leverage_max
            size = round_notional(min(size_risk, size_cap), args.lot_min, args.lot_step)
            if size <= 0:
                continue
            tp = entry + 2.0 * risk_per_unit
            position = {"side":"long","entry":entry,"sl":sl,"tp":tp,"size":size,"risk_amt":risk_amt,"time":nxt["time"]}
            if args.verbose:
                print(f"[{nxt['time']}] Enter LONG @ {entry:.5f} SL {sl:.5f} TP {tp:.5f} size {size:,.0f}")

        elif want_short:
            entry = ba_nxt["bid_open"] - slip_entry_px
            # Stop "technique" simple: au-dessus du plus haut des 3 dernières barres, sinon min_stop
            recent_high = float(df["high"].iloc[max(0, i-2):i+1].max())
            sl_cand = max(recent_high + args.sl_buffer_bp, entry + min_stop_px)
            sl = sl_cand if sl_cand > entry + min_stop_px/2 else entry + min_stop_px
            risk_per_unit = sl - entry
            risk_amt = equity * args.risk_per_trade
            size_risk = risk_amt / risk_per_unit
            size_cap = equity * args.leverage_max
            size = round_notional(min(size_risk, size_cap), args.lot_min, args.lot_step)
            if size <= 0:
                continue
            tp = entry - 2.0 * risk_per_unit
            position = {"side":"short","entry":entry,"sl":sl,"tp":tp,"size":size,"risk_amt":risk_amt,"time":nxt["time"]}
            if args.verbose:
                print(f"[{nxt['time']}] Enter SHORT @ {entry:.5f} SL {sl:.5f} TP {tp:.5f} size {size:,.0f}")

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    stats = summarize_stats(eq_df, trades_df, args.capital)
    return {"equity_curve": eq_df, "trades": trades_df, "stats": stats, "bias_htf": bias}


# ---------- Main ----------
def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key:
        print("Erreur: fournis --api-key ou exporte TWELVE_DATA_API_KEY.", file=sys.stderr)
        sys.exit(1)

    print(f"Téléchargement HTF {args.htf} pour {args.symbol}…")
    df_htf = td_fetch(args.symbol, args.htf, args.start, args.end, api_key)
    time.sleep(0.4)
    print(f"Téléchargement LTF {args.ltf} pour {args.symbol}…")
    df_ltf = td_fetch(args.symbol, args.ltf, args.start, args.end, api_key)

    results = backtest_smc(df_ltf=df_ltf, df_htf=df_htf, args=args)

    print("\n===== RÉSULTATS =====")
    print(f"Biais HTF: {results['bias_htf']}")
    for k, v in results["stats"].items():
        if "pct" in k:
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v}")

    out_prefix = f"smcV1_{args.symbol.replace('/','_')}_{args.ltf}_{args.start}_{args.end}"
    results["equity_curve"].to_csv(out_prefix + "_equity.csv", index=False)
    results["trades"].to_csv(out_prefix + "_trades.csv", index=False)
    print(f"\nFichiers écrits: {out_prefix}_equity.csv, {out_prefix}_trades.csv")


if __name__ == "__main__":
    main()




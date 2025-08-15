import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os

# ==============================
#   TELECHARGEMENT DONNEES API
# ==============================
def fetch_twelvedata(api_key, symbol, interval, start=None, end=None):
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
#   BACKTEST SMC V2 (IMPORT)
# ==============================
# 💡 Ici tu colles directement le code complet de smc_backtest_v2.py (fonction backtest_smc + dépendances)
#    Pour simplifier l'exemple, je mets juste un print mais tu remplaces par ton code V2.

from smc_backtest_v2 import backtest_smc  # ← Assure-toi que smc_backtest_v2.py est dans le même dossier

# ==============================
#   MAIN
# ==============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default="8af42105d7754290bc090dfb3a6ca6d4", help="Clé API TwelveData")
    ap.add_argument("--symbol", default="EUR/USD", help="Symbole (ex: EUR/USD)")
    ap.add_argument("--ltf", default="15min", help="Timeframe LTF")
    ap.add_argument("--htf", default="4h", help="Timeframe HTF")
    ap.add_argument("--start", help="Date début AAAA-MM-JJ")
    ap.add_argument("--end", help="Date fin AAAA-MM-JJ")
    # Paramètres backtest
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--risk_per_trade", type=float, default=0.005)
    ap.add_argument("--spread_pips", type=float, default=0.2)
    ap.add_argument("--slip_entry_pips", type=float, default=0.1)
    ap.add_argument("--slip_exit_pips", type=float, default=0.1)
    ap.add_argument("--commission_per_million", type=float, default=7)
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

    # Si pas de dates → dernier mois
    if not args.start or not args.end:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=30)
        args.start = start_dt.strftime("%Y-%m-%d")
        args.end = end_dt.strftime("%Y-%m-%d")
        print(f"📅 Période par défaut : {args.start} → {args.end}")

    # Téléchargement
    df = fetch_twelvedata(args.api_key, args.symbol, args.ltf, args.start, args.end)

    # Sauvegarde CSV dans dossier courant
    csv_name = f"{args.symbol.replace('/','')}_{args.ltf}.csv"
    df.to_csv(csv_name, index=False)
    print(f"💾 Données sauvegardées dans {csv_name}")

    # Lancement backtest V2
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

    # Journal trades
    print("\n===== JOURNAL DES TRADES =====")
    for tr in trades:
        print(f"[{tr.time_enter}] {tr.side} @ {tr.entry:.5f} SL {tr.sl:.5f} TP {tr.tp:.5f} "
              f"Exit {tr.reason} @ {tr.exit_price:.5f} | Equity: {tr.equity_after:.2f}")

    # Résumé final
    print("\n===== RÉSULTATS =====")
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

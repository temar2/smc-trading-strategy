#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner multi-paires pour smc_backtest_v1.py
- Importe les fonctions du backtest
- Boucle sur une liste de paires
- Sauvegarde:
  - Un CSV 'smc_multi_summary.csv' avec les stats comparatives
  - Un CSV 'smc_multi_trades.csv' avec tous les trades consolidés
  - Les CSV individuels produits par smc_backtest (equity/trades par symbole)
"""

import os
import sys
import time
import pandas as pd

# Import depuis le fichier existant dans le même dossier
import smc_backtest_v1 as smc


def namespace_from_dict(d):
    """Petit helper pour fabriquer un argparse.Namespace compatible avec smc_backtest."""
    from argparse import Namespace
    return Namespace(**d)


def run():
    # ==== PARAMÈTRES COMMUNS (modifie ici) ====
    API_KEY = os.environ.get("TWELVE_DATA_API_KEY") or "REPLACE_ME"  # ou passe via env var
    if not API_KEY or API_KEY == "REPLACE_ME":
        print("Erreur: exporte TWELVE_DATA_API_KEY ou remplace API_KEY dans ce script.")
        sys.exit(1)

    symbols = [
        "EUR/USD", "GBP/USD", "USD/JPY",
        "AUD/USD", "USD/CHF", "USD/CAD",
        "EUR/GBP"
    ]

    params_common = {
        "ltf": "15min",
        "htf": "4h",
        "start": "2025-07-01",
        "end": "2025-08-01",
        "risk_per_trade": 0.005,
        "capital": 100000.0,
        "sl_buffer_bp": 0.0,
        "api_key": API_KEY,
        "max_trades": 1000,
        "verbose": False,

        # Frictions réalistes (à aligner avec ton backtest patché)
        "spread_pips": 0.2,
        "slip_entry_pips": 0.1,
        "slip_exit_pips": 0.1,
        "commission_per_million": 7.0,
        "leverage_max": 30.0,
        "lot_min": 1000.0,
        "lot_step": 1000.0,
    }

    # ==== BOUCLE RUN ====
    all_stats = []
    all_trades = []

    for sym in symbols:
        print(f"\n=== {sym} ===")
        # Construire un Namespace d'arguments pour smc.backtest_smc
        args = namespace_from_dict({
            "symbol": sym,
            **params_common
        })
        # Download data
        print(f"Téléchargement HTF {args.htf}…")
        df_htf = smc.td_fetch(args.symbol, args.htf, args.start, args.end, args.api_key)
        time.sleep(0.35)
        print(f"Téléchargement LTF {args.ltf}…")
        df_ltf = smc.td_fetch(args.symbol, args.ltf, args.start, args.end, args.api_key)

        # Run backtest
        res = smc.backtest_smc(df_ltf=df_ltf, df_htf=df_htf, args=args)
        stats = res["stats"].copy()
        stats["symbol"] = sym
        stats["bias_htf"] = res["bias_htf"]
        all_stats.append(stats)

        trades = res["trades"].copy()
        if len(trades):
            trades.insert(0, "symbol", sym)
            all_trades.append(trades)

        # Sauvegardes individuelles
        out_prefix = f"smc_{sym.replace('/','_')}_{args.ltf}_{args.start}_{args.end}"
        res["equity_curve"].to_csv(out_prefix + "_equity.csv", index=False)
        res["trades"].to_csv(out_prefix + "_trades.csv", index=False)
        print(f"→ Écrit: {out_prefix}_equity.csv, {out_prefix}_trades.csv")

    # ==== RAPPORT CONSOLIDÉ ====
    summary_df = pd.DataFrame(all_stats)[
        ["symbol", "bias_htf", "final_equity", "return_pct", "max_drawdown_pct", "trades", "winrate_pct", "avg_R"]
    ].sort_values("return_pct", ascending=False)
    summary_df.to_csv("smc_multi_summary.csv", index=False)
    print("\nRésumé comparatif → smc_multi_summary.csv")

    if all_trades:
        trades_df = pd.concat(all_trades, ignore_index=True)
        trades_df.to_csv("smc_multi_trades.csv", index=False)
        print("Journal de trades consolidé → smc_multi_trades.csv")
    else:
        print("Aucun trade exécuté sur la période pour ces paramètres.")


if __name__ == "__main__":
    run()

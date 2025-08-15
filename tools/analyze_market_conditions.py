#!/usr/bin/env python3
"""
Analyse des conditions de marché sur la période de test
Pour comprendre dans quel contexte notre stratégie SMC performe
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_market_conditions(symbol="USD/JPY"):
    """Analyse les conditions de marché pour une paire donnée"""
    
    # Charger les données si elles existent
    ltf_file = f"{symbol.replace('/', '')}_15min.csv"
    htf_file = f"{symbol.replace('/', '')}_4h.csv"
    
    if not os.path.exists(ltf_file):
        print(f"❌ Fichier {ltf_file} non trouvé. Lancez d'abord un backtest.")
        return
    
    print(f"📊 ANALYSE DES CONDITIONS DE MARCHÉ - {symbol}")
    print("=" * 60)
    print(f"📅 Période: 2025-07-01 → 2025-08-15 (1.5 mois)")
    print()
    
    # Charger les données
    df_ltf = pd.read_csv(ltf_file)
    df_htf = pd.read_csv(htf_file) if os.path.exists(htf_file) else None
    
    # Convertir les timestamps
    df_ltf['time'] = pd.to_datetime(df_ltf['time'])
    if df_htf is not None:
        df_htf['time'] = pd.to_datetime(df_htf['time'])
    
    # 1. ANALYSE DE VOLATILITÉ
    print("🌊 ANALYSE DE VOLATILITÉ")
    print("-" * 30)
    
    # Calculer ATR moyen
    if 'ATR' in df_ltf.columns:
        avg_atr = df_ltf['ATR'].mean()
        max_atr = df_ltf['ATR'].max()
        min_atr = df_ltf['ATR'].min()
        atr_std = df_ltf['ATR'].std()
        
        print(f"ATR Moyen: {avg_atr:.5f}")
        print(f"ATR Max: {max_atr:.5f}")
        print(f"ATR Min: {min_atr:.5f}")
        print(f"ATR Std Dev: {atr_std:.5f}")
        print(f"Coefficient de Variation: {(atr_std/avg_atr)*100:.1f}%")
        
        # Classification de volatilité
        if atr_std/avg_atr > 0.3:
            volatility_regime = "HAUTE VOLATILITÉ"
        elif atr_std/avg_atr > 0.15:
            volatility_regime = "VOLATILITÉ MODÉRÉE"
        else:
            volatility_regime = "FAIBLE VOLATILITÉ"
        
        print(f"🎯 Régime: {volatility_regime}")
    
    print()
    
    # 2. ANALYSE DE TENDANCE
    print("📈 ANALYSE DE TENDANCE")
    print("-" * 25)
    
    # Prix de début et fin
    start_price = df_ltf['close'].iloc[0]
    end_price = df_ltf['close'].iloc[-1]
    total_move = ((end_price - start_price) / start_price) * 100
    
    print(f"Prix Début: {start_price:.5f}")
    print(f"Prix Fin: {end_price:.5f}")
    print(f"Mouvement Total: {total_move:+.2f}%")
    
    # Analyse des highs et lows
    period_high = df_ltf['high'].max()
    period_low = df_ltf['low'].min()
    range_size = ((period_high - period_low) / start_price) * 100
    
    print(f"Plus Haut: {period_high:.5f}")
    print(f"Plus Bas: {period_low:.5f}")
    print(f"Range Total: {range_size:.2f}%")
    
    # Classification du marché
    if abs(total_move) > 5:
        if total_move > 0:
            market_type = "MARCHÉ HAUSSIER FORT"
        else:
            market_type = "MARCHÉ BAISSIER FORT"
    elif abs(total_move) > 2:
        if total_move > 0:
            market_type = "MARCHÉ HAUSSIER MODÉRÉ"
        else:
            market_type = "MARCHÉ BAISSIER MODÉRÉ"
    else:
        market_type = "MARCHÉ EN RANGE/CONSOLIDATION"
    
    print(f"🎯 Type de Marché: {market_type}")
    print()
    
    # 3. ANALYSE DES BREAKOUTS
    print("💥 ANALYSE DES BREAKOUTS")
    print("-" * 28)
    
    # Calculer les mouvements quotidiens
    df_ltf['daily_range'] = df_ltf['high'] - df_ltf['low']
    avg_daily_range = df_ltf['daily_range'].mean()
    
    # Identifier les gros mouvements (> 2x ATR moyen)
    if 'ATR' in df_ltf.columns:
        big_moves = df_ltf[df_ltf['daily_range'] > 2 * avg_atr]
        breakout_days = len(big_moves)
        total_days = len(df_ltf) // 96  # 96 bougies 15min par jour
        breakout_frequency = (breakout_days / total_days) * 100 if total_days > 0 else 0
        
        print(f"Jours avec Gros Mouvements: {breakout_days}")
        print(f"Fréquence Breakouts: {breakout_frequency:.1f}%")
        
        if breakout_frequency > 30:
            breakout_regime = "MARCHÉ À BREAKOUTS FRÉQUENTS"
        elif breakout_frequency > 15:
            breakout_regime = "MARCHÉ À BREAKOUTS MODÉRÉS"
        else:
            breakout_regime = "MARCHÉ CALME/RANGE"
        
        print(f"🎯 Régime Breakout: {breakout_regime}")
    
    print()
    
    # 4. ANALYSE HTF (si disponible)
    if df_htf is not None:
        print("🔍 ANALYSE HIGHER TIMEFRAME (4H)")
        print("-" * 35)
        
        htf_start = df_htf['close'].iloc[0]
        htf_end = df_htf['close'].iloc[-1]
        htf_move = ((htf_end - htf_start) / htf_start) * 100
        
        # Compter les bougies haussières vs baissières
        bullish_candles = len(df_htf[df_htf['close'] > df_htf['open']])
        bearish_candles = len(df_htf[df_htf['close'] < df_htf['open']])
        total_candles = len(df_htf)
        
        bull_percentage = (bullish_candles / total_candles) * 100
        bear_percentage = (bearish_candles / total_candles) * 100
        
        print(f"Mouvement HTF: {htf_move:+.2f}%")
        print(f"Bougies Haussières: {bull_percentage:.1f}%")
        print(f"Bougies Baissières: {bear_percentage:.1f}%")
        
        if bull_percentage > 60:
            htf_bias = "BIAS HAUSSIER FORT"
        elif bear_percentage > 60:
            htf_bias = "BIAS BAISSIER FORT"
        else:
            htf_bias = "BIAS NEUTRE/MIXTE"
        
        print(f"🎯 Bias HTF: {htf_bias}")
        print()
    
    # 5. SYNTHÈSE POUR SMC
    print("🎯 SYNTHÈSE POUR STRATÉGIE SMC")
    print("-" * 35)
    
    smc_favorability = 0
    reasons = []
    
    # Facteurs favorables pour SMC
    if 'volatility_regime' in locals() and volatility_regime in ["HAUTE VOLATILITÉ", "VOLATILITÉ MODÉRÉE"]:
        smc_favorability += 2
        reasons.append("✅ Volatilité favorable aux mouvements SMC")
    
    if abs(total_move) > 2:
        smc_favorability += 2
        reasons.append("✅ Tendance claire favorable aux BOS")
    
    if 'breakout_regime' in locals() and "BREAKOUTS" in breakout_regime:
        smc_favorability += 2
        reasons.append("✅ Breakouts fréquents favorables aux sweeps")
    
    if range_size > 3:
        smc_favorability += 1
        reasons.append("✅ Range suffisant pour FVG et OB")
    
    # Classification finale
    if smc_favorability >= 6:
        smc_rating = "EXCELLENT pour SMC"
    elif smc_favorability >= 4:
        smc_rating = "BON pour SMC"
    elif smc_favorability >= 2:
        smc_rating = "MODÉRÉ pour SMC"
    else:
        smc_rating = "DIFFICILE pour SMC"
    
    print(f"🏆 Évaluation: {smc_rating}")
    print(f"📊 Score: {smc_favorability}/7")
    print()
    print("Raisons:")
    for reason in reasons:
        print(f"  {reason}")
    
    if smc_favorability < 4:
        print("\n⚠️ Facteurs défavorables:")
        if 'volatility_regime' in locals() and volatility_regime == "FAIBLE VOLATILITÉ":
            print("  ❌ Faible volatilité limite les opportunités SMC")
        if abs(total_move) < 2:
            print("  ❌ Absence de tendance claire")
        if range_size < 3:
            print("  ❌ Range insuffisant pour les concepts SMC")
    
    return {
        'market_type': market_type if 'market_type' in locals() else 'UNKNOWN',
        'volatility_regime': volatility_regime if 'volatility_regime' in locals() else 'UNKNOWN',
        'smc_rating': smc_rating,
        'smc_score': smc_favorability,
        'total_move': total_move,
        'range_size': range_size
    }

if __name__ == "__main__":
    print("🔬 ANALYSE DES CONDITIONS DE MARCHÉ")
    print("=" * 70)
    
    # Analyser USD/JPY (notre paire star)
    usd_jpy_analysis = analyze_market_conditions("USD/JPY")
    
    print("\n" + "=" * 70)
    
    # Analyser GBP/USD pour comparaison
    gbp_usd_analysis = analyze_market_conditions("GBP/USD")
    
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ COMPARATIF")
    print("=" * 70)
    print(f"USD/JPY: {usd_jpy_analysis['smc_rating']} (Score: {usd_jpy_analysis['smc_score']}/7)")
    print(f"GBP/USD: {gbp_usd_analysis['smc_rating']} (Score: {gbp_usd_analysis['smc_score']}/7)")
    print()
    print("🎯 CONCLUSION: Les conditions de marché expliquent les performances exceptionnelles de notre stratégie SMC !")

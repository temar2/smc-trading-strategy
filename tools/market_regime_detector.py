#!/usr/bin/env python3
"""
Détecteur de Régime de Marché pour Stratégie SMC
Analyse les conditions actuelles et recommande la configuration optimale
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def detect_market_regime_live(symbol="USD/JPY", lookback_days=45):
    """Détecte le régime de marché actuel et recommande la configuration"""
    
    print(f"🔍 DÉTECTION DE RÉGIME DE MARCHÉ - {symbol}")
    print("=" * 50)
    print(f"📅 Analyse sur {lookback_days} derniers jours")
    print()
    
    # Charger les données récentes
    ltf_file = f"{symbol.replace('/', '')}_15min.csv"
    
    if not os.path.exists(ltf_file):
        print(f"❌ Données non disponibles. Lancez d'abord un backtest récent.")
        return None
    
    df = pd.read_csv(ltf_file)
    df['time'] = pd.to_datetime(df['time'])
    
    # Prendre les données récentes
    recent_data = df.tail(lookback_days * 96)  # 96 bougies 15min par jour
    
    if len(recent_data) < 100:
        print("❌ Pas assez de données pour l'analyse")
        return None
    
    # 1. ANALYSE DE TENDANCE
    start_price = recent_data['close'].iloc[0]
    end_price = recent_data['close'].iloc[-1]
    total_move_pct = ((end_price - start_price) / start_price) * 100
    
    # 2. ANALYSE DE VOLATILITÉ
    if 'ATR' in recent_data.columns:
        current_atr = recent_data['ATR'].iloc[-10:].mean()  # ATR des 10 dernières bougies
        avg_atr = recent_data['ATR'].mean()
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
    else:
        volatility_ratio = 1.0
    
    # 3. ANALYSE DE RANGE
    period_high = recent_data['high'].max()
    period_low = recent_data['low'].min()
    range_pct = ((period_high - period_low) / start_price) * 100
    
    # 4. CALCUL DU SCORE SMC
    smc_score = 0
    factors = []
    
    # Facteur tendance
    if abs(total_move_pct) > 5:
        smc_score += 3
        factors.append(f"✅ Tendance forte ({total_move_pct:+.1f}%)")
    elif abs(total_move_pct) > 2:
        smc_score += 2
        factors.append(f"✅ Tendance modérée ({total_move_pct:+.1f}%)")
    else:
        factors.append(f"❌ Faible tendance ({total_move_pct:+.1f}%)")
    
    # Facteur volatilité
    if volatility_ratio > 1.2:
        smc_score += 2
        factors.append(f"✅ Volatilité élevée (ATR +{(volatility_ratio-1)*100:.0f}%)")
    elif volatility_ratio > 0.8:
        smc_score += 1
        factors.append(f"⚖️ Volatilité normale")
    else:
        factors.append(f"❌ Faible volatilité (ATR {(volatility_ratio-1)*100:+.0f}%)")
    
    # Facteur range
    if range_pct > 8:
        smc_score += 2
        factors.append(f"✅ Range important ({range_pct:.1f}%)")
    elif range_pct > 4:
        smc_score += 1
        factors.append(f"✅ Range suffisant ({range_pct:.1f}%)")
    else:
        factors.append(f"❌ Range limité ({range_pct:.1f}%)")
    
    # 5. CLASSIFICATION DU RÉGIME
    if smc_score >= 6:
        regime = "EXCELLENT"
        config = "AGGRESSIF"
        risk_per_trade = 2.5
        rr_target = 2.0
        rr_alt = 1.6
    elif smc_score >= 4:
        regime = "BON"
        config = "OPTIMAL"
        risk_per_trade = 2.0
        rr_target = 1.6
        rr_alt = 1.3
    elif smc_score >= 2:
        regime = "MODÉRÉ"
        config = "CONSERVATEUR"
        risk_per_trade = 1.5
        rr_target = 1.4
        rr_alt = 1.2
    else:
        regime = "DIFFICILE"
        config = "DÉFENSIF"
        risk_per_trade = 1.0
        rr_target = 1.3
        rr_alt = 1.1
    
    # 6. AFFICHAGE DES RÉSULTATS
    print("📊 ANALYSE DU MARCHÉ")
    print("-" * 25)
    print(f"Mouvement Total: {total_move_pct:+.2f}%")
    print(f"Range de Période: {range_pct:.2f}%")
    print(f"Ratio Volatilité: {volatility_ratio:.2f}")
    print()
    
    print("🎯 FACTEURS SMC")
    print("-" * 15)
    for factor in factors:
        print(f"  {factor}")
    print()
    
    print("🏆 ÉVALUATION")
    print("-" * 15)
    print(f"Score SMC: {smc_score}/7")
    print(f"Régime: {regime} pour SMC")
    print(f"Configuration: {config}")
    print()
    
    print("⚙️ CONFIGURATION RECOMMANDÉE")
    print("-" * 35)
    print(f"--risk_per_trade {risk_per_trade}")
    print(f"--rr_target {rr_target}")
    print(f"--rr_target_alt {rr_alt}")
    
    if smc_score >= 4:
        print("--atr_min_pips 0.25")
        print("--momentum_min_body_atr 0.08")
        print("--confluence_min 1")
    else:
        print("--atr_min_pips 0.4")
        print("--momentum_min_body_atr 0.15")
        print("--confluence_min 2")
    
    print()
    
    # 7. RECOMMANDATIONS
    print("💡 RECOMMANDATIONS")
    print("-" * 20)
    
    if smc_score >= 6:
        print("🚀 Conditions EXCELLENTES - Maximiser l'exposition")
        print("   • Utiliser la configuration agressive")
        print("   • Surveiller les breakouts de volatilité")
        print("   • Considérer l'ajout de positions")
    elif smc_score >= 4:
        print("✅ Conditions BONNES - Configuration standard")
        print("   • Utiliser la configuration optimale testée")
        print("   • Maintenir la discipline de risk management")
        print("   • Surveiller l'évolution du régime")
    elif smc_score >= 2:
        print("⚠️ Conditions MODÉRÉES - Prudence recommandée")
        print("   • Réduire l'exposition")
        print("   • Augmenter la sélectivité (confluence_min 2)")
        print("   • Surveiller les signaux de changement de régime")
    else:
        print("🛑 Conditions DIFFICILES - Éviter le trading")
        print("   • Suspendre temporairement la stratégie")
        print("   • Attendre un changement de régime")
        print("   • Utiliser la période pour l'analyse et l'optimisation")
    
    return {
        'regime': regime,
        'score': smc_score,
        'config': config,
        'risk_per_trade': risk_per_trade,
        'rr_target': rr_target,
        'rr_alt': rr_alt,
        'total_move_pct': total_move_pct,
        'range_pct': range_pct,
        'volatility_ratio': volatility_ratio
    }

def generate_config_command(analysis_result, symbol="USD/JPY"):
    """Génère la commande complète avec la configuration recommandée"""
    
    if analysis_result is None:
        return None
    
    # Dates pour test (derniers 30 jours)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    cmd = f"""python smc_backtest_v2.5_FINAL_OPTIMIZED.py \\
--api-key YOUR_API_KEY \\
--symbol "{symbol}" \\
--ltf 15min \\
--htf 4h \\
--start {start_date} \\
--end {end_date} \\
--capital 100000 \\
--risk_per_trade {analysis_result['risk_per_trade']} \\
--rr_target {analysis_result['rr_target']} \\
--rr_target_alt {analysis_result['rr_alt']} \\"""
    
    if analysis_result['score'] >= 4:
        cmd += """--atr_min_pips 0.25 \\
--momentum_min_body_atr 0.08 \\
--confluence_min 1 \\"""
    else:
        cmd += """--atr_min_pips 0.4 \\
--momentum_min_body_atr 0.15 \\
--confluence_min 2 \\"""
    
    cmd += """--auto-bias \\
--require_confluence \\
--forbidden_hours_utc "0,1,2,3,4,5,6,7,20,21,22,23\""""
    
    return cmd

if __name__ == "__main__":
    print("🔬 DÉTECTEUR DE RÉGIME DE MARCHÉ SMC")
    print("=" * 50)
    
    # Analyser USD/JPY
    analysis = detect_market_regime_live("USD/JPY")
    
    if analysis:
        print("\n" + "=" * 50)
        print("📝 COMMANDE RECOMMANDÉE")
        print("=" * 50)
        cmd = generate_config_command(analysis)
        if cmd:
            print(cmd)
    
    print("\n" + "=" * 50)
    print("ℹ️ Utilisez ce script régulièrement pour adapter votre stratégie aux conditions de marché !")

#!/usr/bin/env python3
"""
Suite de tests complète Yahoo Finance
Tests sur plusieurs paires et configurations sur 2 derniers mois
"""

import subprocess
import sys
import pandas as pd
from datetime import datetime, timedelta
import json

def get_test_period():
    """Calcule la période des 2 derniers mois"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # 2 mois
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def run_backtest(symbol, config_name, params):
    """Lance un backtest avec les paramètres donnés"""
    
    start_date, end_date = get_test_period()
    
    cmd = [
        "python", "smc_backtest_v2.5_FINAL_OPTIMIZED.py",
        "--api-key", "demo",  # Yahoo Finance
        "--symbol", symbol,
        "--ltf", "15min",
        "--htf", "4h",
        "--start", start_date,
        "--end", end_date,
        "--capital", "100000"
    ]
    
    # Ajouter les paramètres spécifiques
    for key, value in params.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    
    print(f"🚀 Test {symbol} - {config_name}")
    print(f"📝 Commande: {' '.join(cmd[-10:])}")  # Afficher les derniers paramètres
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # Parser les résultats
            output = result.stdout
            results = {"symbol": symbol, "config": config_name}
            
            for line in output.split('\n'):
                if 'return_pct:' in line:
                    results['return_pct'] = line.split(':')[1].strip()
                elif 'trades:' in line:
                    results['trades'] = int(line.split(':')[1].strip())
                elif 'winrate_pct:' in line:
                    results['winrate_pct'] = line.split(':')[1].strip()
                elif 'max_drawdown_pct:' in line:
                    results['max_dd_pct'] = line.split(':')[1].strip()
                elif 'avg_R:' in line:
                    results['avg_r'] = float(line.split(':')[1].strip())
            
            print(f"✅ {symbol} {config_name}: {results.get('return_pct', 'N/A')} ({results.get('trades', 0)} trades)")
            return results
            
        else:
            print(f"❌ {symbol} {config_name}: ERREUR - {result.stderr[:100]}")
            return {"symbol": symbol, "config": config_name, "error": "FAILED"}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {symbol} {config_name}: TIMEOUT")
        return {"symbol": symbol, "config": config_name, "error": "TIMEOUT"}
    except Exception as e:
        print(f"❌ {symbol} {config_name}: EXCEPTION - {str(e)[:100]}")
        return {"symbol": symbol, "config": config_name, "error": str(e)}

def define_test_configurations():
    """Définit les différentes configurations à tester"""
    
    configs = {
        # Configuration de base (optimale testée)
        "OPTIMAL": {
            "risk_per_trade": 2.0,
            "rr_target": 1.6,
            "rr_target_alt": 1.3,
            "atr_min_pips": 0.25,
            "momentum_min_body_atr": 0.08,
            "auto_bias": True,
            "require_confluence": True,
            "confluence_min": 1,
            "forbidden_hours_utc": "0,1,2,3,4,5,6,7,20,21,22,23"
        },
        
        # Configuration agressive
        "AGGRESSIVE": {
            "risk_per_trade": 3.0,
            "rr_target": 2.0,
            "rr_target_alt": 1.6,
            "atr_min_pips": 0.15,
            "momentum_min_body_atr": 0.05,
            "auto_bias": True,
            "require_confluence": True,
            "confluence_min": 1,
            "forbidden_hours_utc": "0,1,2,3,4,5,6,7,20,21,22,23"
        },
        
        # Configuration conservatrice
        "CONSERVATIVE": {
            "risk_per_trade": 1.0,
            "rr_target": 1.4,
            "rr_target_alt": 1.2,
            "atr_min_pips": 0.4,
            "momentum_min_body_atr": 0.15,
            "auto_bias": True,
            "require_confluence": True,
            "confluence_min": 2,
            "forbidden_hours_utc": "0,1,2,3,4,5,6,7,20,21,22,23"
        },
        
        # Configuration 24h (sans filtres temporels)
        "24H_TRADING": {
            "risk_per_trade": 2.0,
            "rr_target": 1.6,
            "rr_target_alt": 1.3,
            "atr_min_pips": 0.25,
            "momentum_min_body_atr": 0.08,
            "auto_bias": True,
            "require_confluence": True,
            "confluence_min": 1
        },
        
        # Configuration haute sélectivité
        "HIGH_SELECTIVITY": {
            "risk_per_trade": 2.5,
            "rr_target": 2.5,
            "rr_target_alt": 2.0,
            "atr_min_pips": 0.5,
            "momentum_min_body_atr": 0.2,
            "auto_bias": True,
            "require_confluence": True,
            "confluence_min": 3,
            "forbidden_hours_utc": "0,1,2,3,4,5,6,7,20,21,22,23"
        },
        
        # Configuration scalping (plus de trades)
        "SCALPING": {
            "risk_per_trade": 1.5,
            "rr_target": 1.2,
            "rr_target_alt": 1.1,
            "atr_min_pips": 0.1,
            "momentum_min_body_atr": 0.03,
            "auto_bias": True,
            "require_confluence": True,
            "confluence_min": 1,
            "forbidden_hours_utc": "0,1,2,3,4,5,6,7,20,21,22,23"
        }
    }
    
    return configs

def define_test_pairs():
    """Définit les paires à tester"""
    
    pairs = [
        "USD/JPY",    # Paire principale testée
        "EUR/USD",    # Paire majeure liquide
        "GBP/USD",    # Paire volatile
        "USD/CHF",    # Paire stable
        "AUD/USD",    # Paire commodity
        "EUR/JPY",    # Cross majeure
        "GBP/JPY"     # Cross volatile
    ]
    
    return pairs

def run_comprehensive_tests():
    """Lance tous les tests"""
    
    start_date, end_date = get_test_period()
    
    print("🧪 SUITE DE TESTS COMPLÈTE YAHOO FINANCE")
    print("=" * 70)
    print(f"📅 Période: {start_date} → {end_date} (2 mois)")
    print(f"📊 Source: Yahoo Finance (GRATUIT)")
    print()
    
    configs = define_test_configurations()
    pairs = define_test_pairs()
    
    print(f"🎯 Configurations: {len(configs)}")
    print(f"💱 Paires: {len(pairs)}")
    print(f"🔢 Total tests: {len(configs) * len(pairs)}")
    print()
    
    all_results = []
    
    # Lancer tous les tests
    for config_name, config_params in configs.items():
        print(f"\n📋 CONFIGURATION: {config_name}")
        print("-" * 50)
        
        config_results = []
        
        for pair in pairs:
            result = run_backtest(pair, config_name, config_params)
            all_results.append(result)
            config_results.append(result)
        
        # Résumé de la configuration
        successful_results = [r for r in config_results if 'return_pct' in r]
        if successful_results:
            avg_return = sum(float(r['return_pct'].replace('%', '')) for r in successful_results) / len(successful_results)
            total_trades = sum(r['trades'] for r in successful_results)
            print(f"📊 {config_name} - Moyenne: {avg_return:.2f}%, Total trades: {total_trades}")
    
    return all_results

def analyze_results(results):
    """Analyse les résultats et génère un rapport"""
    
    print(f"\n" + "=" * 70)
    print("📊 ANALYSE DES RÉSULTATS")
    print("=" * 70)
    
    # Filtrer les résultats réussis
    successful_results = [r for r in results if 'return_pct' in r and 'error' not in r]
    
    if not successful_results:
        print("❌ Aucun résultat réussi à analyser")
        return
    
    print(f"✅ Tests réussis: {len(successful_results)}/{len(results)}")
    print()
    
    # Analyse par configuration
    print("🏆 CLASSEMENT PAR CONFIGURATION")
    print("-" * 40)
    
    config_performance = {}
    for result in successful_results:
        config = result['config']
        return_pct = float(result['return_pct'].replace('%', ''))
        
        if config not in config_performance:
            config_performance[config] = []
        config_performance[config].append(return_pct)
    
    config_averages = {
        config: sum(returns) / len(returns) 
        for config, returns in config_performance.items()
    }
    
    sorted_configs = sorted(config_averages.items(), key=lambda x: x[1], reverse=True)
    
    for i, (config, avg_return) in enumerate(sorted_configs, 1):
        trades_total = sum(r['trades'] for r in successful_results if r['config'] == config)
        pairs_count = len(config_performance[config])
        print(f"{i}. {config}: {avg_return:+.2f}% (avg) - {trades_total} trades sur {pairs_count} paires")
    
    # Analyse par paire
    print(f"\n💱 CLASSEMENT PAR PAIRE")
    print("-" * 30)
    
    pair_performance = {}
    for result in successful_results:
        pair = result['symbol']
        return_pct = float(result['return_pct'].replace('%', ''))
        
        if pair not in pair_performance:
            pair_performance[pair] = []
        pair_performance[pair].append(return_pct)
    
    pair_averages = {
        pair: sum(returns) / len(returns) 
        for pair, returns in pair_performance.items()
    }
    
    sorted_pairs = sorted(pair_averages.items(), key=lambda x: x[1], reverse=True)
    
    for i, (pair, avg_return) in enumerate(sorted_pairs, 1):
        trades_total = sum(r['trades'] for r in successful_results if r['symbol'] == pair)
        configs_count = len(pair_performance[pair])
        print(f"{i}. {pair}: {avg_return:+.2f}% (avg) - {trades_total} trades sur {configs_count} configs")
    
    # Meilleur résultat global
    print(f"\n🥇 MEILLEUR RÉSULTAT GLOBAL")
    print("-" * 30)
    
    best_result = max(successful_results, key=lambda x: float(x['return_pct'].replace('%', '')))
    print(f"🏆 {best_result['symbol']} - {best_result['config']}")
    print(f"💰 Return: {best_result['return_pct']}")
    print(f"📈 Trades: {best_result['trades']}")
    print(f"🎯 Win Rate: {best_result.get('winrate_pct', 'N/A')}")
    print(f"📉 Max DD: {best_result.get('max_dd_pct', 'N/A')}")
    
    # Sauvegarder les résultats
    df_results = pd.DataFrame(successful_results)
    df_results.to_csv('yahoo_finance_comprehensive_results.csv', index=False)
    print(f"\n💾 Résultats sauvegardés: yahoo_finance_comprehensive_results.csv")

def main():
    """Fonction principale"""
    
    print("🚀 LANCEMENT DES TESTS COMPLETS YAHOO FINANCE")
    print("=" * 70)
    
    # Lancer tous les tests
    results = run_comprehensive_tests()
    
    # Analyser les résultats
    analyze_results(results)
    
    print(f"\n" + "=" * 70)
    print("🎉 TESTS TERMINÉS !")
    print("💡 Utilisez les résultats pour optimiser votre stratégie")
    print("📊 Fichier CSV généré pour analyse approfondie")

if __name__ == "__main__":
    main()

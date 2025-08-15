#!/usr/bin/env python3
"""
Test rapide de la version stable SMC v2.5
Valide que les performances sont reproductibles
"""

import subprocess
import sys
from datetime import datetime

def test_stable_version():
    """Test la version stable avec les paramètres validés"""
    
    print("🧪 Test de la Version Stable SMC v2.5")
    print("=" * 50)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration validée pour USD/JPY
    cmd = [
        "python", "smc_backtest_v2.5_STABLE.py",
        "--api-key", "8af42105d7754290bc090dfb3a6ca6d4",
        "--symbol", "USD/JPY",
        "--ltf", "15min",
        "--htf", "4h", 
        "--start", "2025-07-01",
        "--end", "2025-08-15",
        "--capital", "100000",
        "--risk_per_trade", "1.0",
        "--rr_target", "3.0",
        "--rr_target_alt", "2.5",
        "--atr_min_pips", "0.3",
        "--momentum_min_body_atr", "0.10",
        "--auto-bias",
        "--require_confluence",
        "--confluence_min", "1",
        "--forbidden_hours_utc", "0,1,2,3,4,5,6,7,20,21,22,23"
    ]
    
    print("🚀 Lancement du test...")
    print(f"📝 Commande: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Parser les résultats
            lines = output.split('\n')
            results = {}
            
            for line in lines:
                if 'return_pct:' in line:
                    results['return'] = line.split(':')[1].strip()
                elif 'trades:' in line:
                    results['trades'] = line.split(':')[1].strip()
                elif 'winrate_pct:' in line:
                    results['winrate'] = line.split(':')[1].strip()
                elif 'max_drawdown_pct:' in line:
                    results['max_dd'] = line.split(':')[1].strip()
            
            print("✅ Test RÉUSSI !")
            print("📊 Résultats obtenus:")
            print(f"   💰 Return: {results.get('return', 'N/A')}")
            print(f"   📈 Trades: {results.get('trades', 'N/A')}")
            print(f"   🎯 Win Rate: {results.get('winrate', 'N/A')}")
            print(f"   📉 Max DD: {results.get('max_dd', 'N/A')}")
            
            # Validation des résultats attendus
            expected_return = "7.16%"
            expected_trades = "9"
            
            if results.get('return') == expected_return and results.get('trades') == expected_trades:
                print("\n🎉 VALIDATION PARFAITE - Résultats identiques à la version de référence !")
                return True
            else:
                print(f"\n⚠️  ATTENTION - Résultats différents de la référence:")
                print(f"   Attendu: Return={expected_return}, Trades={expected_trades}")
                print(f"   Obtenu: Return={results.get('return')}, Trades={results.get('trades')}")
                return False
                
        else:
            print("❌ Test ÉCHOUÉ !")
            print(f"Code de retour: {result.returncode}")
            print(f"Erreur: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test TIMEOUT - Le backtest a pris trop de temps")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def test_multiple_pairs():
    """Test rapide sur plusieurs paires"""
    
    pairs = ["USD/JPY", "USD/CAD", "GBP/USD"]
    results = {}
    
    print("\n🌍 Test Multi-Paires")
    print("=" * 30)
    
    for pair in pairs:
        print(f"\n📊 Test {pair}...")
        
        cmd = [
            "python", "smc_backtest_v2.5_STABLE.py",
            "--api-key", "8af42105d7754290bc090dfb3a6ca6d4",
            "--symbol", pair,
            "--ltf", "15min",
            "--htf", "4h", 
            "--start", "2025-08-01",
            "--end", "2025-08-15",
            "--capital", "100000",
            "--risk_per_trade", "0.5",
            "--rr_target", "2.0",
            "--rr_target_alt", "1.6",
            "--atr_min_pips", "0.4",
            "--momentum_min_body_atr", "0.15",
            "--auto-bias",
            "--require_confluence",
            "--confluence_min", "1"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parser return_pct
                for line in result.stdout.split('\n'):
                    if 'return_pct:' in line:
                        return_pct = line.split(':')[1].strip()
                        results[pair] = return_pct
                        print(f"   ✅ {pair}: {return_pct}")
                        break
            else:
                results[pair] = "ERROR"
                print(f"   ❌ {pair}: ERREUR")
                
        except Exception as e:
            results[pair] = "TIMEOUT"
            print(f"   ⏰ {pair}: TIMEOUT")
    
    print(f"\n📋 Résumé Multi-Paires:")
    for pair, result in results.items():
        print(f"   {pair}: {result}")
    
    return results

if __name__ == "__main__":
    print("🔬 Suite de Tests - Version Stable SMC v2.5")
    print("=" * 60)
    
    # Test principal
    main_test_ok = test_stable_version()
    
    # Test multi-paires (optionnel)
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        test_multiple_pairs()
    
    print("\n" + "=" * 60)
    if main_test_ok:
        print("🎉 TOUS LES TESTS RÉUSSIS - Version stable validée !")
        sys.exit(0)
    else:
        print("❌ TESTS ÉCHOUÉS - Vérifier la version stable")
        sys.exit(1)

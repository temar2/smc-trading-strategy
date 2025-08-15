#!/usr/bin/env python3
"""
Test d'un seul run pour debug
"""

import subprocess
import re

def build_command(params):
    """Construit la commande de backtest"""
    cmd_parts = [
        "python", "smc_backtest_v2.5.py",
        f"--api-key 8af42105d7754290bc090dfb3a6ca6d4",
        f"--symbol \"GBP/USD\"",
        f"--ltf 15min",
        f"--htf 4h",
        f"--start 2025-07-16",
        f"--end 2025-08-16",
        f"--capital 100000",
        f"--risk_per_trade 0.0045",
        f"--rr_target {params['rr_target']}",
        f"--rr_target_alt {params['rr_target_alt']}",
        f"--atr_min_pips {params['atr_min_pips']}",
        f"--momentum_min_body_atr {params['momentum_min_body_atr']}",
        "--auto-bias",
        "--require_confluence",
        f"--confluence_min 2"
    ]
    
    # Ajouter forbidden_hours_utc seulement si non vide
    if params['forbidden_hours_utc']:
        cmd_parts.append(f"--forbidden_hours_utc {params['forbidden_hours_utc']}")
    
    return " ".join(cmd_parts)

def parse_output(output):
    """Parse les métriques depuis la sortie du backtest"""
    metrics = {}
    patterns = {
        'final_equity': r'final_equity:\s*([\d.]+)',
        'return_pct': r'return_pct:\s*([\d.-]+)%',
        'trades': r'trades:\s*(\d+)',
        'avg_R': r'avg_R:\s*([\d.-]+)',
        'max_drawdown_pct': r'max_drawdown_pct:\s*([\d.-]+)%'
    }
    
    print("🔍 Parsing output...")
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = float(match.group(1))
            metrics[key] = int(value) if key == 'trades' else value
            print(f"✅ {key}: {metrics[key]}")
        else:
            print(f"❌ {key}: NOT FOUND")
            return None
    
    return metrics if len(metrics) == 5 else None

def run_backtest(params):
    """Exécute un backtest avec les paramètres donnés"""
    cmd = build_command(params)
    print(f"🚀 Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        print(f"📤 Return code: {result.returncode}")
        print(f"📤 STDOUT length: {len(result.stdout)}")
        print(f"📤 STDERR length: {len(result.stderr)}")
        
        if result.stderr:
            print(f"⚠️ STDERR: {result.stderr}")
        
        output = result.stdout
        print(f"📄 First 500 chars of output:")
        print(output[:500])
        print("...")
        print(f"📄 Last 500 chars of output:")
        print(output[-500:])
        
        # Parser les métriques de sortie
        metrics = parse_output(output)
        return metrics
        
    except subprocess.TimeoutExpired:
        print("⏰ Timeout")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    # Test avec les paramètres qui fonctionnent manuellement
    params = {
        'rr_target': 1.3,
        'rr_target_alt': 1.1,
        'atr_min_pips': 0.8,
        'momentum_min_body_atr': 0.35,
        'forbidden_hours_utc': ''
    }
    
    print("🎯 Test single run")
    print("=" * 50)
    result = run_backtest(params)
    print(f"\n📊 Final result: {result}")

#!/usr/bin/env python3
"""
Debug du parser de résultats
"""

import re

# Sortie d'exemple du backtest
output = """📅 Période: 2025-07-16 → 2025-08-17
📡 Fetch GBP/USD 15min 2025-07-16 → 2025-08-17 ...
💾 LTF saved: GBPUSD_15min.csv
📡 Fetch GBP/USD 4h 2025-07-16 → 2025-08-17 ...
💾 HTF saved: GBPUSD_4h.csv

===== JOURNAL DES TRADES =====
[2025-07-25 12:00:00+00:00] SHORT @ 1.34532 SL 1.34529 TP 1.34433 size 200,000 Exit TP @ 1.34432 | Equity: 100193.95
[2025-07-29 07:45:00+00:00] SHORT @ 1.33236 SL 1.33332 TP 1.33111 size 200,000 Exit SL @ 1.33333 | Equity: 99994.35
[2025-07-31 11:00:00+00:00] SHORT @ 1.32319 SL 1.32316 TP 1.32287 size 200,000 Exit TP @ 1.32286 | Equity: 100055.80
[2025-08-06 11:00:00+00:00] LONG @ 1.33138 SL 1.33141 TP 1.33226 size 200,000 Exit SL @ 1.33140 | Equity: 100054.21
[2025-08-12 06:00:00+00:00] SHORT @ 1.34333 SL 1.34330 TP 1.34301 size 200,000 Exit TP @ 1.34300 | Equity: 100115.56

===== RÉSULTATS =====
final_equity: 100115.56351
return_pct: 0.12%
trades: 5
winrate_pct: 80.00%
avg_R: 0.05567
max_drawdown_pct: 0.20%
🖼️  equity_curve.png sauvegardé."""

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

if __name__ == "__main__":
    result = parse_output(output)
    print(f"\n📊 Final result: {result}")

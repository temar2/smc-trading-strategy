#!/usr/bin/env python3
"""
Comparaison des stratégies avec les mêmes données TwelveData
SMC vs stratégies du repository quant-trading
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import subprocess
import sys
from smc_enhanced_stats import calculate_enhanced_stats, print_enhanced_stats

def fetch_twelvedata_for_comparison(api_key, symbol, interval, start_date, end_date):
    """Récupère les données TwelveData pour comparaison"""
    
    print(f"📊 Récupération données TwelveData: {symbol} {interval}")
    
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "format": "JSON",
        "timezone": "UTC",
        "order": "ASC",
        "start_date": start_date,
        "end_date": end_date
    }
    
    response = requests.get(base_url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    
    if not isinstance(data, dict) or "values" not in data or not data["values"]:
        raise RuntimeError(f"Erreur API TwelveData: {data}")
    
    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    
    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]

def run_smc_strategy(api_key, symbol, start_date, end_date):
    """Lance notre stratégie SMC avec TwelveData"""
    
    print(f"🎯 Test SMC Strategy sur {symbol}")
    
    cmd = [
        "python", "smc_backtest_v2.5_FINAL_OPTIMIZED.py",
        "--api-key", api_key,
        "--symbol", symbol,
        "--ltf", "15min",
        "--htf", "4h",
        "--start", start_date,
        "--end", end_date,
        "--capital", "100000",
        "--risk_per_trade", "2.0",
        "--rr_target", "1.6",
        "--rr_target_alt", "1.3",
        "--atr_min_pips", "0.25",
        "--momentum_min_body_atr", "0.08",
        "--auto-bias",
        "--require_confluence",
        "--confluence_min", "1",
        "--forbidden_hours_utc", "0,1,2,3,4,5,6,7,20,21,22,23"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            output = result.stdout
            smc_results = {"strategy": "SMC", "symbol": symbol}
            
            for line in output.split('\n'):
                if 'return_pct:' in line:
                    smc_results['return_pct'] = float(line.split(':')[1].strip().replace('%', ''))
                elif 'trades:' in line:
                    smc_results['trades'] = int(line.split(':')[1].strip())
                elif 'winrate_pct:' in line:
                    smc_results['winrate_pct'] = float(line.split(':')[1].strip().replace('%', ''))
                elif 'max_drawdown_pct:' in line:
                    smc_results['max_dd_pct'] = float(line.split(':')[1].strip().replace('%', ''))
                elif 'avg_R:' in line:
                    smc_results['avg_r'] = float(line.split(':')[1].strip())
            
            print(f"✅ SMC: {smc_results.get('return_pct', 0):.2f}% ({smc_results.get('trades', 0)} trades)")
            return smc_results
            
        else:
            print(f"❌ SMC Strategy failed: {result.stderr}")
            return {"strategy": "SMC", "symbol": symbol, "error": "FAILED"}
            
    except Exception as e:
        print(f"❌ SMC Strategy error: {e}")
        return {"strategy": "SMC", "symbol": symbol, "error": str(e)}

def simple_macd_strategy(df, initial_capital=100000):
    """Implémentation simple de MACD basée sur quant-trading"""
    
    print("📈 Test MACD Strategy")
    
    # Calculer MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    
    # Signaux de trading
    df['macd'] = macd
    df['signal'] = signal
    df['histogram'] = histogram
    
    # Générer les signaux
    df['position'] = 0
    df.loc[df['macd'] > df['signal'], 'position'] = 1  # Long
    df.loc[df['macd'] < df['signal'], 'position'] = -1  # Short
    
    # Calculer les trades
    df['position_change'] = df['position'].diff()
    
    trades = []
    current_position = 0
    entry_price = 0
    entry_time = None
    
    for i, row in df.iterrows():
        if row['position_change'] != 0 and current_position == 0:
            # Nouvelle position
            current_position = row['position']
            entry_price = row['close']
            entry_time = row['time']
            
        elif row['position_change'] != 0 and current_position != 0:
            # Fermer position et ouvrir nouvelle
            exit_price = row['close']
            pnl_pct = (exit_price - entry_price) / entry_price * current_position
            pnl = initial_capital * pnl_pct * 0.02  # 2% risk per trade
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': row['time'],
                'direction': 'LONG' if current_position > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100
            })
            
            # Nouvelle position
            current_position = row['position']
            entry_price = row['close']
            entry_time = row['time']
    
    if len(trades) == 0:
        return {"strategy": "MACD", "error": "No trades generated"}
    
    trades_df = pd.DataFrame(trades)
    
    # Calculer les statistiques
    total_return = trades_df['pnl_pct'].sum()
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Max drawdown simple
    cumulative_pnl = trades_df['pnl_pct'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()
    
    results = {
        'strategy': 'MACD',
        'return_pct': total_return,
        'trades': total_trades,
        'winrate_pct': win_rate,
        'max_dd_pct': max_dd,
        'avg_r': total_return / total_trades if total_trades > 0 else 0
    }
    
    print(f"✅ MACD: {results['return_pct']:.2f}% ({results['trades']} trades)")
    return results

def simple_bollinger_strategy(df, initial_capital=100000):
    """Implémentation simple de Bollinger Bands"""
    
    print("📊 Test Bollinger Bands Strategy")
    
    # Calculer Bollinger Bands
    window = 20
    df['sma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['sma'] + (df['std'] * 2)
    df['lower_band'] = df['sma'] - (df['std'] * 2)
    
    # Signaux de trading (mean reversion)
    df['position'] = 0
    df.loc[df['close'] < df['lower_band'], 'position'] = 1  # Long quand prix sous bande basse
    df.loc[df['close'] > df['upper_band'], 'position'] = -1  # Short quand prix sur bande haute
    df.loc[(df['close'] >= df['sma']), 'position'] = 0  # Exit sur SMA
    
    # Calculer les trades (logique similaire à MACD)
    df['position_change'] = df['position'].diff()
    
    trades = []
    current_position = 0
    entry_price = 0
    entry_time = None
    
    for i, row in df.iterrows():
        if abs(row['position_change']) > 0 and current_position == 0:
            # Nouvelle position
            current_position = row['position']
            entry_price = row['close']
            entry_time = row['time']
            
        elif row['position'] == 0 and current_position != 0:
            # Fermer position
            exit_price = row['close']
            pnl_pct = (exit_price - entry_price) / entry_price * current_position
            pnl = initial_capital * pnl_pct * 0.02  # 2% risk per trade
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': row['time'],
                'direction': 'LONG' if current_position > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100
            })
            
            current_position = 0
    
    if len(trades) == 0:
        return {"strategy": "Bollinger", "error": "No trades generated"}
    
    trades_df = pd.DataFrame(trades)
    
    # Calculer les statistiques
    total_return = trades_df['pnl_pct'].sum()
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Max drawdown
    cumulative_pnl = trades_df['pnl_pct'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()
    
    results = {
        'strategy': 'Bollinger',
        'return_pct': total_return,
        'trades': total_trades,
        'winrate_pct': win_rate,
        'max_dd_pct': max_dd,
        'avg_r': total_return / total_trades if total_trades > 0 else 0
    }
    
    print(f"✅ Bollinger: {results['return_pct']:.2f}% ({results['trades']} trades)")
    return results

def shooting_star_strategy(df, initial_capital=100000):
    """Implémentation de Shooting Star basée sur quant-trading"""

    print("⭐ Test Shooting Star Strategy")

    # Préparer les données avec les bonnes colonnes
    df_ss = df.copy()
    df_ss.columns = ['time', 'Open', 'High', 'Low', 'Close']  # Majuscules pour compatibilité

    # Paramètres Shooting Star (plus permissifs)
    lower_bound = 0.5  # Plus permissif pour mèche inférieure
    body_size = 1.0    # Plus permissif pour taille du corps

    # Critères Shooting Star
    # 1. Open >= Close (chandelier rouge)
    df_ss['condition1'] = np.where(df_ss['Open'] >= df_ss['Close'], 1, 0)

    # 2. Petite mèche inférieure
    df_ss['condition2'] = np.where(
        (df_ss['Close'] - df_ss['Low']) < lower_bound * abs(df_ss['Close'] - df_ss['Open']), 1, 0)

    # 3. Petit corps
    df_ss['condition3'] = np.where(
        abs(df_ss['Open'] - df_ss['Close']) < abs(np.mean(df_ss['Open'] - df_ss['Close'])) * body_size, 1, 0)

    # 4. Longue mèche supérieure (au moins 2x le corps)
    df_ss['condition4'] = np.where(
        (df_ss['High'] - df_ss['Open']) >= 2 * (df_ss['Open'] - df_ss['Close']), 1, 0)

    # 5. Tendance haussière (prix monte)
    df_ss['condition5'] = np.where(df_ss['Close'] >= df_ss['Close'].shift(1), 1, 0)
    df_ss['condition6'] = np.where(df_ss['Close'].shift(1) >= df_ss['Close'].shift(2), 1, 0)

    # 6. Confirmation : chandelier suivant reste sous le high
    df_ss['condition7'] = np.where(df_ss['High'].shift(-1) <= df_ss['High'], 1, 0)

    # 7. Confirmation : chandelier suivant ferme sous le close
    df_ss['condition8'] = np.where(df_ss['Close'].shift(-1) <= df_ss['Close'], 1, 0)

    # Signal Shooting Star (toutes conditions réunies)
    df_ss['signals'] = (df_ss['condition1'] * df_ss['condition2'] * df_ss['condition3'] *
                       df_ss['condition4'] * df_ss['condition5'] * df_ss['condition6'] *
                       df_ss['condition7'] * df_ss['condition8'])

    # Shooting Star = signal SHORT
    df_ss['signals'] = -df_ss['signals']

    # Générer les trades avec stop loss et holding period
    stop_threshold = 0.05  # 5% stop loss/profit
    holding_period = 7     # 7 jours max

    # Trouver les points d'entrée
    entry_indices = df_ss[df_ss['signals'] == -1].index

    trades = []

    for entry_idx in entry_indices:
        if entry_idx >= len(df_ss) - holding_period:
            continue  # Pas assez de données pour le holding period

        entry_price = df_ss['Close'].iloc[entry_idx]
        entry_time = df_ss['time'].iloc[entry_idx]

        # Chercher la sortie
        exit_found = False
        for i in range(1, holding_period + 1):
            exit_idx = entry_idx + i

            if exit_idx >= len(df_ss):
                break

            current_price = df_ss['Close'].iloc[exit_idx]
            exit_time = df_ss['time'].iloc[exit_idx]

            # Check stop loss/profit (5%)
            price_change = abs(current_price / entry_price - 1)

            if price_change > stop_threshold or i == holding_period:
                # Sortie trouvée
                # Pour SHORT: profit si prix baisse
                pnl_pct = (entry_price - current_price) / entry_price * 100
                pnl = initial_capital * pnl_pct * 0.02  # 2% risk per trade

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_days': i
                })

                exit_found = True
                break

    if len(trades) == 0:
        return {"strategy": "Shooting Star", "error": "No trades generated"}

    trades_df = pd.DataFrame(trades)

    # Calculer les statistiques
    total_return = trades_df['pnl_pct'].sum()
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

    # Max drawdown
    cumulative_pnl = trades_df['pnl_pct'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_dd = drawdown.min()

    # Holding period moyen
    avg_holding = trades_df['holding_days'].mean()

    results = {
        'strategy': 'Shooting Star',
        'return_pct': total_return,
        'trades': total_trades,
        'winrate_pct': win_rate,
        'max_dd_pct': max_dd,
        'avg_r': total_return / total_trades if total_trades > 0 else 0,
        'avg_holding_days': avg_holding
    }

    print(f"✅ Shooting Star: {results['return_pct']:.2f}% ({results['trades']} trades, {avg_holding:.1f}d avg)")
    return results

def compare_strategies():
    """Compare toutes les stratégies sur les mêmes données TwelveData"""
    
    print("🔬 COMPARAISON STRATÉGIES - MÊMES DONNÉES TWELVEDATA")
    print("=" * 70)
    
    # Configuration
    api_key = "8af42105d7754290bc090dfb3a6ca6d4"
    symbol = "USD/JPY"
    start_date = "2024-11-15"
    end_date = "2025-01-15"
    
    print(f"📅 Période: {start_date} → {end_date}")
    print(f"💱 Symbole: {symbol}")
    print(f"📊 Source: TwelveData (vraies données)")
    print()
    
    # Récupérer les données TwelveData
    try:
        df = fetch_twelvedata_for_comparison(api_key, symbol, "15min", start_date, end_date)
        print(f"✅ Données récupérées: {len(df)} candles 15min")
    except Exception as e:
        print(f"❌ Erreur récupération données: {e}")
        return
    
    # Tester toutes les stratégies
    results = []
    
    # 1. Notre stratégie SMC
    smc_result = run_smc_strategy(api_key, symbol, start_date, end_date)
    results.append(smc_result)
    
    # 2. MACD Strategy
    macd_result = simple_macd_strategy(df.copy())
    results.append(macd_result)
    
    # 3. Bollinger Bands Strategy
    bollinger_result = simple_bollinger_strategy(df.copy())
    results.append(bollinger_result)

    # 4. Shooting Star Strategy
    shooting_star_result = shooting_star_strategy(df.copy())
    results.append(shooting_star_result)
    
    # Afficher les résultats comparatifs
    print(f"\n" + "=" * 70)
    print("📊 RÉSULTATS COMPARATIFS")
    print("=" * 70)
    
    # Filtrer les résultats réussis
    successful_results = [r for r in results if 'error' not in r and 'return_pct' in r]
    
    if successful_results:
        # Trier par performance
        successful_results.sort(key=lambda x: x['return_pct'], reverse=True)
        
        print(f"{'Rang':<4} {'Stratégie':<12} {'Return':<8} {'Trades':<7} {'Win Rate':<9} {'Max DD':<8}")
        print("-" * 60)
        
        for i, result in enumerate(successful_results, 1):
            return_pct = result.get('return_pct', 0)
            trades = result.get('trades', 0)
            winrate = result.get('winrate_pct', 0)
            max_dd = result.get('max_dd_pct', 0)
            
            print(f"{i:<4} {result['strategy']:<12} {return_pct:>6.2f}% {trades:>6} {winrate:>7.1f}% {max_dd:>6.2f}%")
        
        # Meilleure stratégie
        best = successful_results[0]
        print(f"\n🏆 GAGNANT: {best['strategy']}")
        print(f"💰 Return: {best['return_pct']:.2f}%")
        print(f"📈 Trades: {best['trades']}")
        print(f"🎯 Win Rate: {best.get('winrate_pct', 0):.1f}%")
        
    else:
        print("❌ Aucun résultat valide à comparer")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    compare_strategies()

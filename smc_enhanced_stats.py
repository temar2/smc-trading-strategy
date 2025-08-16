#!/usr/bin/env python3
"""
Enhanced Statistics Module for SMC Strategy
Intégration des métriques avancées du repository quant-trading
"""

import pandas as pd
import numpy as np
import scipy.integrate
import scipy.stats
import yfinance as yf
from datetime import datetime

def omega_ratio(returns, risk_free_rate=0.0):
    """
    Calcule l'Omega Ratio - variation du Sharpe Ratio
    Utilise la distribution Student T au lieu de normale
    """
    degree_of_freedom = len(returns) - 1
    maximum = np.max(returns)
    minimum = np.min(returns)
    
    # Intégrale des returns au-dessus du seuil
    y = scipy.integrate.quad(
        lambda g: 1 - scipy.stats.t.cdf(g, degree_of_freedom),
        risk_free_rate, maximum
    )
    
    # Intégrale des returns en-dessous du seuil
    x = scipy.integrate.quad(
        lambda g: scipy.stats.t.cdf(g, degree_of_freedom),
        minimum, risk_free_rate
    )
    
    if x[0] == 0:
        return np.inf
    
    return y[0] / x[0]

def sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calcule le Sortino Ratio - focus sur la déviation négative
    """
    degree_of_freedom = len(returns) - 1
    growth_rate = np.mean(returns)
    minimum = np.min(returns)
    
    # Écart-type des returns négatifs seulement
    v = np.sqrt(np.abs(scipy.integrate.quad(
        lambda g: ((risk_free_rate - g) ** 2) * scipy.stats.t.pdf(g, degree_of_freedom),
        risk_free_rate, minimum
    )))
    
    if v[0] == 0:
        return np.inf
    
    return (growth_rate - risk_free_rate) / v[0]

def maximum_drawdown(equity_curve):
    """
    Calcule le Maximum Drawdown
    """
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calmar_ratio(returns, max_dd):
    """
    Calcule le Calmar Ratio = CAGR / |Max Drawdown|
    """
    if max_dd == 0:
        return np.inf
    
    cagr = (1 + np.mean(returns)) ** 252 - 1  # Annualisé
    return cagr / abs(max_dd)

def calculate_enhanced_stats(trades_df, initial_capital=100000, benchmark_symbol="^GSPC"):
    """
    Calcule toutes les statistiques avancées pour notre stratégie SMC
    """
    
    if len(trades_df) == 0:
        return {"error": "No trades to analyze"}
    
    # Créer la courbe d'équité
    trades_df = trades_df.copy()
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['equity'] = initial_capital + trades_df['cumulative_pnl']
    trades_df['returns'] = trades_df['equity'].pct_change().fillna(0)
    
    # Statistiques de base
    total_return = (trades_df['equity'].iloc[-1] / initial_capital) - 1
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculs avancés
    returns = trades_df['returns'].dropna()
    
    if len(returns) == 0:
        returns = pd.Series([0])
    
    # CAGR (Compound Annual Growth Rate)
    days_traded = (trades_df.index[-1] - trades_df.index[0]).days if len(trades_df) > 1 else 1
    cagr = (trades_df['equity'].iloc[-1] / initial_capital) ** (365.25 / days_traded) - 1
    
    # Maximum Drawdown
    max_dd = maximum_drawdown(trades_df['equity'])
    
    # Volatilité
    volatility = returns.std() * np.sqrt(252)  # Annualisée
    
    # Sharpe Ratio (assumant risk-free rate = 0)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
    
    # Ratios avancés
    try:
        omega = omega_ratio(returns.values)
    except:
        omega = 0
    
    try:
        sortino = sortino_ratio(returns.values)
    except:
        sortino = 0
    
    # Calmar Ratio
    calmar = calmar_ratio(returns.values, max_dd)
    
    # Profit Factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Average Trade
    avg_trade = trades_df['pnl'].mean()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    # Expectancy
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Recovery Factor
    recovery_factor = total_return / abs(max_dd) if max_dd != 0 else np.inf
    
    # Benchmark comparison (optionnel)
    benchmark_return = 0
    try:
        start_date = trades_df.index[0].strftime('%Y-%m-%d')
        end_date = trades_df.index[-1].strftime('%Y-%m-%d')
        benchmark = yf.download(benchmark_symbol, start=start_date, end=end_date)
        if len(benchmark) > 0:
            benchmark_return = float((benchmark['Close'].iloc[-1] / benchmark['Close'].iloc[0]) - 1)
    except:
        pass
    
    # Compiler les résultats
    stats = {
        # Statistiques de base
        'Total Return': f"{total_return:.2%}",
        'CAGR': f"{cagr:.2%}",
        'Total Trades': total_trades,
        'Win Rate': f"{win_rate:.2%}",
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        
        # Métriques de risque
        'Maximum Drawdown': f"{max_dd:.2%}",
        'Volatility (Annual)': f"{volatility:.2%}",
        
        # Ratios de performance
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Sortino Ratio': f"{sortino:.3f}",
        'Calmar Ratio': f"{calmar:.3f}",
        'Omega Ratio': f"{omega:.3f}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Recovery Factor': f"{recovery_factor:.2f}",
        
        # Analyse des trades
        'Average Trade': f"${avg_trade:.2f}",
        'Average Win': f"${avg_win:.2f}",
        'Average Loss': f"${avg_loss:.2f}",
        'Expectancy': f"${expectancy:.2f}",
        
        # Benchmark
        'Benchmark Return': f"{benchmark_return:.2%}",
        'Excess Return': f"{total_return - benchmark_return:.2%}"
    }
    
    return stats

def print_enhanced_stats(stats):
    """
    Affiche les statistiques de manière formatée
    """
    print("\n" + "="*60)
    print("📊 ENHANCED STATISTICS - SMC STRATEGY")
    print("="*60)
    
    print(f"\n🎯 PERFORMANCE METRICS")
    print(f"Total Return:        {stats['Total Return']}")
    print(f"CAGR:               {stats['CAGR']}")
    print(f"Benchmark Return:    {stats['Benchmark Return']}")
    print(f"Excess Return:       {stats['Excess Return']}")
    
    print(f"\n📈 TRADE ANALYSIS")
    print(f"Total Trades:        {stats['Total Trades']}")
    print(f"Win Rate:           {stats['Win Rate']}")
    print(f"Winning Trades:      {stats['Winning Trades']}")
    print(f"Losing Trades:       {stats['Losing Trades']}")
    
    print(f"\n⚠️ RISK METRICS")
    print(f"Maximum Drawdown:    {stats['Maximum Drawdown']}")
    print(f"Volatility:         {stats['Volatility (Annual)']}")
    
    print(f"\n🏆 PERFORMANCE RATIOS")
    print(f"Sharpe Ratio:        {stats['Sharpe Ratio']}")
    print(f"Sortino Ratio:       {stats['Sortino Ratio']}")
    print(f"Calmar Ratio:        {stats['Calmar Ratio']}")
    print(f"Omega Ratio:         {stats['Omega Ratio']}")
    print(f"Profit Factor:       {stats['Profit Factor']}")
    print(f"Recovery Factor:     {stats['Recovery Factor']}")
    
    print(f"\n💰 TRADE PROFITABILITY")
    print(f"Average Trade:       {stats['Average Trade']}")
    print(f"Average Win:         {stats['Average Win']}")
    print(f"Average Loss:        {stats['Average Loss']}")
    print(f"Expectancy:         {stats['Expectancy']}")
    
    print("="*60)

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer des données de test
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    test_trades = pd.DataFrame({
        'pnl': [100, -50, 200, -30, 150, -80, 300, -40, 120, -60]
    }, index=dates)
    
    stats = calculate_enhanced_stats(test_trades)
    print_enhanced_stats(stats)

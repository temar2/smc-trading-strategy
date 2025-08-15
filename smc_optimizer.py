#!/usr/bin/env python3
"""
SMC Strategy Autonomous Optimizer
Optimise les paramètres de la stratégie SMC selon les contraintes définies
"""

import subprocess
import pandas as pd
import json
import os
import re
from datetime import datetime
from itertools import product
import time

class SMCOptimizer:
    def __init__(self):
        # Contraintes du set valide
        self.max_drawdown_limit = 1.50
        self.trades_min = 6
        self.trades_max = 30
        self.avg_r_min = 0.10
        
        # Paramètres à optimiser + bornes
        self.param_ranges = {
            'rr_target': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            'rr_target_alt': [1.0, 1.1, 1.2, 1.3, 1.4],
            'atr_min_pips': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            'momentum_min_body_atr': [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
            'forbidden_hours_utc': [
                "",  # D: aucune restriction (24h/24)
                "0,1,2,3,4,5,6,7,20,21,22,23",  # A: autorise 08-19
                "0,1,2,3,4,5,6,7,19,20,21,22,23",  # B: autorise 08-18
                "0,1,2,3,4,5,6,7,18,19,20,21,22,23"  # C: autorise 08-17
            ]
        }
        
        # Hyperparamètres fixes
        self.fixed_params = {
            'api_key': '8af42105d7754290bc090dfb3a6ca6d4',
            'symbol': 'GBP/USD',
            'ltf': '15min',
            'htf': '4h',
            'start': '2025-07-16',
            'end': '2025-08-16',
            'capital': 100000,
            'risk_per_trade': 0.0045,
            'spread_pips': 0.2,
            'slip_entry_pips': 0.1,
            'slip_exit_pips': 0.1,
            'commission_per_million': 7,
            'leverage_max': 30,
            'lot_min': 1000,
            'lot_step': 1000,
            'max_lot': 2.0,
            'atr_period': 14,
            'atr_alpha': 0.4,
            'min_stop_pips': 3.0,
            'hard_min_stop_pips': 3.0,
            'buffer_sl_pips': 0.25,
            'atr_rr_switch_pips': 9,
            'partial_take_r': 1.0,
            'move_be_r': 0.7,
            'trail_struct_window': 3,
            'pending_ttl_bars': 60,
            'fallback_market_bars': 24,
            'pullback_bars': 2,
            'news_free_minutes': 15,
            'session_start_utc': 6,
            'session_end_utc': 20,
            'confluence_min': 2
        }
        
        self.results_file = "results_refined.csv"
        self.best_params_file = "best_params.json"
        self.leaderboard_file = "leaderboard_top5.csv"
        self.readme_file = "README_OPTIM.md"
        
        # Initialiser les fichiers de résultats
        self.init_results_file()
        
    def init_results_file(self):
        """Initialise le fichier de résultats s'il n'existe pas"""
        if not os.path.exists(self.results_file):
            columns = ['timestamp', 'rr_target', 'rr_target_alt', 'atr_min_pips', 
                      'momentum_min_body_atr', 'forbidden_hours_utc', 'final_equity', 
                      'return_pct', 'trades', 'avg_R', 'max_drawdown_pct', 'score', 'status']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.results_file, index=False)
    
    def calculate_score(self, return_pct, avg_R, trades):
        """Calcule le score selon la fonction objectif"""
        penalty = max(0, (trades - 20) / 10) if trades > 20 else 0
        return return_pct + 0.5 * avg_R - 0.2 * penalty
    
    def is_valid_set(self, metrics):
        """Vérifie si un set respecte les contraintes"""
        return (metrics['max_drawdown_pct'] <= self.max_drawdown_limit and
                self.trades_min <= metrics['trades'] <= self.trades_max and
                metrics['avg_R'] >= self.avg_r_min)
    
    def build_command(self, params):
        """Construit la commande de backtest"""
        cmd_parts = [
            "python", "smc_backtest_v2.5.py",
            f"--api-key {self.fixed_params['api_key']}",
            f"--symbol \"{self.fixed_params['symbol']}\"",
            f"--ltf {self.fixed_params['ltf']}",
            f"--htf {self.fixed_params['htf']}",
            f"--start {self.fixed_params['start']}",
            f"--end {self.fixed_params['end']}",
            f"--capital {self.fixed_params['capital']}",
            f"--risk_per_trade {self.fixed_params['risk_per_trade']}",
            f"--rr_target {params['rr_target']}",
            f"--rr_target_alt {params['rr_target_alt']}",
            f"--atr_min_pips {params['atr_min_pips']}",
            f"--momentum_min_body_atr {params['momentum_min_body_atr']}",
            "--auto-bias",
            "--require_confluence",
            f"--confluence_min {self.fixed_params['confluence_min']}"
        ]

        # Ajouter forbidden_hours_utc seulement si non vide
        if params['forbidden_hours_utc']:
            cmd_parts.append(f"--forbidden_hours_utc {params['forbidden_hours_utc']}")

        return " ".join(cmd_parts)
    
    def run_backtest(self, params):
        """Exécute un backtest avec les paramètres donnés"""
        cmd = self.build_command(params)
        print(f"🚀 Running: {params}")
        
        try:
            # Utiliser encoding utf-8 pour éviter les problèmes d'emojis
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=180, encoding='utf-8', errors='ignore')
            output = result.stdout
            
            # Parser les métriques de sortie
            metrics = self.parse_output(output)
            if metrics:
                metrics['score'] = self.calculate_score(metrics['return_pct'], metrics['avg_R'], metrics['trades'])
                metrics['status'] = 'success'
            else:
                metrics = {'final_equity': 0, 'return_pct': 0, 'trades': 0, 'avg_R': 0, 
                          'max_drawdown_pct': 999, 'score': -999, 'status': 'error'}
            
            # Sauvegarder le résultat
            self.save_result(params, metrics)
            return metrics
            
        except subprocess.TimeoutExpired:
            print("⏰ Timeout - skipping")
            metrics = {'final_equity': 0, 'return_pct': 0, 'trades': 0, 'avg_R': 0, 
                      'max_drawdown_pct': 999, 'score': -999, 'status': 'timeout'}
            self.save_result(params, metrics)
            return metrics
        except Exception as e:
            print(f"❌ Error: {e}")
            metrics = {'final_equity': 0, 'return_pct': 0, 'trades': 0, 'avg_R': 0, 
                      'max_drawdown_pct': 999, 'score': -999, 'status': 'error'}
            self.save_result(params, metrics)
            return metrics
    
    def parse_output(self, output):
        """Parse les métriques depuis la sortie du backtest"""
        metrics = {}
        patterns = {
            'final_equity': r'final_equity:\s*([\d.]+)',
            'return_pct': r'return_pct:\s*([\d.-]+)%',
            'trades': r'trades:\s*(\d+)',
            'avg_R': r'avg_R:\s*([\d.-]+)',
            'max_drawdown_pct': r'max_drawdown_pct:\s*([\d.-]+)%'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                value = float(match.group(1))
                metrics[key] = int(value) if key == 'trades' else value
            else:
                return None
        
        return metrics if len(metrics) == 5 else None
    
    def save_result(self, params, metrics):
        """Sauvegarde un résultat dans le fichier CSV"""
        row = {
            'timestamp': datetime.now().isoformat(),
            **params,
            **metrics
        }
        
        df = pd.DataFrame([row])
        df.to_csv(self.results_file, mode='a', header=False, index=False)

    def load_existing_results(self):
        """Charge les résultats existants"""
        if os.path.exists(self.results_file):
            return pd.read_csv(self.results_file)
        return pd.DataFrame()

    def get_best_valid_set(self):
        """Retourne le meilleur set valide actuel"""
        df = self.load_existing_results()
        if df.empty:
            return None

        # Filtrer les sets valides
        valid_df = df[
            (df['max_drawdown_pct'] <= self.max_drawdown_limit) &
            (df['trades'] >= self.trades_min) &
            (df['trades'] <= self.trades_max) &
            (df['avg_R'] >= self.avg_r_min) &
            (df['status'] == 'success')
        ]

        if valid_df.empty:
            return None

        # Retourner le meilleur par score
        best = valid_df.loc[valid_df['score'].idxmax()]
        return {
            'rr_target': best['rr_target'],
            'rr_target_alt': best['rr_target_alt'],
            'atr_min_pips': best['atr_min_pips'],
            'momentum_min_body_atr': best['momentum_min_body_atr'],
            'forbidden_hours_utc': best['forbidden_hours_utc']
        }, best.to_dict()

    def generate_coarse_grid(self):
        """Génère le coarse grid initial"""
        grid_params = {
            'rr_target': [1.3, 1.5, 1.7],
            'rr_target_alt': [1.1, 1.2],
            'atr_min_pips': [0.8, 1.0, 1.2],  # Plus permissif
            'momentum_min_body_atr': [0.35, 0.45, 0.55],  # Plus permissif
            'forbidden_hours_utc': [
                "",  # D: aucune restriction
                "0,1,2,3,4,5,6,7,20,21,22,23"  # A
            ]
        }

        combinations = list(product(*grid_params.values()))
        param_sets = []
        for combo in combinations:
            param_set = dict(zip(grid_params.keys(), combo))
            param_sets.append(param_set)

        return param_sets

    def generate_neighbors(self, best_params):
        """Génère les voisins d'un set de paramètres"""
        neighbors = []

        # Voisins pour rr_target
        current_rr = best_params['rr_target']
        for delta in [-0.1, 0.1]:
            new_rr = round(current_rr + delta, 1)
            if new_rr in self.param_ranges['rr_target']:
                neighbor = best_params.copy()
                neighbor['rr_target'] = new_rr
                neighbors.append(neighbor)

        # Voisins pour rr_target_alt
        current_rr_alt = best_params['rr_target_alt']
        for delta in [-0.1, 0.1]:
            new_rr_alt = round(current_rr_alt + delta, 1)
            if new_rr_alt in self.param_ranges['rr_target_alt']:
                neighbor = best_params.copy()
                neighbor['rr_target_alt'] = new_rr_alt
                neighbors.append(neighbor)

        # Voisins pour atr_min_pips
        current_atr = best_params['atr_min_pips']
        for delta in [-0.1, 0.1]:
            new_atr = round(current_atr + delta, 1)
            if new_atr in self.param_ranges['atr_min_pips']:
                neighbor = best_params.copy()
                neighbor['atr_min_pips'] = new_atr
                neighbors.append(neighbor)

        # Voisins pour momentum_min_body_atr
        current_mom = best_params['momentum_min_body_atr']
        for delta in [-0.05, 0.05]:
            new_mom = round(current_mom + delta, 2)
            if new_mom in self.param_ranges['momentum_min_body_atr']:
                neighbor = best_params.copy()
                neighbor['momentum_min_body_atr'] = new_mom
                neighbors.append(neighbor)

        # Variantes de forbidden_hours_utc
        for forbidden in self.param_ranges['forbidden_hours_utc']:
            if forbidden != best_params['forbidden_hours_utc']:
                neighbor = best_params.copy()
                neighbor['forbidden_hours_utc'] = forbidden
                neighbors.append(neighbor)

        return neighbors

    def save_outputs(self, best_params, best_metrics):
        """Sauvegarde tous les fichiers de sortie"""
        # best_params.json
        with open(self.best_params_file, 'w') as f:
            json.dump({**best_params, **best_metrics}, f, indent=2)

        # Commandes one-liner
        cmd = self.build_command(best_params)

        # PowerShell
        with open("best_cmd.ps1", 'w') as f:
            f.write(cmd + "\n")

        # Bash
        with open("best_cmd.sh", 'w') as f:
            f.write("#!/bin/bash\n" + cmd + "\n")

        # Leaderboard top 5
        df = self.load_existing_results()
        valid_df = df[
            (df['max_drawdown_pct'] <= self.max_drawdown_limit) &
            (df['trades'] >= self.trades_min) &
            (df['trades'] <= self.trades_max) &
            (df['avg_R'] >= self.avg_r_min) &
            (df['status'] == 'success')
        ].sort_values('score', ascending=False).head(5)

        valid_df.to_csv(self.leaderboard_file, index=False)

        # README
        with open(self.readme_file, 'w') as f:
            f.write(f"# SMC Strategy Optimization Results\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Best Parameters\n")
            f.write(f"- RR Target: {best_params['rr_target']}\n")
            f.write(f"- RR Target Alt: {best_params['rr_target_alt']}\n")
            f.write(f"- ATR Min Pips: {best_params['atr_min_pips']}\n")
            f.write(f"- Momentum Min Body ATR: {best_params['momentum_min_body_atr']}\n")
            f.write(f"- Forbidden Hours UTC: {best_params['forbidden_hours_utc']}\n\n")
            f.write(f"## Best Metrics\n")
            f.write(f"- Return: {best_metrics['return_pct']:.2f}%\n")
            f.write(f"- Trades: {best_metrics['trades']}\n")
            f.write(f"- Avg R: {best_metrics['avg_R']:.3f}\n")
            f.write(f"- Max Drawdown: {best_metrics['max_drawdown_pct']:.2f}%\n")
            f.write(f"- Final Equity: {best_metrics['final_equity']:.2f}\n")
            f.write(f"- Score: {best_metrics['score']:.3f}\n\n")
            f.write(f"## Command\n")
            f.write(f"```bash\n{cmd}\n```\n")

    def run_optimization(self, max_iterations=5, max_runs=80):
        """Exécute l'optimisation complète"""
        print("🎯 SMC Strategy Autonomous Optimizer")
        print("=" * 50)

        total_runs = 0

        # Étape 1: Seed (charger existant ou coarse grid)
        best_result = self.get_best_valid_set()

        if best_result is None:
            print("📊 Aucun résultat valide trouvé - Exécution du coarse grid...")
            coarse_sets = self.generate_coarse_grid()
            print(f"🔄 {len(coarse_sets)} combinaisons à tester")

            for i, params in enumerate(coarse_sets):
                if total_runs >= max_runs:
                    break
                print(f"[{i+1}/{len(coarse_sets)}] ", end="")
                metrics = self.run_backtest(params)
                total_runs += 1

                if self.is_valid_set(metrics):
                    print(f"✅ Score: {metrics['score']:.3f}")
                else:
                    print(f"❌ Invalid: DD={metrics['max_drawdown_pct']:.2f}%, T={metrics['trades']}, R={metrics['avg_R']:.3f}")

            # Récupérer le meilleur après coarse grid
            best_result = self.get_best_valid_set()

        if best_result is None:
            print("❌ Aucun set valide trouvé même après coarse grid!")
            return

        best_params, best_metrics = best_result
        print(f"\n🏆 Meilleur set initial: Score={best_metrics['score']:.3f}")

        # Étape 2: Local search
        no_improvement_count = 0

        for iteration in range(max_iterations):
            if total_runs >= max_runs:
                break

            print(f"\n🔍 Itération {iteration + 1}/{max_iterations}")
            neighbors = self.generate_neighbors(best_params)
            print(f"🔄 {len(neighbors)} voisins à tester")

            iteration_improved = False

            for i, neighbor_params in enumerate(neighbors):
                if total_runs >= max_runs:
                    break

                print(f"[{i+1}/{len(neighbors)}] ", end="")
                metrics = self.run_backtest(neighbor_params)
                total_runs += 1

                if self.is_valid_set(metrics) and metrics['score'] > best_metrics['score']:
                    improvement = metrics['score'] - best_metrics['score']
                    if improvement >= 0.05:  # Amélioration significative
                        best_params = neighbor_params
                        best_metrics = metrics
                        iteration_improved = True
                        print(f"🚀 Nouveau meilleur! Score: {metrics['score']:.3f} (+{improvement:.3f})")
                    else:
                        print(f"✅ Score: {metrics['score']:.3f} (amélioration mineure)")
                elif self.is_valid_set(metrics):
                    print(f"✅ Score: {metrics['score']:.3f}")
                else:
                    print(f"❌ Invalid: DD={metrics['max_drawdown_pct']:.2f}%, T={metrics['trades']}, R={metrics['avg_R']:.3f}")

            if not iteration_improved:
                no_improvement_count += 1
                if no_improvement_count >= 2:
                    print("🏁 Convergence atteinte (pas d'amélioration sur 2 itérations)")
                    break
            else:
                no_improvement_count = 0

        # Sauvegarder les résultats finaux
        self.save_outputs(best_params, best_metrics)

        print(f"\n🎉 Optimisation terminée!")
        print(f"📊 Total runs: {total_runs}")
        print(f"🏆 Meilleur score: {best_metrics['score']:.3f}")
        print(f"💰 Return: {best_metrics['return_pct']:.2f}%")
        print(f"📈 Trades: {best_metrics['trades']}")
        print(f"⚖️ Avg R: {best_metrics['avg_R']:.3f}")
        print(f"📉 Max DD: {best_metrics['max_drawdown_pct']:.2f}%")

        return best_params, best_metrics

if __name__ == "__main__":
    optimizer = SMCOptimizer()
    optimizer.run_optimization()

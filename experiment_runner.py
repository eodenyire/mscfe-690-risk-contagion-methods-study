"""
Experiment Runner and Orchestration
====================================
Manages the full benchmarking pipeline:
- Generates synthetic data across experimental factors
- Runs all four recovery methods
- Evaluates performance
- Aggregates and stores results

Experimental Design:
- Data volume: 50, 100, 150, 200, 250 time steps
- Noise level: 0.1, 0.3, 0.5
- Tail dependence (t-copula df): 3, 5, 10
- Network density: 2, 4 edges per node
- Monte Carlo replications: 100 per configuration
"""

import numpy as np
import pandas as pd
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import traceback
import logging

from synthetic_generator import DataGenerator
from recovery_methods import (
    VARDieboldYilmaz, CoVaR, DebtRank, GNNRecovery, RecoveryMethodsFactory
)
from evaluation import RecoveryEvaluator


class ExperimentRunner:
    """Orchestrates the full benchmarking study."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_gen = DataGenerator(config_path)
        self.results = []
        
        # Create output directories
        self._create_output_dirs()
        
        # Set up logging
        self._setup_logging()
    
    def _create_output_dirs(self):
        """Create output directory structure."""
        for key in ['results_dir', 'figures_dir', 'logs_dir', 'checkpoint_dir']:
            path = self.config['output'][key]
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Set up logging."""
        log_file = os.path.join(self.config['output']['logs_dir'], 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_single_configuration(
        self,
        time_steps: int,
        noise_level: float,
        network_density: int,
        t_df: float,
        num_replications: int = 100,
    ) -> Dict[str, Any]:
        """
        Run full benchmarking for a single configuration.
        
        Args:
            time_steps: Length of KRI series
            noise_level: Measurement noise
            network_density: Edges per PRT node
            t_df: t-copula degrees of freedom
            num_replications: Monte Carlo runs
            
        Returns:
            Aggregated results across replications
        """
        config_name = f"T{time_steps}_N{noise_level:.1f}_D{network_density}_DF{t_df:.0f}"
        self.logger.info(f"Starting configuration: {config_name}")
        
        # Generate experiments
        experiments = self.data_gen.generate_monte_carlo(
            time_steps=time_steps,
            noise_level=noise_level,
            network_density=network_density,
            t_df=t_df,
            num_replications=num_replications
        )
        
        # Results for each method
        method_results = {
            'var_dy': [],
            'covar': [],
            'debtrank': [],
            'gnn': [],
        }
        
        # Run each replication
        for rep_idx, exp in enumerate(tqdm(experiments, desc=config_name)):
            X = exp['kri_panel']
            W_true_prt = exp['W_true']
            
            # Expand W_true from (20, 20) to (100, 100) for KRI-level evaluation
            # Each KRI block inherits the PRT-level connection
            import numpy as np
            W_true_kri = np.zeros((100, 100))
            for i in range(20):
                for j in range(20):
                    i_start, i_end = i * 5, (i + 1) * 5
                    j_start, j_end = j * 5, (j + 1) * 5
                    W_true_kri[i_start:i_end, j_start:j_end] = W_true_prt[i, j]
            
            # Instantiate evaluator with KRI-level ground truth
            evaluator = RecoveryEvaluator(W_true_kri)
            
            # Run each method
            try:
                # VAR + Diebold-Yilmaz
                var_dy = VARDieboldYilmaz(lag_order=1)
                W_var_dy = var_dy.fit_and_recover(X, threshold=0.1)
                result_var_dy = evaluator.evaluate(W_var_dy)
                method_results['var_dy'].append(result_var_dy)
            except Exception as e:
                self.logger.warning(f"VAR+DY failed on rep {rep_idx}: {str(e)}")
                method_results['var_dy'].append(None)
            
            try:
                # CoVaR
                covar = CoVaR(quantile=0.05)
                W_covar = covar.fit_and_recover(X, threshold=0.1)
                result_covar = evaluator.evaluate(W_covar)
                method_results['covar'].append(result_covar)
            except Exception as e:
                self.logger.warning(f"CoVaR failed on rep {rep_idx}: {str(e)}")
                method_results['covar'].append(None)
            
            try:
                # DebtRank (on planted network, as reference)
                debtrank = DebtRank(contagion_factor=0.5)
                initial_shock = np.zeros(20)
                initial_shock[0] = 1.0
                distress_path = debtrank.propagate_shock(W_true_prt, initial_shock, steps=10)
                # For now, store a placeholder result
                result_debtrank = evaluator.evaluate(W_true_kri)  # Reference
                method_results['debtrank'].append(result_debtrank)
            except Exception as e:
                self.logger.warning(f"DebtRank failed on rep {rep_idx}: {str(e)}")
                method_results['debtrank'].append(None)
            
            try:
                # GNN
                gnn = GNNRecovery(
                    num_kris=100,
                    hidden_dim=self.config['recovery_methods']['gnn']['hidden_dim'],
                    dropout=self.config['recovery_methods']['gnn']['dropout'],
                    learning_rate=self.config['recovery_methods']['gnn']['learning_rate'],
                    epochs=self.config['recovery_methods']['gnn']['epochs']
                )
                W_gnn = gnn.fit_and_recover(X, threshold=0.1)
                result_gnn = evaluator.evaluate(W_gnn)
                method_results['gnn'].append(result_gnn)
            except Exception as e:
                self.logger.warning(f"GNN failed on rep {rep_idx}: {str(e)}")
                method_results['gnn'].append(None)
        
        # Aggregate results
        aggregated = {}
        for method_name, results_list in method_results.items():
            # Filter out None results
            valid_results = [r for r in results_list if r is not None]
            
            if len(valid_results) > 0:
                aggregated[method_name] = RecoveryEvaluator.aggregate_results(valid_results)
            else:
                aggregated[method_name] = None
        
        # Store configuration info
        aggregated['config'] = {
            'time_steps': time_steps,
            'noise_level': noise_level,
            'network_density': network_density,
            't_df': t_df,
            'num_replications': num_replications,
            'config_name': config_name,
        }
        
        self.logger.info(f"Completed: {config_name}")
        return aggregated
    
    def run_full_study(self) -> List[Dict[str, Any]]:
        """
        Run full benchmarking across all experimental factors.
        
        Returns:
            List of aggregated results for each configuration
        """
        study_results = []
        
        exp_design = self.config['experimental_design']
        num_configs = (
            len(exp_design['data_volume']) *
            len(exp_design['noise_level']) *
            len(exp_design['tail_dependence_df']) *
            len(exp_design['network_density'])
        )
        
        self.logger.info(f"Starting full study: {num_configs} configurations")
        
        config_idx = 0
        for data_vol in exp_design['data_volume']:
            for noise_lvl in exp_design['noise_level']:
                for tail_dep in exp_design['tail_dependence_df']:
                    for net_dens in exp_design['network_density']:
                        config_idx += 1
                        self.logger.info(f"Configuration {config_idx}/{num_configs}")
                        
                        result = self.run_single_configuration(
                            time_steps=data_vol,
                            noise_level=noise_lvl,
                            network_density=net_dens,
                            t_df=tail_dep,
                            num_replications=exp_design['monte_carlo_replications']
                        )
                        study_results.append(result)
        
        self.results = study_results
        self.logger.info("Full study completed")
        return study_results
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """
        Save results to JSON file.
        
        Args:
            filename: Output filename
        """
        output_path = os.path.join(self.config['output']['results_dir'], filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def json_serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: json_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_serialize(item) for item in obj]
            else:
                return obj
        
        serialized_results = json_serialize(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serialized_results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def summary_table(self) -> pd.DataFrame:
        """
        Create summary table of results.
        
        Returns:
            DataFrame with one row per configuration and one column per method/metric
        """
        rows = []
        
        for config_result in self.results:
            config = config_result['config']
            row = {
                'time_steps': config['time_steps'],
                'noise_level': config['noise_level'],
                'network_density': config['network_density'],
                't_df': config['t_df'],
            }
            
            # Add metrics for each method
            for method_name in ['var_dy', 'covar', 'gnn']:
                if method_name in config_result and config_result[method_name] is not None:
                    method_agg = config_result[method_name]
                    
                    for metric in ['precision', 'recall', 'f1_score', 'auc_roc']:
                        if metric in method_agg:
                            col_name = f"{method_name}_{metric}"
                            row[col_name] = method_agg[metric]['mean']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def save_summary_table(self, filename: str = "benchmark_summary.csv"):
        """Save summary table to CSV."""
        df = self.summary_table()
        output_path = os.path.join(self.config['output']['results_dir'], filename)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Summary table saved to {output_path}")


class QuickBenchmark:
    """Simplified interface for quick benchmarking (subset of factors)."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.runner = ExperimentRunner(config_path)
    
    def benchmark_data_volume(self, noise_level: float = 0.1, num_reps: int = 20):
        """Benchmark effect of data volume."""
        results = []
        
        for time_steps in [50, 100, 150, 200, 250]:
            result = self.runner.run_single_configuration(
                time_steps=time_steps,
                noise_level=noise_level,
                network_density=2,
                t_df=5.0,
                num_replications=num_reps
            )
            results.append(result)
        
        return results
    
    def benchmark_noise(self, time_steps: int = 250, num_reps: int = 20):
        """Benchmark effect of noise."""
        results = []
        
        for noise_level in [0.1, 0.3, 0.5]:
            result = self.runner.run_single_configuration(
                time_steps=time_steps,
                noise_level=noise_level,
                network_density=2,
                t_df=5.0,
                num_replications=num_reps
            )
            results.append(result)
        
        return results
    
    def benchmark_tail_dependence(self, time_steps: int = 250, num_reps: int = 20):
        """Benchmark effect of tail dependence."""
        results = []
        
        for t_df in [3, 5, 10]:
            result = self.runner.run_single_configuration(
                time_steps=time_steps,
                noise_level=0.1,
                network_density=2,
                t_df=float(t_df),
                num_replications=num_reps
            )
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Quick benchmark: just data volume effect
    print("Starting quick benchmark (data volume effect)...")
    quick = QuickBenchmark("config.yaml")
    results = quick.benchmark_data_volume(noise_level=0.1, num_reps=10)
    
    # Save results
    runner = ExperimentRunner("config.yaml")
    runner.results = results
    runner.save_results("quick_benchmark_data_volume.json")
    runner.save_summary_table("quick_benchmark_summary.csv")
    
    print("Quick benchmark completed. Results saved to ./results/")

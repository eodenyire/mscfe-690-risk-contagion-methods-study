"""
Minimal Visualization Module
============================
Produces publication-ready figures for the methods study:

1. Adjacency Heatmaps: Compare planted network with recovered networks
2. Degradation Curves: Show how F1 and AUC-ROC degrade with data volume and noise
3. ROC Curves: Compare methods on ROC curves
4. Weight Correlation: Scatter plots of estimated vs. true edge weights

Focus: Interpretability and communication, not interactive dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import json
import os


class AdjacencyHeatmaps:
    """Visualize adjacency matrices as heatmaps."""
    
    @staticmethod
    def plot_comparison(W_true: np.ndarray, W_est_dict: Dict[str, np.ndarray], 
                       output_path: str = None, figsize: Tuple = (16, 12)):
        """
        Plot planted network and recovered networks side-by-side.
        
        Args:
            W_true (20, 20): Planted network
            W_est_dict: Dictionary of {method_name: (20, 20) matrix}
            output_path: Save path (if None, display only)
            figsize: Figure size
        """
        num_methods = len(W_est_dict) + 1
        fig, axes = plt.subplots(2, num_methods // 2 + num_methods % 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot planted network
        im0 = axes[0].imshow(W_true, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        axes[0].set_title('Planted Network\n(Ground Truth)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Target PRT')
        axes[0].set_ylabel('Source PRT')
        plt.colorbar(im0, ax=axes[0])
        
        # Plot recovered networks
        for idx, (method_name, W_est) in enumerate(W_est_dict.items(), 1):
            im = axes[idx].imshow(W_est, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
            axes[idx].set_title(f'Recovered: {method_name}\n', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Target PRT')
            axes[idx].set_ylabel('Source PRT')
            plt.colorbar(im, ax=axes[idx])
        
        # Hide unused subplots
        for idx in range(num_methods + 1, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()


class DegradationCurves:
    """Plot how recovery performance degrades with data constraints."""
    
    @staticmethod
    def plot_f1_by_data_volume(results: List[Dict], output_path: str = None):
        """
        Plot F1 score vs. data volume for all methods.
        
        Args:
            results: List of experiment results (from experiment_runner)
            output_path: Save path
        """
        # Extract results by data volume
        data_volumes = sorted(set(r['config']['time_steps'] for r in results))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['var_dy', 'covar', 'gnn']
        colors = {'var_dy': 'blue', 'covar': 'green', 'gnn': 'red'}
        markers = {'var_dy': 'o', 'covar': 's', 'gnn': '^'}
        
        for method in methods:
            f1_scores = []
            f1_stds = []
            
            for vol in data_volumes:
                # Find result matching this volume
                matching = [r for r in results if r['config']['time_steps'] == vol 
                           and r['config']['noise_level'] == 0.1
                           and r['config']['network_density'] == 2
                           and r['config']['t_df'] == 5.0]
                
                if matching and method in matching[0]:
                    agg = matching[0][method]
                    if agg and 'f1_score' in agg:
                        f1_scores.append(agg['f1_score']['mean'])
                        f1_stds.append(agg['f1_score']['std'])
            
            ax.errorbar(data_volumes[:len(f1_scores)], f1_scores, yerr=f1_stds,
                       label=method.upper(), marker=markers[method], color=colors[method],
                       linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Time Steps (Data Volume)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Network Recovery Performance vs. Data Volume\n(Noise=0.1, Network Density=2)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_f1_by_noise(results: List[Dict], output_path: str = None):
        """Plot F1 score vs. noise level."""
        noise_levels = sorted(set(r['config']['noise_level'] for r in results))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['var_dy', 'covar', 'gnn']
        colors = {'var_dy': 'blue', 'covar': 'green', 'gnn': 'red'}
        markers = {'var_dy': 'o', 'covar': 's', 'gnn': '^'}
        
        for method in methods:
            f1_scores = []
            f1_stds = []
            
            for noise in noise_levels:
                matching = [r for r in results if r['config']['time_steps'] == 250 
                           and r['config']['noise_level'] == noise
                           and r['config']['network_density'] == 2
                           and r['config']['t_df'] == 5.0]
                
                if matching and method in matching[0]:
                    agg = matching[0][method]
                    if agg and 'f1_score' in agg:
                        f1_scores.append(agg['f1_score']['mean'])
                        f1_stds.append(agg['f1_score']['std'])
            
            ax.errorbar(noise_levels[:len(f1_scores)], f1_scores, yerr=f1_stds,
                       label=method.upper(), marker=markers[method], color=colors[method],
                       linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Measurement Noise (Std Dev)', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Network Recovery Performance vs. Noise Level\n(Time Steps=250, Network Density=2)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()
    
    @staticmethod
    def plot_auc_roc_by_data_volume(results: List[Dict], output_path: str = None):
        """Plot AUC-ROC vs. data volume."""
        data_volumes = sorted(set(r['config']['time_steps'] for r in results))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = ['var_dy', 'covar', 'gnn']
        colors = {'var_dy': 'blue', 'covar': 'green', 'gnn': 'red'}
        markers = {'var_dy': 'o', 'covar': 's', 'gnn': '^'}
        
        for method in methods:
            auc_scores = []
            auc_stds = []
            
            for vol in data_volumes:
                matching = [r for r in results if r['config']['time_steps'] == vol 
                           and r['config']['noise_level'] == 0.1
                           and r['config']['network_density'] == 2
                           and r['config']['t_df'] == 5.0]
                
                if matching and method in matching[0]:
                    agg = matching[0][method]
                    if agg and 'auc_roc' in agg:
                        auc_scores.append(agg['auc_roc']['mean'])
                        auc_stds.append(agg['auc_roc']['std'])
            
            ax.errorbar(data_volumes[:len(auc_scores)], auc_scores, yerr=auc_stds,
                       label=method.upper(), marker=markers[method], color=colors[method],
                       linewidth=2, markersize=8, capsize=5)
        
        ax.set_xlabel('Time Steps (Data Volume)', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('ROC Performance vs. Data Volume\n(Noise=0.1, Network Density=2)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()


class MethodComparison:
    """Compare all methods across metrics."""
    
    @staticmethod
    def plot_metric_heatmap(results: List[Dict], metric: str = 'f1_score', 
                           output_path: str = None):
        """
        Plot heatmap of metrics across configurations.
        
        Args:
            results: List of experiment results
            metric: Metric name (f1_score, auc_roc, precision, recall)
            output_path: Save path
        """
        # Build pivot table
        rows = []
        for r in results:
            config = r['config']
            for method in ['var_dy', 'covar', 'gnn']:
                if method in r and r[method] is not None:
                    agg = r[method]
                    if metric in agg:
                        rows.append({
                            'time_steps': config['time_steps'],
                            'noise_level': config['noise_level'],
                            'method': method,
                            metric: agg[metric]['mean']
                        })
        
        df = pd.DataFrame(rows)
        
        # Create heatmap for each noise level
        noise_levels = sorted(df['noise_level'].unique())
        fig, axes = plt.subplots(1, len(noise_levels), figsize=(15, 5))
        
        for idx, noise in enumerate(noise_levels):
            df_subset = df[df['noise_level'] == noise]
            pivot = df_subset.pivot(index='method', columns='time_steps', values=metric)
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                       ax=axes[idx], cbar_kws={'label': metric})
            axes[idx].set_title(f'Noise Level = {noise}')
            axes[idx].set_ylabel('Method' if idx == 0 else '')
            axes[idx].set_xlabel('Time Steps')
        
        plt.suptitle(f'Network Recovery: {metric.upper()} across Configurations', 
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()


class Visualizer:
    """Main interface for all visualizations."""
    
    def __init__(self, config_path: str = "config.yaml", results_path: str = None):
        """
        Args:
            config_path: Path to config.yaml
            results_path: Path to benchmark_results.json
        """
        self.config_path = config_path
        self.results = []
        
        if results_path:
            with open(results_path, 'r') as f:
                self.results = json.load(f)
    
    def plot_all(self, output_dir: str = "results/figures"):
        """Generate all standard visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating degradation curves...")
        DegradationCurves.plot_f1_by_data_volume(
            self.results,
            os.path.join(output_dir, "f1_vs_data_volume.png")
        )
        
        DegradationCurves.plot_f1_by_noise(
            self.results,
            os.path.join(output_dir, "f1_vs_noise.png")
        )
        
        DegradationCurves.plot_auc_roc_by_data_volume(
            self.results,
            os.path.join(output_dir, "auc_roc_vs_data_volume.png")
        )
        
        print("Generating metric heatmaps...")
        MethodComparison.plot_metric_heatmap(
            self.results, metric='f1_score',
            output_path=os.path.join(output_dir, "f1_heatmap.png")
        )
        
        MethodComparison.plot_metric_heatmap(
            self.results, metric='auc_roc',
            output_path=os.path.join(output_dir, "auc_roc_heatmap.png")
        )
        
        print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example: visualize quick benchmark results
    viz = Visualizer(
        config_path="config.yaml",
        results_path="results/quick_benchmark_data_volume.json"
    )
    viz.plot_all()

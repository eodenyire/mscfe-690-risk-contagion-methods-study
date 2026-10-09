#!/usr/bin/env python3
"""
Main Entry Point for Capstone Methods Study
============================================

Usage:
    python main.py [--quick | --full | --visualize | --test]

Options:
    --quick       Run quick benchmark (data volume effect, 5 min)
    --full        Run full study (all 90 configs, 4-6 hours)
    --visualize   Generate plots from saved results
    --test        Test installation (verify imports)
"""

import argparse
import sys
import json
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment_runner import ExperimentRunner, QuickBenchmark
from visualization import Visualizer
from synthetic_generator import DataGenerator
from recovery_methods import VARDieboldYilmaz


def test_installation():
    """Verify all imports and basic functionality."""
    print("Testing installation...")
    
    try:
        from synthetic_generator import DataGenerator, PlantedNetworkGenerator
        from recovery_methods import VARDieboldYilmaz, CoVaR, DebtRank, GNNRecovery
        from evaluation import RecoveryEvaluator
        from experiment_runner import ExperimentRunner
        from visualization import Visualizer
        print("✓ All imports successful")
        
        # Quick sanity check
        gen = DataGenerator("config.yaml")
        exp = gen.generate_experiment(
            time_steps=100,
            noise_level=0.1,
            network_density=2,
            t_df=5.0
        )
        print(f"✓ Generated synthetic data: KRI shape {exp['kri_panel'].shape}")
        print(f"✓ Planted network: shape {exp['W_true'].shape}, "
              f"{(exp['W_true'] > 0).sum()} edges")
        
        # Test recovery
        var_dy = VARDieboldYilmaz()
        W_est = var_dy.fit_and_recover(exp['kri_panel'], threshold=0.1)
        print(f"✓ VAR+DY recovery: shape {W_est.shape}, "
              f"{(W_est > 0).sum()} recovered edges")
        
        # Test evaluation
        from evaluation import RecoveryEvaluator
        evaluator = RecoveryEvaluator(exp['W_true'])
        results = evaluator.evaluate(W_est)
        print(f"✓ Evaluation: F1={results['f1_score']:.3f}, "
              f"AUC-ROC={results['auc_roc']:.3f}")
        
        print("\n✓✓✓ Installation verified!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_benchmark():
    """Run quick benchmark: data volume effect (5 min, 10 reps per config)."""
    print("\n" + "="*70)
    print("QUICK BENCHMARK: Data Volume Effect")
    print("="*70)
    print("Testing how F1 score changes with data volume (T=50,100,150,200,250)")
    print("This will take ~5 minutes with 10 replications per configuration\n")
    
    quick = QuickBenchmark("config.yaml")
    results = quick.benchmark_data_volume(noise_level=0.1, num_reps=10)
    
    # Save results
    output_file = "results/quick_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print("\n" + "-"*70)
    print("Summary: F1 Score vs. Data Volume (noise=0.1, density=2, df=5)")
    print("-"*70)
    print(f"{'Time Steps':>12} | {'VAR+DY':>10} | {'CoVaR':>10} | {'GNN':>10}")
    print("-"*70)
    
    for result in results:
        config = result['config']
        T = config['time_steps']
        
        var_dy_f1 = result['var_dy']['f1_score']['mean'] if result['var_dy'] else 0
        covar_f1 = result['covar']['f1_score']['mean'] if result['covar'] else 0
        gnn_f1 = result['gnn']['f1_score']['mean'] if result['gnn'] else 0
        
        print(f"{T:>12} | {var_dy_f1:>10.3f} | {covar_f1:>10.3f} | {gnn_f1:>10.3f}")
    
    print("-"*70)
    print("✓ Quick benchmark complete!")
    print("\nNext: python main.py --visualize")


def run_full_study():
    """Run full study (all 9,000 experiments, 4-6 hours on GPU)."""
    print("\n" + "="*70)
    print("FULL BENCHMARKING STUDY")
    print("="*70)
    print("Configurations:")
    print("  Data volume:     50, 100, 150, 200, 250 (5 levels)")
    print("  Noise level:     0.1, 0.3, 0.5 (3 levels)")
    print("  Tail dependence: 3, 5, 10 (3 levels)")
    print("  Density:         2, 4 edges/node (2 levels)")
    print("  Replications:    100 per configuration")
    print(f"  Total:           5 × 3 × 3 × 2 × 100 = 9,000 experiments")
    print("\nEstimated time:")
    print("  GPU:             4–6 hours")
    print("  CPU:             12–24 hours")
    print("="*70)
    
    response = input("\nProceed? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Cancelled.")
        return
    
    print("\nStarting full study...\n")
    
    runner = ExperimentRunner("config.yaml")
    results = runner.run_full_study()
    
    print("\nSaving results...")
    runner.save_results("benchmark_results.json")
    runner.save_summary_table("benchmark_summary.csv")
    
    print("\n✓ Full study completed!")
    print("✓ Results: results/benchmark_results.json")
    print("✓ Summary: results/benchmark_summary.csv")
    print("\nNext: python main.py --visualize")


def visualize_results(results_file="results/quick_benchmark.json"):
    """Generate visualizations from saved results."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    if not Path(results_file).exists():
        print(f"✗ Results file not found: {results_file}")
        print("Run one of:")
        print("  python main.py --quick")
        print("  python main.py --full")
        return
    
    print(f"Loading results from {results_file}...")
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} configurations")
        
        viz = Visualizer("config.yaml")
        viz.results = results
        output_dir = "results/figures"
        
        print(f"\nGenerating plots to {output_dir}/...")
        viz.plot_all(output_dir)
        
        print("\n✓ Visualizations complete!")
        print("✓ Output directory: results/figures/")
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Capstone Methods Study: Network Recovery Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --test         # Verify installation (10 sec)
  python main.py --quick        # Quick benchmark (5 min)
  python main.py --full         # Full study (4-6 hours on GPU)
  python main.py --visualize    # Generate plots from saved results

For details, see README.md and QUICKSTART.md
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test installation (verify imports and basic functionality)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark (data volume effect, ~5 min)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full study (all configurations, 4-6 hours on GPU)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations from saved results'
    )
    parser.add_argument(
        '--results',
        type=str,
        default='results/quick_benchmark.json',
        help='Results file for visualization (default: results/quick_benchmark.json)'
    )
    
    args = parser.parse_args()
    
    # If no arguments, print help
    if not any([args.test, args.quick, args.full, args.visualize]):
        parser.print_help()
        return
    
    # Execute requested command
    if args.test:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    if args.quick:
        run_quick_benchmark()
    
    if args.full:
        run_full_study()
    
    if args.visualize:
        visualize_results(args.results)


if __name__ == "__main__":
    main()

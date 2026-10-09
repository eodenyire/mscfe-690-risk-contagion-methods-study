#!/usr/bin/env python3
"""Complete test suite for Capstone project."""

import sys
import warnings
warnings.filterwarnings('ignore')

# Set output encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CAPSTONE TESTS")
    print("="*70)
    
    try:
        # Test 1: Data Generation
        print("\n[TEST 1] Data Generation...")
        from synthetic_generator import DataGenerator
        gen = DataGenerator('config.yaml')
        exp = gen.generate_experiment(time_steps=50, noise_level=0.1)
        assert exp['kri_panel'].shape == (50, 100)
        print("[PASS] Data generation")
        
        # Test 2: VAR+DY Recovery
        print("\n[TEST 2] VAR+DY Recovery...")
        from recovery_methods import VARDieboldYilmaz
        var_dy = VARDieboldYilmaz()
        W_est = var_dy.fit_and_recover(exp['kri_panel'], threshold=0.1)
        assert W_est.shape == (100, 100)
        print(f"[PASS] VAR+DY recovery (recovered {(W_est > 0).sum()} edges)")
        
        # Test 3: Evaluation
        print("\n[TEST 3] Evaluation Framework...")
        import numpy as np
        from evaluation import RecoveryEvaluator
        W_true_prt = exp['W_true']
        W_true_kri = np.zeros((100, 100))
        for i in range(20):
            for j in range(20):
                i_start, i_end = i * 5, (i + 1) * 5
                j_start, j_end = j * 5, (j + 1) * 5
                W_true_kri[i_start:i_end, j_start:j_end] = W_true_prt[i, j]
        
        evaluator = RecoveryEvaluator(W_true_kri)
        results = evaluator.evaluate(W_est)
        print(f"[PASS] Evaluation (F1={results['f1_score']:.3f}, AUC={results['auc_roc']:.3f})")
        
        # Test 4: Experiment Runner
        print("\n[TEST 4] Experiment Runner...")
        from experiment_runner import ExperimentRunner
        runner = ExperimentRunner()
        result = runner.run_single_configuration(
            time_steps=50, noise_level=0.1, network_density=2, 
            t_df=5.0, num_replications=1
        )
        print("[PASS] Experiment runner")
        
        # Test 5: Visualization
        print("\n[TEST 5] Visualization...")
        from visualization import Visualizer
        viz = Visualizer("config.yaml")
        print("[PASS] Visualization module")
        
        print("\n" + "="*70)
        print("SUCCESS: ALL TESTS PASSED")
        print("="*70)
        print("\nNext: python main.py --quick")
        return 0
        
    except Exception as e:
        print(f"\nFAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Quick test of all components."""

import sys
import warnings
warnings.filterwarnings('ignore')

print("[1/5] Importing DataGenerator...")
from synthetic_generator import DataGenerator
print("✓ DataGenerator imported")

print("[2/5] Creating instance...")
gen = DataGenerator("config.yaml")
print("✓ DataGenerator created")

print("[3/5] Generating experiment (T=50)...")
exp = gen.generate_experiment(time_steps=50, noise_level=0.1)
print(f"✓ Generated: X shape {exp['kri_panel'].shape}, W_true shape {exp['W_true'].shape}")

print("[4/5] Testing VAR+DY recovery...")
from recovery_methods import VARDieboldYilmaz
var_dy = VARDieboldYilmaz()
W_est = var_dy.fit_and_recover(exp['kri_panel'], threshold=0.1)
print(f"✓ Recovered: W_est shape {W_est.shape}, edges {(W_est > 0).sum()}")

print("[5/5] Testing evaluation...")
from evaluation import RecoveryEvaluator
evaluator = RecoveryEvaluator(exp['W_true'])
results = evaluator.evaluate(W_est)
print(f"✓ F1={results['f1_score']:.3f}, AUC-ROC={results['auc_roc']:.3f}")

print("\n✓✓✓ Installation verified successfully!")
print("\nNow run: python main.py --quick")

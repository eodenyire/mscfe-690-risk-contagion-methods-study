#!/usr/bin/env python3
"""Debug VAR+DY output shape."""

from synthetic_generator import DataGenerator
from recovery_methods import VARDieboldYilmaz

gen = DataGenerator('config.yaml')
exp = gen.generate_experiment(time_steps=50, noise_level=0.1)
X = exp['kri_panel']
W_true = exp['W_true']

print(f"X shape: {X.shape}")
print(f"W_true shape: {W_true.shape}")

var_dy = VARDieboldYilmaz()
W_est = var_dy.fit_and_recover(X, threshold=0.1)

print(f"W_est shape: {W_est.shape}")
print(f"W_est type: {type(W_est)}")
print(f"W_est[:5, :5]:\n{W_est[:5, :5]}")

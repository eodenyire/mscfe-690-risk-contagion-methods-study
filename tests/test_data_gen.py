#!/usr/bin/env python3
"""Test data generation."""

print('[1] Importing DataGenerator...')
from synthetic_generator import DataGenerator
print('[2] Instantiating...')
gen = DataGenerator('config.yaml')
print('[3] Generating experiment (T=50)...')
exp = gen.generate_experiment(time_steps=50, noise_level=0.1)
print('✓ X shape:', exp['kri_panel'].shape)
print('✓ W_true shape:', exp['W_true'].shape, 'edges:', (exp['W_true'] > 0).sum())
print('\n✓ Data generation works!')

"""
Network Recovery Methods Study - Source Package

Modules:
    synthetic_generator - VAR(1) + t-copula data generation
    recovery_methods - Four network recovery techniques
    evaluation - Performance metrics and evaluation
    experiment_runner - Orchestration and Monte Carlo execution
    visualization - Publication-ready plotting
"""

__version__ = "1.0.0"
__author__ = "Group 17869: Emmanuel Odenyire & Francis Kwami Dzikpe"

from .synthetic_generator import DataGenerator, PlantedNetworkGenerator
from .recovery_methods import VARDieboldYilmaz, CoVaR, DebtRank, GNNRecovery
from .evaluation import RecoveryEvaluator
from .experiment_runner import ExperimentRunner, QuickBenchmark
from .visualization import Visualizer, AdjacencyHeatmaps, DegradationCurves, MethodComparison

__all__ = [
    'DataGenerator',
    'PlantedNetworkGenerator',
    'VARDieboldYilmaz',
    'CoVaR',
    'DebtRank',
    'GNNRecovery',
    'RecoveryEvaluator',
    'ExperimentRunner',
    'QuickBenchmark',
    'Visualizer',
    'AdjacencyHeatmaps',
    'DegradationCurves',
    'MethodComparison',
]

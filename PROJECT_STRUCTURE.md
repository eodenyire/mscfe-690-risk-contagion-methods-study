# Project Structure

Organized directory layout for the Network Recovery Methods Study.

```
mscfe-690-risk-contagion-methods-study/
├── src/                          # Core source code
│   ├── __init__.py              # Package initialization
│   ├── synthetic_generator.py   # VAR(1) + t-copula data generation
│   ├── recovery_methods.py      # VAR+DY, CoVaR, DebtRank, GNN implementations
│   ├── evaluation.py            # Performance metrics (6 metrics)
│   ├── experiment_runner.py     # Monte Carlo orchestration
│   └── visualization.py         # Publication-ready plots
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_suite.py            # Comprehensive test suite
│   ├── test_quick.py            # Quick functionality tests
│   └── test_data_gen.py         # Data generation validation
│
├── data/                         # Data files and references
│   ├── KRI_Definitions.csv      # 100 KRIs mapped to 20 PRTs
│   ├── Contagion_Propagation.csv # Reference contagion data
│   └── [other data files]
│
├── results/                      # Output directory (generated)
│   ├── benchmark_results.json   # Full 9000 experiment results
│   ├── benchmark_summary.csv    # Aggregated results (90 rows)
│   ├── quick_benchmark.json     # Sample quick mode results
│   ├── figures/                 # Visualization outputs
│   │   ├── f1_vs_data_volume.png
│   │   ├── f1_vs_noise.png
│   │   ├── auc_roc_vs_data_volume.png
│   │   ├── f1_heatmap.png
│   │   └── auc_roc_heatmap.png
│   ├── logs/                    # Experiment logs
│   └── checkpoints/             # Model checkpoints (if applicable)
│
├── Documentation Files
│   ├── README.md                # Comprehensive project documentation
│   ├── METHODOLOGY.md           # Technical specification (study design, math, methods)
│   ├── LITERATURE_REVIEW.md     # Literature review + competitor analysis + SWOT
│   ├── QUICKSTART.md            # Usage guide for running experiments
│   ├── INDEX.md                 # File and module index
│   └── PROJECT_STRUCTURE.md     # This file
│
├── Configuration & Scripts
│   ├── main.py                  # CLI entry point (--test, --quick, --full, --visualize)
│   ├── config.yaml              # Experimental parameters and random seeds
│   ├── requirements.txt          # Python dependencies
│   └── debug_var_dy.py          # Debugging script (optional)
│
├── Version Control
│   ├── .git/                    # Git repository
│   ├── .gitignore               # Git ignore rules
│   └── STATUS.md                # Development status (optional)
│
└── Development Files (auto-generated, can be cleaned)
    └── __pycache__/             # Python bytecode cache
```

---

## File Descriptions

### Core Source (`src/`)

1. **synthetic_generator.py**
   - Implements VAR(1) process with Student's t-copula
   - Generates 100 KRI time series from planted 20-node network
   - Supports experimental factors (noise, tail dependence, data volume)

2. **recovery_methods.py**
   - VAR + Diebold-Yilmaz connectedness (baseline)
   - CoVaR (Adrian & Brunnermeier)
   - DebtRank (Battiston et al.)
   - Graph Neural Network (optional, graceful skip if PyTorch unavailable)

3. **evaluation.py**
   - Computes 6 evaluation metrics
   - Precision, Recall, F1 Score
   - AUC-ROC, Edge Weight Correlation, Impulse-Response MSE

4. **experiment_runner.py**
   - Orchestrates 90 × 100 = 9,000 experiments
   - Manages Monte Carlo replications
   - Saves results to JSON and CSV

5. **visualization.py**
   - Degradation curves (F1 vs data volume, F1 vs noise, AUC-ROC vs data volume)
   - Heatmaps (F1 and AUC-ROC across configurations)
   - Publication-ready PNG output

### Tests (`tests/`)

1. **test_suite.py** - Comprehensive validation tests
2. **test_quick.py** - Quick functionality checks
3. **test_data_gen.py** - Data generation validation

### Data (`data/`)

1. **KRI_Definitions.csv** - Maps 100 KRIs to 20 PRTs (5 per PRT)
2. **Contagion_Propagation.csv** - Reference data
3. Other CSV files as needed

### Results (`results/`)

Generated outputs (can be deleted and regenerated):
- **benchmark_results.json** - Full results (all 9,000 experiments)
- **benchmark_summary.csv** - Aggregated summary (90 rows)
- **figures/** - 5 publication-ready PNG visualizations
- **logs/** - Experiment execution logs

### Documentation

- **README.md** - Main project documentation (4500+ words)
- **METHODOLOGY.md** - Technical specification (2000+ words)
- **LITERATURE_REVIEW.md** - Literature review, competitor analysis, SWOT (2000+ words)
- **QUICKSTART.md** - Quick start guide
- **INDEX.md** - File and module index

### Configuration

- **main.py** - Entry point with CLI interface
- **config.yaml** - Experimental parameters (time steps, noise, tail dependence, density, random seed)
- **requirements.txt** - Python dependencies

---

## Key Structure Principles

1. **Separation of Concerns**
   - `src/` contains core algorithms and logic
   - `tests/` contains validation and test code
   - `data/` contains input data files
   - `results/` contains output files

2. **Clean Root**
   - Only `main.py` (entry point), config files, and documentation at root
   - All source code in `src/` subdirectory
   - All tests in `tests/` subdirectory

3. **Reproducibility**
   - `config.yaml` controls all experimental parameters
   - `requirements.txt` specifies all dependencies
   - All random seeds fixed for reproducibility
   - Results saved in machine-readable formats (JSON, CSV)

4. **Modularity**
   - Each module has a single responsibility
   - Modules import from `src/` package
   - Clear interfaces between components

---

## Running the Project

From the project root:

```bash
# Test installation
python main.py --test

# Run quick benchmark (5 minutes)
python main.py --quick

# Run full study (4-6 hours on GPU)
python main.py --full

# Generate visualizations
python main.py --visualize
```

All imports automatically resolve via `sys.path` configuration in `main.py`.

---

## Development Workflow

1. Edit source files in `src/` as needed
2. Add tests in `tests/` for new functionality
3. Run tests: `python tests/test_suite.py`
4. Execute experiments: `python main.py --full`
5. Generate visualizations: `python main.py --visualize`
6. Commit changes: `git add . && git commit -m "message"`
7. Push to GitHub: `git push origin main`

---

## Maintenance

Regular cleanup (optional):
```bash
# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove generated logs
rm -rf results/logs/*
```

These are already in `.gitignore`, so they won't be committed to GitHub.

---

**Last Updated:** October 9, 2026  
**Status:** Ready for production

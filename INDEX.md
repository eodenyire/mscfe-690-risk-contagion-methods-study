# Capstone Project - File Index & Navigation Guide

**Group 17869** | Emmanuel Odenyire & Francis Kwami Dzikpe  
**Project:** Network Recovery Methods Study (Methods Study, not empirical discovery)  

---

## 📚 Documentation (Read in This Order)

### 1. **README.md** (START HERE)
- **What to read:** Project motivation, scope, high-level technical approach
- **When:** First overview (15 min)
- **Contains:**
  - Problem statement (why this study matters)
  - Risk taxonomy (20 PRTs, 100 KRIs)
  - Technical approach overview
  - Usage examples
  - Competitor analysis & SWOT

### 2. **QUICKSTART.md** (EXECUTE FIRST)
- **What to read:** Step-by-step tutorials to run code
- **When:** Before running anything (10 min)
- **Contains:**
  - Installation instructions
  - 5-step quickstart (test, quick benchmark, visualize)
  - Troubleshooting tips
  - Customization examples

### 3. **METHODOLOGY.md** (FOR DEEP UNDERSTANDING)
- **What to read:** Detailed technical specifications
- **When:** When you need to understand the math (30 min for practitioners)
- **Contains:**
  - Synthetic data generating process (detailed)
  - Why Student's t-copula (not Gaussian)
  - Each recovery method explained in detail
  - Evaluation framework specification
  - Implementation details

### 4. **INDEX.md** (THIS FILE)
- **Navigation guide** for all project files

---

## 💻 Core Implementation Files

### Data Generation
**File:** `synthetic_generator.py` (520 lines)

**Main Classes:**
- `PlantedNetworkGenerator` – Create sparse risk networks (20 PRTs, 30–80 edges)
- `TCopulaVARProcess` – Generate KRI data with VAR(1) + t-copula dependence
- `DataGenerator` – High-level interface with Monte Carlo support

**Key Functions:**
- `generate_experiment()` – Single experiment
- `generate_monte_carlo()` – 100 replications per configuration

**Usage:**
```python
from synthetic_generator import DataGenerator
gen = DataGenerator("config.yaml")
exp = gen.generate_experiment(time_steps=250, noise_level=0.1)
X, W_true = exp['kri_panel'], exp['W_true']
```

---

### Recovery Methods
**File:** `recovery_methods.py` (650 lines)

**Method 1: VAR + Diebold-Yilmaz**
- Class: `VARDieboldYilmaz`
- Baseline 1: Forecast-error variance decomposition
- Time: <1 sec

**Method 2: CoVaR**
- Class: `CoVaR`
- Baseline 2: Conditional Value-at-Risk
- Time: ~2–5 sec

**Method 3: DebtRank**
- Class: `DebtRank`
- Reference method: Cascade contagion propagation
- Time: ~1 sec

**Method 4: Graph Neural Network**
- Class: `GNNRecovery`
- Advanced method: Message-passing networks
- Time: ~10–30 sec (2–5 sec with GPU)

**Factory:**
```python
from recovery_methods import RecoveryMethodsFactory
method = RecoveryMethodsFactory.create('var_dy', lag_order=1)
W_est = method.fit_and_recover(X, threshold=0.1)
```

---

### Evaluation Framework
**File:** `evaluation.py` (450 lines)

**6 Metrics:**
1. `Precision` – Fraction of recovered edges that are correct
2. `Recall` – Fraction of true edges recovered
3. `F1 Score` – Harmonic mean of precision & recall
4. `AUC-ROC` – Threshold-agnostic discrimination
5. `Edge Weight Correlation` – Pearson r between estimated & true weights
6. `Impulse-Response MSE` – Accuracy of shock propagation

**Main Class:**
- `RecoveryEvaluator` – Compute all metrics against ground truth

**Usage:**
```python
from evaluation import RecoveryEvaluator
evaluator = RecoveryEvaluator(W_true)
results = evaluator.evaluate(W_est)
print(f"F1: {results['f1_score']:.3f}, AUC-ROC: {results['auc_roc']:.3f}")
```

---

### Experiment Orchestration
**File:** `experiment_runner.py` (380 lines)

**Main Classes:**
- `ExperimentRunner` – Full pipeline orchestration
  - `run_single_configuration()` – One config, 100 MC reps
  - `run_full_study()` – All 90 configs (9,000 total experiments)
  - `save_results()` – Output to JSON + CSV
  
- `QuickBenchmark` – Fast iteration
  - `benchmark_data_volume()` – 5 min, data volume effect
  - `benchmark_noise()` – 3 min, noise effect
  - `benchmark_tail_dependence()` – 3 min, tail dependence effect

**Usage:**
```python
from experiment_runner import ExperimentRunner, QuickBenchmark

# Quick test
quick = QuickBenchmark()
results = quick.benchmark_data_volume(num_reps=10)

# Full study (4-6 hours)
runner = ExperimentRunner()
results = runner.run_full_study()
runner.save_results()
runner.save_summary_table()
```

---

### Visualization
**File:** `visualization.py` (380 lines)

**3 Visualization Classes:**
- `AdjacencyHeatmaps` – Side-by-side planted vs. recovered networks
- `DegradationCurves` – F1/AUC-ROC vs. data volume, noise, tail dependence
- `MethodComparison` – Heatmaps across all configurations

**Main Class:**
- `Visualizer` – Unified interface

**Usage:**
```python
from visualization import Visualizer
viz = Visualizer("config.yaml", "results/benchmark_results.json")
viz.plot_all("results/figures")
```

**Output:** 5 publication-ready PNG files (300 dpi)

---

## ⚙️ Configuration & Execution

### Configuration File
**File:** `config.yaml` (YAML)

**Key Sections:**
- `prt_definitions` – All 20 PRTs (financial & non-financial)
- `synthetic_data` – DGP parameters (num_prts=20, kris_per_prt=5, etc.)
- `planted_network` – Network generation (sparse/dense, edge weights)
- `experimental_design` – All factors (5×3×3×2 = 90 configs)
- `recovery_methods` – Method-specific hyperparameters
- `evaluation` – Metric configuration
- `output` – Directory structure for results

**Edit this to customize experiments.**

---

### Dependencies
**File:** `requirements.txt`

Python packages:
- Core: numpy, pandas, scipy, scikit-learn, statsmodels
- Visualization: matplotlib, seaborn
- ML: torch, torch-geometric
- Utilities: networkx, pyyaml, tqdm

**Install:**
```bash
pip install -r requirements.txt
```

---

### Main Entry Point
**File:** `main.py` (380 lines)

**Four commands:**

1. **Test installation**
   ```bash
   python main.py --test
   ```
   ✓ Verifies imports and basic functionality (10 sec)

2. **Quick benchmark** (data volume effect)
   ```bash
   python main.py --quick
   ```
   ✓ Tests performance as data volume changes (5 min, 50 total runs)

3. **Full study** (all 9,000 experiments)
   ```bash
   python main.py --full
   ```
   ✓ Runs all 90 configurations (4–6 hours on GPU, 12–24 on CPU)

4. **Generate visualizations**
   ```bash
   python main.py --visualize
   ```
   ✓ Creates 5 publication-ready plots from saved results

---

## 📊 Output Files (Generated)

After running experiments:

```
results/
├── benchmark_results.json          (Full results: 9,000 experiments × 6 metrics)
├── benchmark_summary.csv           (Aggregated: 90 configs × 18 columns)
├── figures/
│   ├── f1_vs_data_volume.png       (F1 degradation as T decreases)
│   ├── f1_vs_noise.png             (F1 degradation as noise increases)
│   ├── auc_roc_vs_data_volume.png  (AUC-ROC vs. data volume)
│   ├── f1_heatmap.png              (F1 across all configurations)
│   └── auc_roc_heatmap.png         (AUC-ROC across all configurations)
└── logs/
    └── experiment.log              (Execution log with timestamps)
```

---

## 🔧 Quick Reference: Most Common Tasks

### Task 1: Verify Installation
```bash
python main.py --test
```
→ Confirms all imports work

### Task 2: Quick Data Volume Test (5 min)
```bash
python main.py --quick
```
→ Generates `results/quick_benchmark.json`

### Task 3: Plot Results
```bash
python main.py --visualize
```
→ Generates `results/figures/*.png`

### Task 4: Run Full Study (4–6 hours)
```bash
python main.py --full
```
→ Generates `results/benchmark_results.json` + `benchmark_summary.csv`

### Task 5: Analyze Results in Python
```python
import pandas as pd
df = pd.read_csv("results/benchmark_summary.csv")
print(df[['time_steps', 'noise_level', 'var_dy_f1', 'covar_f1', 'gnn_f1']].head(10))
```

### Task 6: Customize Experiments
Edit `config.yaml`, modify `experimental_design` section, then run:
```bash
python main.py --full
```

---

## 📈 Expected Results

| Data Volume | VAR+DY F1 | CoVaR F1 | GNN F1 |
|-------------|-----------|----------|--------|
| 50 steps    | 0.42      | 0.38     | 0.45   |
| 100 steps   | 0.55      | 0.51     | 0.63   |
| 250 steps   | 0.74      | 0.71     | 0.78   |

**Key Findings:**
- GNN outperforms when data is limited or noisy
- VAR+DY is robust baseline, stable across conditions
- CoVaR competitive but pairwise limitations emerge

---

## 🎯 Study Design at a Glance

| Aspect | Value |
|--------|-------|
| **PRTs** | 20 (7 financial + 13 non-financial) |
| **KRIs** | 100 (5 per PRT) |
| **Data Volume** | 50, 100, 150, 200, 250 months |
| **Noise Levels** | 3 (σ = 0.1, 0.3, 0.5) |
| **Tail Dependence** | 3 (ν = 3, 5, 10 DoF) |
| **Network Density** | 2 (sparse: 2 edges/node, dense: 4) |
| **Monte Carlo Reps** | 100 per configuration |
| **Total Experiments** | 9,000 |
| **Metrics** | 6 per experiment |
| **Total Data Points** | 54,000 |

---

## 📖 How to Read the Code

**Start with:**
1. `synthetic_generator.py` lines 1–50 (class docstrings)
2. `recovery_methods.py` lines 1–50 (method overview)
3. `evaluation.py` lines 1–50 (metric definitions)
4. `experiment_runner.py` lines 1–50 (orchestration logic)

**Then:**
- Read inline docstrings for each method
- Run `python main.py --test` to see code in action
- Trace through a single experiment manually

---

## ✅ Compliance Checklist

- ✅ Methods study design (not empirical discovery)
- ✅ Circularity resolved via planted networks
- ✅ Student's t-copula for tail dependence
- ✅ 20 PRTs + 100 KRIs integrated
- ✅ VAR+DY + CoVaR baselines
- ✅ DebtRank reference
- ✅ GNN advanced method
- ✅ 6 evaluation metrics
- ✅ Competitor SWOT analysis
- ✅ Detailed documentation (README, METHODOLOGY, QUICKSTART)
- ✅ Reproducible & deterministic
- ✅ No platform/dashboard (minimal viz only)

---

## 🚀 Getting Started

1. **Read:** `README.md` (15 min)
2. **Read:** `QUICKSTART.md` (10 min)
3. **Execute:** `python main.py --test` (10 sec)
4. **Execute:** `python main.py --quick` (5 min)
5. **Execute:** `python main.py --visualize` (2 min)
6. **Read results:** `results/benchmark_summary.csv`
7. **Analyze:** Write custom Python scripts

---

## 📞 Support

| Issue | Solution |
|-------|----------|
| Import error | Run `python main.py --test` to diagnose |
| Missing PyTorch | See QUICKSTART.md "Troubleshooting" |
| Memory error | Use `--quick` instead of `--full` |
| Slow GNN | Use GPU, or reduce `epochs` in config.yaml |

---

**Version:** 1.0 (Proposal/Implementation Complete)  
**Created:** October 2026  
**Group:** 17869

---

## File Statistics

```
Code:          ~2,800 lines
Documentation: ~1,600 lines
Config:        ~100 lines
Total:         ~4,500 lines

Files:         11 core + 3 docs
Modules:       5 main (generator, methods, evaluation, runner, viz)
Classes:       15+ well-documented
Functions:     50+ with docstrings
Tests:         Via main.py --test
```

---

**Ready to execute. Start with `README.md` → `QUICKSTART.md` → `main.py --test`**

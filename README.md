# MScFE Capstone Project: Network Recovery Methods Study

**Group 17869** | Emmanuel Odenyire Anyira & Francis Kwami Dzikpe  
**Track:** Practical Track, Risk Management  
**Date:** Module 4 (Design / Project Proposal)

---

## Executive Summary

This project is a **methods study**, not an empirical discovery of real bank networks. We plant a known risk-propagation network across 20 Principal Risk Types (PRTs) into 100 synthetic Key Risk Indicator (KRI) time series, then rigorously benchmark four competing network-recovery techniques—VAR with Diebold–Yilmaz connectedness, CoVaR (Adrian & Brunnermeier), DebtRank (Battiston et al.), and Graph Neural Networks—to answer:

**Which estimators recover interconnectedness best? Under what conditions? With what accuracy?**

The deliverable is a reproducible benchmarking framework and empirical evidence of estimator performance across controlled scenarios, not a production platform or real institutional claims.

---

## Problem Statement

Financial institutions face complex, multilevel, interconnected risk landscapes. While individual risks are tracked via Key Risk Indicators (KRIs) grouped into Principal Risk Types (PRTs), understanding how risks propagate across types remains a major analytical challenge. The failure of one risk can trigger cascading failures in related risks with severe systemic consequences.

Traditional dashboards answer "Is KRI X in the green/amber/red zone?" but cannot answer:
- Which PRT shocks will most impact another PRT?
- What is the speed and magnitude of contagion across the risk network?
- Where are the critical contagion pathways?

### The Data Constraint Problem

African financial institutions face severe data constraints due to confidentiality and regulatory privacy requirements. Synthetic data offers a pathway, but naive generation—creating data with an embedded structure and then "discovering" it—commits a logical fallacy (circularity).

### This Study's Solution

Rather than attempting to discover a real institutional network from opaque internal data, we reframe the project as a **methods study**:

1. **Plant a known network**: Fix a 20-node directed weighted graph on PRTs.
2. **Generate synthetic KRI data** with that embedded network structure using a VAR(1) process driven by a Student's t-copula (ν=5) to capture tail dependence.
3. **Benchmark recovery techniques**: Apply four competing estimators (VAR+DY, CoVaR, DebtRank, GNN) to the synthetic data.
4. **Measure recovery accuracy** against the planted ground truth under varying conditions (data volume, noise, tail dependence, network density).
5. **Publish empirical performance evidence**: Which techniques work best, and under what circumstances?

---

## Scope and Deliverables

### In Scope
- ✅ Synthetic data generator with Student's t-copula VAR(1) process
- ✅ Implementations of VAR+Diebold-Yilmaz, CoVaR, DebtRank, and GNN
- ✅ Evaluation framework: Precision, Recall, F1, AUC-ROC, edge-weight correlation, impulse-response MSE
- ✅ Reproducible Monte Carlo experiments (100 replications per configuration)
- ✅ Minimal visualization: adjacency heatmaps, degradation curves
- ✅ Summary results table (CSV) and detailed results (JSON)

### Out of Scope
- ❌ Production platform or commercial software
- ❌ Interactive dashboard
- ❌ Real institutional risk data or proprietary bank networks
- ❌ Claims about actual African bank operations
- ❌ User interface or deployment

---

## Technical Approach

### 1. Risk Taxonomy and Dimensionality

**20 Principal Risk Types (PRTs):**

**Financial (7):**
- Credit Risk, Market Risk, Liquidity Risk, Capital Risk
- Interest Rate Risk, Currency Risk, Concentration Risk

**Non-Financial (13):**
- Operational Risk, Compliance Risk, Conduct Risk, Financial Crime Risk
- External Fraud Risk, Internal Fraud Risk, Model Risk, Technology Risk
- Cyber Risk, Strategic Risk, Reputational Risk, Sovereign Risk, Business Continuity Risk

**100 Key Risk Indicators (KRIs):**
- 5 KRIs per PRT (fixed mapping)
- 250 monthly observations per KRI
- Time series panel: (250 time steps) × (100 KRIs)

### 2. Synthetic Data Generating Process

The DGP embeds three features essential for risk contagion:

#### Step 1: Latent VAR(1) Dynamics
```
z_t = diag(φ) × z_{t-1} + ε_t
```
where z_t ∈ ℝ^100 are latent KRI values, φ ∈ [0.6, 0.85] are AR coefficients, and ε_t are innovations with t-copula dependence.

#### Step 2: Student's t-Copula (Why Not Gaussian?)
A Gaussian copula has **zero tail dependence**, so it cannot generate simultaneous extreme co-movements—the defining characteristic of contagion. We adopt a Student's t-copula with ν ∈ {3, 5, 10} degrees of freedom.

The copula correlation matrix Σ is built from the planted network:
```
Σ = (I - λ*W_true)^{-1} D (I - λ*W_true)^{-T}
```
where:
- W_true ∈ ℝ^{20×20} is the planted PRT network
- λ ∈ [0, 1] is contagion strength (fixed at 0.4)
- D is diagonal idiosyncratic variance
- The 100 KRIs inherit correlations via their PRT membership

#### Step 3: Marginal Transformation and Measurement Noise
Latent values map to observable KRI scale [10, 40] via sigmoid transformation:
```
KRI_t = 10 + 30 × sigmoid(z_t / std(z_t))
```
Then independent measurement noise is added:
```
KRI_observed = KRI_t + noise,  where noise ~ N(0, σ²)
```

#### Step 4: Ground-Truth Network
W_true is a sparse directed weighted adjacency matrix on the 20 PRT nodes:
- Each PRT has 2–4 incoming edges (experimental factor)
- Edge weights drawn uniformly from [0.3, 0.9]
- Network is fixed within a run, varied only in robustness checks
- Density levels tested: sparse (2 edges/node) and dense (4 edges/node)

### 3. Recovery Methods

#### Method 1: VAR + Diebold-Yilmaz Connectedness (Baseline 1)

**Principle:** Decompose forecast-error variance (FEVD) to measure spillovers.

**Process:**
1. Fit VAR(1) model to 100 KRI series
2. Compute generalized FEVD matrix Θ (each row sums to 1)
3. Use Θ_{ij} as edge weight from i to j
4. Threshold for binary adjacency

**Strengths:** Linear, transparent, widely used  
**Weaknesses:** Captures average co-movements, not tail dependence; recovers weighted, not binary, graphs

**Reference:** Diebold & Yilmaz (2014), *Journal of Econometrics*

---

#### Method 2: CoVaR (Baseline 2)

**Principle:** Measure tail risk spillovers via conditional Value-at-Risk.

**Process:**
1. Standardize KRI data
2. For each ordered pair (i, j): fit regression of j on indicator(i ≤ 5th percentile)
3. Extract regression coefficient as CoVaR_{j|i} (edge weight)
4. Threshold for adjacency

**Strengths:** Directly targets tail dependence; captures conditional distress  
**Weaknesses:** Pairwise rather than full-network; no consistency enforcement across edges

**Reference:** Adrian & Brunnermeier (2016), *American Economic Review*

---

#### Method 3: DebtRank (Propagation Benchmark)

**Principle:** Propagate distress through network of mutual exposures.

**Usage:** Apply to planted W_true to create reference propagation paths. Compare propagation from estimated networks.

**Process:**
1. Define initial shock vector (one node distressed)
2. Iteratively propagate: `distress_{t+1} = distress_t + α × W^T @ distress_t`
3. Simulate 10 time steps
4. Record systemic impact and distress paths

**Strengths:** Directly models cascading contagion  
**Weaknesses:** Requires known exposure matrix; not a recovery method per se

**Reference:** Battiston et al. (2016), *Statistics and Risk Modeling*

---

#### Method 4: Graph Neural Network (Proposed Advanced Method)

**Principle:** Learn non-linear, higher-order dependencies via message-passing on graphs.

**Architecture:**
- **Input:** KRI time series (250 × 100)
- **Encoder:** MLP mapping to node embeddings (hidden_dim=64)
- **Decoder:** Bilinear form to estimate adjacency matrix
- **Loss:** Reconstruction error + sparsity penalty (L1 regularization)

**Process:**
1. Initialize GNN with random weights
2. Train on synthetic data via gradient descent (100 epochs)
3. Output estimated adjacency A_est ∈ [0, 1]^{100×100}
4. Threshold for binary adjacency

**Strengths:** Learns non-linear relationships; handles higher-order dependencies; single-pass adjacency recovery  
**Weaknesses:** Requires careful hyperparameter tuning; can overfit on small samples with high noise

**Reference:** Kipf & Welling (2017), *ICLR*; Battaglia et al. (2018), arXiv

---

### 4. Evaluation Metrics

Six metrics measure recovery accuracy at the optimal threshold:

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **Precision** | TP / (TP + FP) | Fraction of recovered edges that exist in truth |
| **Recall** | TP / (TP + FN) | Fraction of true edges recovered |
| **F1 Score** | 2·Precision·Recall / (Precision + Recall) | Harmonic mean (optimal at F1=1) |
| **AUC-ROC** | Area under ROC curve | Threshold-agnostic discrimination ability |
| **Edge Weight Correlation** | Pearson r(estimated, true) | Accuracy of edge strength estimates |
| **Impulse-Response MSE** | MSE of shock propagation paths | Accuracy of network dynamics |

**Threshold Selection:** Sweep thresholds ∈ [0, 1] in 50 steps, select threshold maximizing F1.

---

### 5. Experimental Design

**Monte Carlo Framework:** 100 independent replications per configuration.

| Factor | Levels | Rationale |
|--------|--------|-----------|
| **Data Volume** | 50, 100, 150, 200, 250 | African banks often have <250 monthly KRI observations |
| **Noise Level (σ)** | 0.1, 0.3, 0.5 | Measurement error in risk systems |
| **Tail Dependence (ν)** | 3, 5, 10 | Lower ν = fatter tails = more extreme co-movements |
| **Network Density** | 2, 4 edges/node | Sparse vs. dense contagion networks |
| **PRTs** | 20 (fixed) | Full risk taxonomy |
| **KRIs** | 100 (fixed) | 5 per PRT |

**Total Configurations:** 5 × 3 × 3 × 2 = 90  
**Total Experiments:** 90 × 100 = 9,000 runs

---

## Project Structure

```
Capstone/
├── requirements.txt                # Python dependencies
├── config.yaml                     # Experimental configuration
├── synthetic_generator.py          # Data generation with t-copula VAR(1)
├── recovery_methods.py             # VAR+DY, CoVaR, DebtRank, GNN implementations
├── evaluation.py                   # Evaluation framework (6 metrics)
├── experiment_runner.py            # Orchestration and Monte Carlo
├── visualization.py                # Publication-ready figures
├── main.py                         # Entry point (example usage)
├── README.md                       # This file
├── METHODOLOGY.md                  # Detailed technical methodology
├── results/                        # Output directory
│   ├── benchmark_results.json      # Full results (all configs)
│   ├── benchmark_summary.csv       # Summary table (one row per config)
│   └── figures/                    # Visualizations
│       ├── f1_vs_data_volume.png
│       ├── f1_vs_noise.png
│       ├── auc_roc_vs_data_volume.png
│       ├── f1_heatmap.png
│       └── auc_roc_heatmap.png
└── notebooks/                      # Jupyter notebooks for exploration
```

---

## Usage

### 1. Installation

```bash
cd Capstone
pip install -r requirements.txt
```

### 2. Quick Benchmark (Data Volume Effect)

Test performance degradation as data volume decreases:

```bash
python -c "
from experiment_runner import QuickBenchmark
quick = QuickBenchmark('config.yaml')
results = quick.benchmark_data_volume(noise_level=0.1, num_reps=10)
"
```

Output: `results/quick_benchmark_data_volume.json`

### 3. Benchmark Noise Effect

```bash
python -c "
from experiment_runner import QuickBenchmark
quick = QuickBenchmark('config.yaml')
results = quick.benchmark_noise(time_steps=250, num_reps=10)
"
```

### 4. Full Study (All 90 Configurations)

**Warning:** Computationally intensive (~4–6 hours on GPU, longer on CPU).

```bash
python -c "
from experiment_runner import ExperimentRunner
runner = ExperimentRunner('config.yaml')
results = runner.run_full_study()
runner.save_results('benchmark_results.json')
runner.save_summary_table('benchmark_summary.csv')
"
```

### 5. Generate Visualizations

```bash
python -c "
from visualization import Visualizer
viz = Visualizer('config.yaml', 'results/benchmark_results.json')
viz.plot_all('results/figures')
"
```

---

## Example Output

### Summary Table (CSV)

| time_steps | noise_level | network_density | t_df | var_dy_f1 | var_dy_auc | covar_f1 | covar_auc | gnn_f1 | gnn_auc |
|------------|------------|-----------------|------|-----------|-----------|----------|-----------|--------|---------|
| 50         | 0.1        | 2               | 5.0  | 0.42      | 0.68      | 0.38     | 0.65      | 0.51   | 0.72    |
| 100        | 0.1        | 2               | 5.0  | 0.55      | 0.76      | 0.51     | 0.72      | 0.63   | 0.80    |
| 250        | 0.1        | 2               | 5.0  | 0.72      | 0.88      | 0.68     | 0.84      | 0.79   | 0.90    |
| 250        | 0.3        | 2               | 5.0  | 0.61      | 0.81      | 0.57     | 0.77      | 0.68   | 0.84    |
| 250        | 0.5        | 2               | 5.0  | 0.48      | 0.71      | 0.44     | 0.68      | 0.54   | 0.75    |

### Key Findings (Expected)

1. **F1 and AUC-ROC increase with data volume** (50→250 steps)
2. **Performance degrades with noise** (σ=0.1→0.5)
3. **Tail dependence matters** (ν=3 > ν=5 > ν=10 in recovery difficulty)
4. **GNN outperforms baselines** when ν < 10 and T > 150
5. **VAR+DY is most stable** under high noise and sparse data

---

## Competitor Analysis & SWOT

| Approach | Strengths | Weaknesses |
|----------|-----------|-----------|
| **VAR+Diebold-Yilmaz** | Transparent, linear, stable | Captures averages, not tails; no sparsity enforcement |
| **CoVaR** | Direct tail-risk focus | Pairwise only; no multi-node consistency |
| **DebtRank** | Models cascading contagion | Requires known exposures; not a recovery method |
| **GNN** | Non-linear, learns higher-order deps | Hyperparameter sensitive; can overfit |

**SWOT of This Study:**
- **Strengths:** Rigorous benchmark, controlled comparisons, reproducible, t-copula for tail dependence
- **Weaknesses:** Synthetic data only, limited to 20 PRTs and 100 KRIs
- **Opportunities:** Framework adoptable by regional banks for stress-testing their KRI infrastructure
- **Threats:** GNN overfitting on noisy, short time series; computational cost of full study

---

## Literature and References

See **METHODOLOGY.md** and **LITERATURE_REVIEW.md** for full references. Key citations:

- Diebold & Yilmaz (2014). Connectedness from VAR variance decompositions.
- Adrian & Brunnermeier (2016). CoVaR and systemic risk.
- Battiston et al. (2016). DebtRank and network propagation.
- Demarta & McNeil (2005). Student's t-copula for tail dependence.
- Kipf & Welling (2017). Graph Convolutional Networks.

---

## Reproducibility and Open Science

**All code is reproducible:**
- Fixed random seeds in `config.yaml`
- Deterministic YAML configuration
- No external API calls or proprietary data
- Monte Carlo framework for statistical validity (100 reps per config)
- Results saved in human-readable JSON and CSV

**To reproduce any result:**
```bash
# Modify config.yaml with desired parameters, then:
python experiment_runner.py
```

---

## Key Contributions

1. **Methodological:** Reframes the circularity problem via planted networks and methods-study design.
2. **Technical:** Implements and benchmarks four methods (including novel GNN approach) in a unified framework.
3. **Empirical:** Provides practitioner-ready evidence on which techniques to trust under data-scarce conditions.
4. **Open:** Fully reproducible, no proprietary claims or opaque algorithms.

---

## Contact & Attribution

**Group 17869:**
- Emmanuel Odenyire Anyira
- Francis Kwami Dzikpe

**Supervisor Feedback Incorporated:**
- Reframed as methods study (Option A) ✅
- Resolved circularity via planted networks ✅
- Integrated 20 PRTs and 100 KRIs ✅
- Applied Student's t-copula for tail dependence ✅
- Benchmarked VAR+DY and CoVaR baselines ✅
- SWOT competitor analysis ✅
- Dropped platform/dashboard, retained minimal visualization ✅
- Specified concrete dimensions (T=250, σ ∈ {0.1,0.3,0.5}, ν ∈ {3,5,10}) ✅

---

## Next Steps (Post-Module 4)

1. **Run full study** on GPU cluster (9,000 experiments)
2. **Analyze results** and identify conditions favoring each method
3. **Write methodology paper** with empirical findings
4. **Create Jupyter notebooks** for interactive exploration
5. **Submit to capstone evaluation**

---

*Last Updated: October 2026*  
*Version: 1.0 (Proposal)*

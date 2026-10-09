# Detailed Methodology & Technical Specification

## Table of Contents
1. [Study Design](#study-design)
2. [Synthetic Data Generation](#synthetic-data-generation)
3. [Recovery Methods](#recovery-methods)
4. [Evaluation Framework](#evaluation-framework)
5. [Experimental Workflow](#experimental-workflow)
6. [Implementation Details](#implementation-details)

---

## Study Design

### 1.1 Scope Declaration

**This is a methods study, not an empirical discovery.**

We establish a controlled environment where:
1. A known 20-node risk network is planted (ground truth)
2. 100 KRI time series are generated from that network structure using VAR(1) + t-copula
3. Four recovery methods are applied to reconstruct the network
4. Performance is measured against the known ground truth
5. We vary data difficulty (volume, noise, tail dependence) to identify conditions favoring each method

**Research Questions:**
- Q1: Which recovery method best reconstructs planted networks?
- Q2: How does performance degrade with limited data volume?
- Q3: How does noise affect recovery?
- Q4: Does explicit tail-dependence modeling (t-copula) improve GNN recovery?
- Q5: Under what conditions is GNN worth the computational cost vs. simpler baselines?

### 1.2 Rationale for Reframing

**The Circularity Problem:**
If you generate synthetic data with an embedded structure and then use correlation/Granger causality/ML to "discover" it, you only recover what you hardcoded. This proves nothing about estimator validity.

**Our Solution:**
- **Plant a sparse network** (30–80 edges among 20×20=400 possible)
- **Generate 100 KRI series** with that embedded structure
- **Apply methods** that do not know the planting structure
- **Compare output** to ground truth
- **Measure recovery error** under varying noise, data volume, and tail dependence

This breaks the circle: the methods must **infer** the network from noisy observations; they cannot simply recover what was hardcoded.

---

## Synthetic Data Generation

### 2.1 Risk Taxonomy

**20 Principal Risk Types (PRTs):**

| Type | Category | Examples |
|------|----------|----------|
| Credit Risk | Financial | Default probability, LGD migration |
| Market Risk | Financial | VaR, FX volatility |
| Liquidity Risk | Financial | Liquidity coverage ratio (LCR) |
| Capital Risk | Financial | Risk-weighted assets (RWA) |
| Interest Rate Risk | Financial | Key rate durations |
| Currency Risk | Financial | FX exposure concentration |
| Concentration Risk | Financial | Top 10 borrowers % of portfolio |
| Operational Risk | Non-Financial | Loss events per month |
| Compliance Risk | Non-Financial | Regulatory violations |
| Conduct Risk | Non-Financial | Customer complaints |
| Financial Crime Risk | Non-Financial | AML/CFT alerts |
| External Fraud Risk | Non-Financial | Card fraud, theft |
| Internal Fraud Risk | Non-Financial | Rogue trader incidents |
| Model Risk | Non-Financial | Backtesting failures |
| Technology Risk | Non-Financial | System downtime hours |
| Cyber Risk | Non-Financial | Security breaches |
| Strategic Risk | Non-Financial | Market share loss |
| Reputational Risk | Non-Financial | Media mentions (negative) |
| Sovereign Risk | Non-Financial | Country risk exposure |
| Business Continuity Risk | Non-Financial | Recovery time objective (RTO) |

**100 Key Risk Indicators:**
- 5 KRIs per PRT (fixed mapping, known to practitioners)
- Examples:
  - Credit Risk: KRI_001–KRI_005 (PD, LGD, EAD, migration rate, stress PD)
  - Operational Risk: KRI_151–KRI_155 (loss count, severity, tail loss, recovery time)

### 2.2 The Data Generating Process (DGP)

#### Step 1: Planted Network on PRTs

Generate a sparse directed weighted adjacency matrix W_true ∈ ℝ^{20×20}:

```
Algorithm: PlantedNetworkGenerator
Input:  num_prts=20, edges_per_node=2 or 4, seed
Output: W_true (20×20)

For target_node = 1 to 20:
    num_incoming = RandomUniform(1, edges_per_node)
    source_nodes = RandomChoice(20, num_incoming, no_replacement)
    For each source in source_nodes:
        if source ≠ target:
            weight ~ Uniform(0.3, 0.9)
            W_true[source, target] = weight
Return W_true
```

**Properties:**
- Sparse: ~40–80 edges (density 0.1–0.2)
- Weighted: captures heterogeneous interaction strengths
- Directed: distinguishes source (shock originator) from target (receiver)
- Fixed within experiment: ensures fair comparison across methods

#### Step 2: t-Copula Correlation Structure

Build the joint dependence structure from the planted network.

**Why Student's t-Copula?**
- **Gaussian copula:** Zero tail dependence → Cannot capture simultaneous extreme events (contagion definition)
- **Student's t-copula:** Positive tail dependence → Captures simultaneous extremes across KRIs
- **Empirical evidence:** Embrechts et al. (2002), Demarta & McNeil (2005)

**Construction:**

The t-copula correlation matrix is derived from W_true via the inverse relationship:

```
Σ = (I - λ·W_true)^{-1} D (I - λ·W_true)^{-T}
```

where:
- λ = 0.4 (contagion strength: fixed experimental parameter)
- D = diag of idiosyncratic variances (0.5 each)
- Ensures strong correlation between connected PRTs, weaker correlation otherwise

**Why this formula?**
- Reflects **multiplier effect** in economics: shocks propagate as (I - λW)^{-1}·impulse
- Ensures Σ is positive definite (symmetric, eigenvalues > 0)
- Scales with network structure: denser networks → higher correlations

**Expansion to 100 KRIs:**
The 100-dimensional t-copula correlation is constructed by expanding the 20×20 PRT correlation:

```
Σ_100[i, j] = Σ_20[PRT(i), PRT(j)]  ∀i,j ∈ {1,...,100}
```

where PRT(i) is the PRT to which KRI i belongs.

**Sampling from t-copula:**

```
Algorithm: SampleTCopula(Σ, ν, T)
Input:  Σ (100×100) correlation, ν degrees of freedom, T samples
Output: Z (T×100) samples from t-copula

L = Cholesky(Σ)                          # Cholesky decomposition
For t = 1 to T:
    Z_normal_t ~ N(0, I)                # Standard normal
    χ²_t ~ ChiSquared(ν)                # Chi-squared
    scaling_t = sqrt(ν / χ²_t)
    Z[t, :] = L @ (Z_normal_t × scaling_t)  # Apply scaling and Cholesky
Return Z
```

This generates innovations with the desired correlation structure **and** tail dependence.

#### Step 3: VAR(1) Latent Dynamics

Generate latent KRI values with temporal persistence:

```
z_t = diag(φ) × z_{t-1} + ε_t

where:
  z_t ∈ ℝ^100          latent KRI values at time t
  φ_i ~ Uniform(0.6, 0.85)   AR coefficient for KRI i
  ε_t                  t-copula innovations from Step 2
```

**Initialization:** z_0 = ε_0

**Simulation:** For t = 1, 2, ..., T_max:
```
z_t = diag(φ) × z_{t-1} + ε_t
```

**Why VAR(1)?**
- Simple, interpretable, standard in risk modeling
- Captures temporal autocorrelation (typical for KRIs: ρ ≈ 0.75)
- Sufficient for impulse-response analysis

**AR Coefficients:**
- Drawn uniformly from [0.6, 0.85] (realistic persistence for monthly risk metrics)
- Different φ_i for each KRI (heterogeneous dynamics)
- Fixed once drawn (within an experiment)

#### Step 4: Marginal Transformation and Measurement Noise

Map latent values to observable KRI scales:

```
KRI_t = 10 + 30 × sigmoid(z_t / std(z_t))

where sigmoid(x) = 1 / (1 + exp(-x))
```

This gives KRI ∈ [10, 40], a realistic range for many risk metrics (e.g., capital ratios, stress scores).

**Measurement Noise:**
```
KRI_observed = KRI_t + noise,  where noise ~ N(0, σ²)
```

σ ∈ {0.1, 0.3, 0.5} (experimental factor)

**Total Variance:**
- Signal: ~18 (KRI range 10–40)
- Noise-to-signal ratio: σ² / 18 ∈ {0.001, 0.005, 0.014}

---

### 2.3 Planted Network Examples

**Sparse Network (edges_per_node=2):**
```
Expected edges:    20 nodes × 2 edges/node = 40 edges total
Density:           40 / 400 = 0.10 (10%)
Example paths:     Cyber → Operational → Liquidity → Capital
```

**Dense Network (edges_per_node=4):**
```
Expected edges:    20 nodes × 4 edges/node = 80 edges total
Density:           80 / 400 = 0.20 (20%)
Example paths:     Multiple pathways; higher redundancy
```

---

## Recovery Methods

### 3.1 VAR + Diebold-Yilmaz Connectedness

**Reference:** Diebold & Yilmaz (2014), *Journal of Econometrics*

#### Fitting VAR(1)

```
X_t = c + A_1 × X_{t-1} + u_t

where:
  X_t ∈ ℝ^100        KRI values at time t
  c ∈ ℝ^100          constant vector
  A_1 ∈ ℝ^{100×100}  lag-1 coefficient matrix
  u_t ~ N(0, Σ_u)    residuals
```

**OLS Estimation:**

Build lagged design matrix:
```
X_lagged = [1, X_{t-1}, ..., X_{t-p}]  for all t

OLS: B = (X_lagged^T X_lagged)^{-1} X_lagged^T Y
```

Extract A_1 (rows corresponding to lag-1 terms).

#### Generalized Forecast-Error Variance Decomposition (FEVD)

Define impulse responses:
```
Φ_h = A_1^h  for h = 0, 1, ..., H
```

Cumulative FEVD (horizon H=10):
```
FEV_{ij} = Σ_{h=0}^{H} (Φ_h × Cholesky(Σ_u))_{ij}^2
```

Normalize:
```
Θ_{ij} = FEV_{ij} / Σ_k FEV_{kj}

Note: Θ_{.j} sums to 1 (variance shares)
```

**Interpretation:** Θ_{ij} = fraction of variance in i explained by shocks to j

#### Adjacency Matrix from FEVD

Use FEVD directly as edge weights:
```
W_est = Θ

Threshold for binary: A_est = (W_est > threshold) ? 1 : 0
```

**Strengths:**
- Transparent, linear, interpretable
- Widely used in central banking
- Computationally fast (O(n³) for 100 variables)

**Weaknesses:**
- Captures average relationships, not tail dependence
- Recovered weights often continuous [0,1], hard to threshold
- Does not account for non-linear feedbacks

---

### 3.2 CoVaR (Adrian & Brunnermeier, 2016)

**Reference:** Adrian & Brunnermeier (2016), *American Economic Review*

#### Concept

CoVaR_{j|i} = Value-at-Risk of j conditional on i being at its distress level (5th percentile).

Measures: "How much does j lose when i is in trouble?"

#### Implementation

```
Algorithm: CoVaR Recovery
Input:  X (T×100), quantile=0.05
Output: W_est (100×100)

For each target j = 1 to 100:
    For each source i = 1 to 100:
        if i ≠ j:
            X_std = (X - mean(X)) / std(X)          # Standardize
            stress_threshold = percentile(X_std[:,i], 5)
            stress_indicator = (X_std[:,i] ≤ threshold) ? 1 : 0
            
            # Regression: X_std[:,j] ~ stress_indicator
            Design = [1, stress_indicator]
            β = (Design^T Design)^{-1} Design^T X_std[:,j]
            
            W_est[i,j] = |β_stress|                 # Use coefficient magnitude
        else:
            W_est[i,j] = 0

# Normalize and threshold
W_est = W_est / max(W_est)
W_est[W_est < threshold] = 0
```

#### Interpretation

- β > 0: j tends to decline when i is stressed (contagion)
- β < 0: j tends to improve when i is stressed (hedging/diversification)
- |β| magnitude: strength of tail relationship

#### Strengths

- **Tail-focused:** Explicitly targets extreme co-movements
- **Simple:** Easy to explain and implement
- **Pairwise:** No strong assumptions about network structure

#### Weaknesses

- **Pairwise only:** Does not enforce consistency (e.g., if A→B and B→C, doesn't constrain A→C)
- **High-dimensional regression:** With 100 variables, many coefficients estimated (combinatorial growth)
- **Sparsity:** CoVaR matrices often dense (hard to threshold); recovered networks may have too many edges

---

### 3.3 DebtRank (Battiston et al., 2016)

**Reference:** Battiston et al. (2016), *Statistics and Risk Modeling*

#### Concept

DebtRank measures systemic importance and contagion propagation via network clearing model (Eisenberg & Noe, 2001).

**Note:** DebtRank is **not a recovery method**. Instead, it is a **propagation benchmark**: apply it to the known W_true to create a reference propagation path, then compare against propagation paths estimated by VAR+DY and GNN.

#### Algorithm

```
Algorithm: PropagateShock(W, initial_shock, steps=10)
Input:  W (20×20) adjacency, initial_shock (20,), steps
Output: distress_path (steps×20)

# Normalize (column-stochastic)
W_normalized = W / colsum(W)

distress = initial_shock
distress_path[0] = distress

For t = 1 to steps:
    incoming_distress = W_normalized^T @ distress
    distress = distress + α × incoming_distress
    distress = clip(distress, 0, 1)              # Bound to [0,1]
    distress_path[t] = distress

Return distress_path
```

#### Interpretation

- At time t=0: Node 0 is initially distressed (shock=1.0, others=0)
- At time t>0: Distress propagates through network
- distress_path[t, i] = accumulated distress at node i by time t
- Total systemic impact = sum of distress across all nodes

#### Use in Benchmarking

**Reference propagation (ground truth):**
```
ref_path = PropagateShock(W_true, shock, steps=10)
```

**Estimated propagation:**
```
est_path_var_dy = PropagateShock(W_est_var_dy, shock, steps=10)
est_path_gnn = PropagateShock(W_est_gnn, shock, steps=10)
```

**Comparison:**
```
IR_MSE = mean((ref_path - est_path)^2)
```

This measures: "How accurately does the estimated network reproduce the true shock dynamics?"

---

### 3.4 Graph Neural Network (GNN)

**Reference:** Kipf & Welling (2017), *ICLR*; Battaglia et al. (2018), arXiv

#### Architecture

**Goal:** Learn to recover the adjacency matrix A from KRI time series X.

**Model:**
```
Input:  X (T×100) KRI panel
Hidden: Embeddings h_i ∈ ℝ^{64} for each KRI i
Output: A_est (100×100) estimated adjacency
```

**Forward Pass:**

1. **Encoder:** Map time series to node embeddings
   ```
   h_i = ReLU(Linear(X[:, i]))  # h_i ∈ ℝ^{64}
   ```

2. **Decoder:** Estimate adjacency via pairwise similarity
   ```
   A_est[i,j] = sigmoid(Bilinear(h_i, h_j))  for i ≠ j
   A_est[i,i] = 0  (no self-loops)
   ```

3. **Output:** A_est ∈ [0,1]^{100×100} (soft adjacency)

#### Training Objective

```
Loss = Reconstruction Error + Sparsity Penalty

Let X_rec = A_est @ X^T  (reconstructed time series)

Loss = ||X - X_rec||_F^2 + λ × ||A_est||_1

where:
  λ = 0.01 (sparsity weight)
  ||·||_1  = sum of absolute values (encourages zero weights)
```

**Optimization:**
- Optimizer: Adam (lr=0.001)
- Epochs: 100
- Batch size: Full-batch (since X is 250×100, not large)

#### Why GNN for Network Recovery?

1. **Non-linear:** GNNs learn non-linear functions h(x) that correlations/regressions cannot
2. **End-to-end:** Single forward pass produces full adjacency matrix (vs. pairwise CoVaR)
3. **Message passing:** Implicitly captures higher-order dependencies through graph structure
4. **Regularizable:** Sparsity penalty encourages sparse adjacency (more realistic)

#### Training Details

```python
# Initialization
model = GNNAdjacencyEstimator(num_kris=100, hidden_dim=64)
optimizer = Adam(lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    A_est = model(X)                    # Forward pass
    X_rec = A_est @ X.T                 # Reconstruct
    loss = MSE(X, X_rec) + 0.01 * L1(A_est)
    loss.backward()
    optimizer.step()

# Output
W_est = A_est.detach().numpy()
W_est[W_est < 0.1] = 0                  # Threshold
```

#### Potential Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| **Overfitting on noise** | L1 regularization (λ=0.01); early stopping |
| **Vanishing gradients** | ReLU activation; careful weight initialization |
| **All-to-all connectivity** | Sparsity penalty encourages zeros |
| **Slow on 100D** | Computationally tractable (no convolutions, just MLPs) |

---

## Evaluation Framework

### 4.1 Metrics

#### 1. Precision & Recall

```
Precision = TP / (TP + FP)   # Of recovered edges, fraction correct
Recall    = TP / (TP + FN)   # Of true edges, fraction recovered

where:
  TP = # edges correctly identified
  FP = # non-edges incorrectly identified as edges
  FN = # true edges missed
```

**Trade-off:**
- High precision: Few false positives (conservative recovery)
- High recall: Few false negatives (comprehensive recovery)
- F1 balances both

#### 2. F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Range: [0, 1], where 1 = perfect recovery
```

#### 3. AUC-ROC

Threshold-agnostic metric: sweep threshold ∈ [0, 1], plot True Positive Rate vs. False Positive Rate.

```
TPR = TP / (TP + FN)    # Same as Recall
FPR = FP / (FP + TN)    # False positive rate

AUC = ∫ TPR(FPR) dFPR

Range: [0, 1], where 1 = perfect discrimination, 0.5 = random
```

**Advantage:** Not sensitive to threshold choice or class imbalance (few edges vs. many non-edges).

#### 4. Edge Weight Correlation

```
r = Pearson correlation between estimated and true weights

r = Cov(W_est_flat, W_true_flat) / (std(W_est_flat) × std(W_true_flat))

Range: [-1, 1], where 1 = perfect linear agreement
```

Measures: "Do estimated edge strengths match true weights?"

#### 5. Impulse-Response MSE

```
Let φ_h = impulse response at horizon h

MSE = mean over all h and shocks: ||φ_est,h - φ_true,h||_2^2
```

Measures: "Doe recovered network reproduce true shock dynamics?"

#### 6. Optimal Threshold Selection

Since edge weights are continuous, we must threshold to compute precision/recall.

```
Algorithm: FindOptimalThreshold
Input:  W_est, W_true, num_thresholds=50
Output: threshold_opt

thresholds = linspace(0, 1, 50)
f1_max = 0
threshold_opt = 0

For each t in thresholds:
    A_est = (W_est > t) ? 1 : 0
    Precision, Recall = compute metrics
    F1 = 2 × Precision × Recall / (Precision + Recall)
    
    if F1 > f1_max:
        f1_max = F1
        threshold_opt = t

Return threshold_opt, f1_max
```

---

### 4.2 Aggregation Across Monte Carlo Runs

For each configuration, run 100 independent experiments:

```
For rep = 1 to 100:
    data ~ GenerateData(T, σ, ν, density)
    For each method in {VAR+DY, CoVaR, GNN}:
        W_est = method.fit_and_recover(data)
        metric = evaluate(W_est, W_true)
        results[method][rep] = metric

# Aggregate
mean_metric = mean(results[method])
std_metric = std(results[method])
p25 = percentile(results[method], 25)
p75 = percentile(results[method], 75)
```

---

## Experimental Workflow

### 5.1 Configuration Space

**Factors:**
| Factor | Levels | Range |
|--------|--------|-------|
| Data Volume (T) | 5 | 50, 100, 150, 200, 250 |
| Noise Level (σ) | 3 | 0.1, 0.3, 0.5 |
| Tail Dependence (ν) | 3 | 3, 5, 10 |
| Network Density (d) | 2 | 2, 4 edges/node |
| Monte Carlo Reps | 100 | fixed |

**Total Configurations:** 5 × 3 × 3 × 2 = 90  
**Total Experiments:** 90 × 100 = 9,000

### 5.2 Full Study Workflow

```
Phase 1: Setup (Week 1)
  - Generate config.yaml
  - Instantiate DataGenerator, RecoveryMethods, Evaluator
  - Verify synthetic data generation works

Phase 2: Pilot (Week 2)
  - Run QuickBenchmark: data volume effect (10 reps per config)
  - Verify methods execute without error
  - Estimate computational time

Phase 3: Full Study (Weeks 3–4)
  - Run ExperimentRunner.run_full_study()
  - Save results to benchmark_results.json
  - Generate summary CSV

Phase 4: Analysis (Week 5)
  - Compute aggregated metrics
  - Generate visualizations
  - Interpret findings
```

### 5.3 Quick Benchmark: Data Volume Effect

For rapid iteration:

```python
from experiment_runner import QuickBenchmark

quick = QuickBenchmark()
results = quick.benchmark_data_volume(
    noise_level=0.1,
    num_reps=10  # Subset for speed
)
```

This runs only:
- Data volumes: 50, 100, 150, 200, 250 (5 levels)
- Noise: 0.1 (fixed)
- Tail dependence: 5 (fixed)
- Density: 2 (fixed)
- Reps: 10 (vs. 100)

**Total runs:** 5 × 1 × 1 × 1 × 10 = 50 (vs. 9,000 full)  
**Estimated time:** 5 minutes (vs. 4–6 hours)

---

## Implementation Details

### 6.1 File Structure

```
Capstone/
├── synthetic_generator.py
│   ├── PlantedNetworkGenerator
│   │   ├── __init__(num_prts, edges_per_node, seed)
│   │   └── generate() → W_true (20×20)
│   ├── TCopulaVARProcess
│   │   ├── __init__(num_prts, kris_per_prt, time_steps, t_df, ...)
│   │   ├── _sample_t_copula(Sigma, num_samples) → Z (T×100)
│   │   └── generate_data(W_true, noise_level) → (kri_panel, z_latent)
│   └── DataGenerator
│       ├── __init__(config_path)
│       ├── generate_experiment(...) → Dict with kri_panel, W_true
│       └── generate_monte_carlo(..., num_replications=100) → List[Dict]
│
├── recovery_methods.py
│   ├── VARDieboldYilmaz
│   │   ├── _fit_var(X) → (A_1, Sigma)
│   │   ├── _variance_decomposition(A_1, Sigma, horizon=10) → Theta
│   │   └── fit_and_recover(X, threshold=0.1) → W_est
│   ├── CoVaR
│   │   └── fit_and_recover(X, threshold=0.1) → W_est
│   ├── DebtRank
│   │   ├── propagate_shock(W, initial_shock, steps=10) → distress_path
│   │   └── compute_systemic_impact(W) → impact
│   ├── GNNRecovery
│   │   ├── __init__(num_kris, hidden_dim, dropout, learning_rate, epochs)
│   │   └── fit_and_recover(X, threshold=0.1) → W_est
│   └── RecoveryMethodsFactory
│       └── create(method_name, **kwargs) → method_object
│
├── evaluation.py
│   ├── EdgeBinarizer
│   │   ├── binarize(W, threshold) → A
│   │   └── remove_diagonal(A) → A
│   ├── ConfusionMetrics
│   │   └── compute_at_threshold(W_est, W_true, threshold) → Dict
│   ├── ROCAnalysis
│   │   └── compute_roc_curve(W_est, W_true, num_thresholds=50) → Dict
│   ├── WeightCorrelation
│   │   └── compute(W_est, W_true) → Dict
│   ├── ImpulseResponseError
│   │   └── compute_ir_mse(W_est, W_true, steps=10) → Dict
│   └── RecoveryEvaluator
│       ├── __init__(W_true)
│       ├── evaluate(W_est, optimal_threshold=None) → Dict (all metrics)
│       ├── evaluate_batch(W_est_list) → List[Dict]
│       └── aggregate_results(results_list) → Dict (mean, std, percentiles)
│
├── experiment_runner.py
│   ├── ExperimentRunner
│   │   ├── __init__(config_path)
│   │   ├── run_single_configuration(...) → Dict
│   │   ├── run_full_study() → List[Dict]
│   │   ├── save_results(filename)
│   │   └── summary_table() → DataFrame
│   └── QuickBenchmark
│       ├── benchmark_data_volume(noise_level, num_reps)
│       ├── benchmark_noise(time_steps, num_reps)
│       └── benchmark_tail_dependence(time_steps, num_reps)
│
├── visualization.py
│   ├── AdjacencyHeatmaps
│   │   └── plot_comparison(W_true, W_est_dict, output_path)
│   ├── DegradationCurves
│   │   ├── plot_f1_by_data_volume(results, output_path)
│   │   ├── plot_f1_by_noise(results, output_path)
│   │   └── plot_auc_roc_by_data_volume(results, output_path)
│   ├── MethodComparison
│   │   └── plot_metric_heatmap(results, metric, output_path)
│   └── Visualizer
│       ├── __init__(config_path, results_path)
│       └── plot_all(output_dir)
│
├── config.yaml                       # Configuration
├── requirements.txt                  # Dependencies
└── results/                          # Output
    ├── benchmark_results.json
    ├── benchmark_summary.csv
    └── figures/
```

### 6.2 Key Classes & Methods

**DataGenerator Example:**
```python
from synthetic_generator import DataGenerator

gen = DataGenerator("config.yaml")
exp = gen.generate_experiment(
    time_steps=250,
    noise_level=0.1,
    network_density=2,
    t_df=5.0
)

X = exp['kri_panel']              # (250, 100)
W_true = exp['W_true']            # (20, 20)
z_latent = exp['z_latent']        # (250, 100)
```

**Recovery Example:**
```python
from recovery_methods import VARDieboldYilmaz

var_dy = VARDieboldYilmaz(lag_order=1)
W_est = var_dy.fit_and_recover(X, threshold=0.1)  # (20, 20)
```

**Evaluation Example:**
```python
from evaluation import RecoveryEvaluator

evaluator = RecoveryEvaluator(W_true)
results = evaluator.evaluate(W_est)

print(f"F1: {results['f1_score']:.3f}")
print(f"AUC-ROC: {results['auc_roc']:.3f}")
print(f"Weight Correlation: {results['edge_weight_correlation']:.3f}")
```

**Experiment Example:**
```python
from experiment_runner import ExperimentRunner

runner = ExperimentRunner("config.yaml")
config_result = runner.run_single_configuration(
    time_steps=250,
    noise_level=0.1,
    network_density=2,
    t_df=5.0,
    num_replications=100
)

# config_result['var_dy']['f1_score'] = {'mean': 0.72, 'std': 0.08, ...}
```

### 6.3 Output Formats

**JSON Results:**
```json
{
  "config": {
    "time_steps": 250,
    "noise_level": 0.1,
    "network_density": 2,
    "t_df": 5.0,
    "config_name": "T250_N0.1_D2_DF5"
  },
  "var_dy": {
    "f1_score": {
      "mean": 0.72,
      "std": 0.08,
      "min": 0.54,
      "max": 0.85,
      "p25": 0.67,
      "p50": 0.73,
      "p75": 0.78,
      "n": 100
    },
    "auc_roc": {...},
    "precision": {...},
    ...
  },
  "covar": {...},
  "gnn": {...}
}
```

**CSV Summary:**
```csv
time_steps,noise_level,network_density,t_df,var_dy_f1,var_dy_auc_roc,covar_f1,covar_auc_roc,gnn_f1,gnn_auc_roc
50,0.1,2,5.0,0.42,0.68,0.38,0.65,0.51,0.72
100,0.1,2,5.0,0.55,0.76,0.51,0.72,0.63,0.80
...
```

---

## Reproducibility Checklist

- [ ] config.yaml specifies all seeds (network, data, train, gnn)
- [ ] Random seeds set at import (numpy, torch)
- [ ] Data generation is deterministic given seed
- [ ] Methods are deterministic (VAR+DY, CoVaR) or seeded (GNN)
- [ ] Results saved as JSON (machine-readable) and CSV (human-readable)
- [ ] Visualization code is standalone (can be re-run from saved results)
- [ ] All dependencies pinned in requirements.txt

---

*Last Updated: October 2026*

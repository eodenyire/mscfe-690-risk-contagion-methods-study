"""
Recovery Methods for Network Reconstruction
=============================================
Four competing techniques to recover the planted risk network from synthetic KRI data:

1. VAR + Diebold-Yilmaz Connectedness
   - Baseline 1: Linear, variance decomposition-based spillover indices
   - Captures average co-movements (not tail dependence)

2. CoVaR (Adrian & Brunnermeier, 2016)
   - Baseline 2: Tail-risk focused, measures conditional VaR
   - Captures which PRT stress affects another at 5th percentile

3. DebtRank (Battiston et al., 2016)
   - Propagation benchmark: Cascading contagion through known exposure matrix
   - Simulates shock propagation rather than recovery

4. Graph Neural Network (GNN)
   - Advanced method: Message-passing architecture
   - Learns nonlinear, higher-order dependencies directly from data
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, linregress
from sklearn.preprocessing import StandardScaler
import networkx as nx
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# METHOD 1: VAR + DIEBOLD-YILMAZ CONNECTEDNESS
# ============================================================================

class VARDieboldYilmaz:
    """
    Connectedness via vector autoregression (VAR) and variance decomposition.
    
    Process:
    1. Fit a VAR(1) model to the KRI panel
    2. Compute generalized forecast-error variance decompositions (FEVD)
    3. Extract directional spillover indices as edge weights
    4. Threshold to produce adjacency matrix
    
    Reference: Diebold & Yilmaz (2014), Journal of Econometrics
    """
    
    def __init__(self, lag_order: int = 1):
        self.lag_order = lag_order
        self.var_coefs = None
        self.residual_cov = None
    
    def _fit_var(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit VAR(lag_order) model: X_t = c + A_1 X_{t-1} + ... + A_p X_{t-p} + u_t
        
        Args:
            X (T, n): Panel data (T=time steps, n=num_kris)
            
        Returns:
            A_1 (n, n): Coefficient matrix for lag 1
            Sigma (n, n): Residual covariance matrix
        """
        T, n = X.shape
        
        # Build lagged design matrix
        X_lagged = np.hstack([np.ones((T - self.lag_order, 1))])
        for lag in range(1, self.lag_order + 1):
            X_lagged = np.hstack([X_lagged, X[self.lag_order - lag:-lag, :]])
        
        # OLS: Y = X_lagged @ B
        Y = X[self.lag_order:, :]
        B = np.linalg.lstsq(X_lagged, Y, rcond=None)[0]
        
        # Extract first lag coefficient (ignore constant and other lags)
        A_1 = B[1:n+1, :].T  # (n, n)
        
        # Residuals and covariance
        residuals = Y - X_lagged @ B
        Sigma = residuals.T @ residuals / (T - self.lag_order)
        
        return A_1, Sigma
    
    def _variance_decomposition(self, A_1: np.ndarray, Sigma: np.ndarray, horizon: int = 10) -> np.ndarray:
        """
        Compute generalized forecast-error variance decomposition (FEVD).
        
        Args:
            A_1 (n, n): VAR coefficient matrix
            Sigma (n, n): Residual covariance
            horizon: Forecast horizon (default 10 steps)
            
        Returns:
            Theta (n, n): Cumulative FEVD matrix
                Theta[i,j] = fraction of variance in i explained by shocks to j
        """
        n = A_1.shape[0]
        
        # Impulse responses: Phi_h = A_1^h for h=0,1,...,horizon
        Phi = np.zeros((n, n, horizon + 1))
        Phi[:, :, 0] = np.eye(n)
        
        for h in range(1, horizon + 1):
            Phi[:, :, h] = A_1 @ Phi[:, :, h - 1]
        
        # Cholesky decomposition of Sigma
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # Regularize if not PD
            L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(n))
        
        # Forecast-error variance decomposition
        FEV = np.zeros((n, n))
        for h in range(horizon + 1):
            FEV += (Phi[:, :, h] @ L) ** 2
        
        # Normalize to sum to 1 (variance shares)
        Theta = FEV / FEV.sum(axis=1, keepdims=True)
        
        return Theta
    
    def fit_and_recover(self, X: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Fit VAR model and recover adjacency matrix from connectedness.
        
        Args:
            X (T, n): Panel data
            threshold: Edge weight threshold for adjacency
            
        Returns:
            W_est (n, n): Estimated adjacency matrix
        """
        A_1, Sigma = self._fit_var(X)
        Theta = self._variance_decomposition(A_1, Sigma, horizon=10)
        
        # Directional spillover: Theta[i,j] is spillover from j to i
        # Use Theta as edge weight; threshold for binary adjacency
        W_est = Theta.copy()
        W_est[W_est < threshold] = 0
        
        return W_est


# ============================================================================
# METHOD 2: COVAR (ADRIAN & BRUNNERMEIER, 2016)
# ============================================================================

class CoVaR:
    """
    Conditional Value-at-Risk: measure tail-risk spillovers.
    
    For each pair (i, j), estimate:
        CoVaR_j|i = VaR of j conditional on i at its 5th percentile stress
    
    Process:
    1. Standardize KRI data
    2. For each (i, j) pair: fit regression of j on indicator(i ≤ 5th pct)
    3. Extract conditional tail coefficient as edge weight
    4. Threshold to adjacency matrix
    
    Reference: Adrian & Brunnermeier (2016), American Economic Review
    """
    
    def __init__(self, quantile: float = 0.05):
        """
        Args:
            quantile: Stress level (5th percentile = 0.05)
        """
        self.quantile = quantile
    
    def fit_and_recover(self, X: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Recover adjacency matrix from pairwise CoVaR estimates.
        
        Args:
            X (T, n): Panel data
            threshold: Edge weight threshold
            
        Returns:
            W_est (n, n): Estimated adjacency matrix
        """
        T, n = X.shape
        
        # Standardize
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Compute pairwise CoVaR
        W_est = np.zeros((n, n))
        
        for j in range(n):  # Target (receiver of shock)
            for i in range(n):  # Source (shock originator)
                if i == j:
                    continue
                
                # Identify stress periods: when i is below its 5th percentile
                stress_threshold = np.percentile(X_std[:, i], self.quantile * 100)
                stress_indicator = (X_std[:, i] <= stress_threshold).astype(float)
                
                # Regression: X_std[:, j] ~ stress_indicator
                if stress_indicator.sum() > 0:
                    # Coefficient of stress indicator
                    X_design = np.column_stack([np.ones(T), stress_indicator])
                    try:
                        coef = np.linalg.lstsq(X_design, X_std[:, j], rcond=None)[0]
                        # Use absolute value of coefficient as edge weight
                        W_est[i, j] = np.abs(coef[1])
                    except:
                        W_est[i, j] = 0
        
        # Normalize to [0, 1]
        if W_est.max() > 0:
            W_est = W_est / W_est.max()
        
        # Threshold
        W_est[W_est < threshold] = 0
        
        return W_est


# ============================================================================
# METHOD 3: DEBTRANK (BATTISTON ET AL., 2016)
# ============================================================================

class DebtRank:
    """
    Network propagation measure for systemic risk.
    
    DebtRank computes distress transmitted through a system of mutual exposures.
    Given a known exposure matrix, it simulates contagion by propagating a
    shock through the network.
    
    Usage: Apply to the planted W_true to create a benchmark propagation path,
    then compare against propagation paths from estimated networks.
    
    Reference: Battiston et al. (2016), Statistics and Risk Modeling
    """
    
    def __init__(self, contagion_factor: float = 0.5):
        """
        Args:
            contagion_factor: Fraction of distress transmitted per edge
        """
        self.contagion_factor = contagion_factor
    
    def propagate_shock(self, W: np.ndarray, initial_shock: np.ndarray, steps: int = 10) -> np.ndarray:
        """
        Propagate an initial shock through the network over multiple steps.
        
        Args:
            W (n, n): Adjacency matrix (edge weights)
            initial_shock (n,): Initial distress level per node
            steps: Number of propagation steps
            
        Returns:
            distress_path (steps, n): Distress at each node over time
        """
        n = W.shape[0]
        distress = initial_shock.copy()
        distress_path = np.zeros((steps, n))
        distress_path[0] = distress
        
        # Normalize adjacency for each node (outgoing links)
        W_normalized = W.copy()
        for i in range(n):
            out_degree = W[i, :].sum()
            if out_degree > 0:
                W_normalized[i, :] = W[i, :] / out_degree
        
        # Propagate
        for t in range(1, steps):
            # Distress propagates: distress_{t+1} = distress_t + α * W^T @ distress_t
            incoming_distress = W_normalized.T @ distress
            distress = distress + self.contagion_factor * incoming_distress
            distress = np.clip(distress, 0, 1)  # Bound to [0, 1]
            distress_path[t] = distress
        
        return distress_path
    
    def compute_systemic_impact(self, W: np.ndarray) -> np.ndarray:
        """
        Compute systemic impact for each node: average distress propagated if that node fails.
        
        Args:
            W (n, n): Adjacency matrix
            
        Returns:
            impact (n,): Systemic impact score per node
        """
        n = W.shape[0]
        impact = np.zeros(n)
        
        for i in range(n):
            # Shock at node i
            shock = np.zeros(n)
            shock[i] = 1.0
            
            # Propagate
            distress_path = self.propagate_shock(W, shock, steps=10)
            
            # Total propagated distress (excluding initial node)
            impact[i] = distress_path[-1, :].sum() - 1.0
        
        return impact


# ============================================================================
# METHOD 4: GRAPH NEURAL NETWORK (GNN)
# ============================================================================

if TORCH_AVAILABLE:
    class GNNLayer(nn.Module):
        """Graph Convolutional layer for network recovery."""
        
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
            super().__init__()
            self.conv = GCNConv(in_dim, out_dim)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, edge_index, edge_weight=None):
            x = self.conv(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = self.dropout(x)
            return x


    class GNNAdjacencyEstimator(nn.Module):
        """
        Graph Neural Network for adjacency matrix recovery.
        
        Architecture:
        - Input: KRI time series (T, n)
        - 2 GCN layers with message passing
        - Output: Estimated adjacency matrix (n, n)
        
        Training: Minimize reconstruction error + sparsity penalty
        """
        
        def __init__(self, num_kris: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            
            self.num_kris = num_kris
            self.hidden_dim = hidden_dim
            
            # Encoder: map time series to node embeddings
            self.encoder = nn.Sequential(
                nn.Linear(num_kris, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Decoder: reconstruct adjacency via dot product
            self.decoder = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        def forward(self, X: torch.Tensor) -> torch.Tensor:
            """
            Forward pass: estimate adjacency matrix.
            
            Args:
                X (T, n): KRI panel data
                
            Returns:
                A_est (n, n): Estimated adjacency matrix
            """
            # Get mean embeddings per node (across time)
            embeddings = self.encoder(X.T)  # (n, hidden_dim)
            
            # Reconstruct adjacency: A[i,j] = sigmoid(embed_i · W · embed_j)
            n = embeddings.shape[0]
            A_est = torch.zeros(n, n, device=X.device)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        logit = self.decoder(embeddings[i:i+1], embeddings[j:j+1])
                        A_est[i, j] = torch.sigmoid(logit).squeeze()
            
            return A_est
else:
    # Dummy classes when PyTorch is not available
    class GNNLayer:
        pass
    class GNNAdjacencyEstimator:
        pass


class GNNRecovery:
    """Wrapper for GNN-based network recovery."""
    
    def __init__(self, num_kris: int = 100, hidden_dim: int = 64, dropout: float = 0.2, 
                 learning_rate: float = 0.001, epochs: int = 100):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Cannot use GNN method.")
        
        self.num_kris = num_kris
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = GNNAdjacencyEstimator(num_kris, hidden_dim, dropout=dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def fit_and_recover(self, X: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Train GNN and recover adjacency matrix.
        
        Args:
            X (T, n): Panel data
            threshold: Edge weight threshold
            
        Returns:
            W_est (n, n): Estimated adjacency matrix
        """
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            A_est = self.model(X_tensor)
            
            # Loss: reconstruction error + sparsity
            # For now, use simple L2 loss on reconstructed data
            X_reconstructed = A_est @ X_tensor.T
            recon_loss = ((X_tensor.T - X_reconstructed) ** 2).mean()
            sparsity_loss = A_est.abs().sum() * 0.01  # L1 regularization
            
            loss = recon_loss + sparsity_loss
            loss.backward()
            self.optimizer.step()
        
        # Extract adjacency matrix
        self.model.eval()
        with torch.no_grad():
            W_est = self.model(X_tensor).cpu().numpy()
        
        # Threshold
        W_est[W_est < threshold] = 0
        W_est = np.clip(W_est, 0, 1)
        
        return W_est


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

class RecoveryMethodsFactory:
    """Factory for instantiating recovery methods."""
    
    @staticmethod
    def create(method_name: str, **kwargs) -> object:
        """
        Args:
            method_name: 'var_dy', 'covar', 'debtrank', or 'gnn'
            
        Returns:
            Fitted recovery method object
        """
        if method_name.lower() == 'var_dy':
            return VARDieboldYilmaz(**kwargs)
        elif method_name.lower() == 'covar':
            return CoVaR(**kwargs)
        elif method_name.lower() == 'debtrank':
            return DebtRank(**kwargs)
        elif method_name.lower() == 'gnn':
            return GNNRecovery(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method_name}")


if __name__ == "__main__":
    # Test methods
    from synthetic_generator import DataGenerator
    
    gen = DataGenerator("config.yaml")
    exp = gen.generate_experiment(time_steps=250, noise_level=0.1)
    X = exp['kri_panel']
    W_true = exp['W_true']
    
    print("Testing recovery methods...")
    
    # VAR+DY
    print("\n1. VAR + Diebold-Yilmaz")
    var_dy = VARDieboldYilmaz()
    W_var_dy = var_dy.fit_and_recover(X)
    print(f"   Recovered edges: {np.count_nonzero(W_var_dy)}")
    
    # CoVaR
    print("2. CoVaR")
    covar = CoVaR()
    W_covar = covar.fit_and_recover(X)
    print(f"   Recovered edges: {np.count_nonzero(W_covar)}")
    
    # DebtRank
    print("3. DebtRank")
    dr = DebtRank()
    shock = np.zeros(20)
    shock[0] = 1.0
    distress_path = dr.propagate_shock(W_true, shock, steps=10)
    print(f"   Propagation steps: {distress_path.shape[0]}")
    
    # GNN
    if TORCH_AVAILABLE:
        print("4. GNN")
        gnn = GNNRecovery(num_kris=100, epochs=50)
        W_gnn = gnn.fit_and_recover(X)
        print(f"   Recovered edges: {np.count_nonzero(W_gnn)}")
    else:
        print("4. GNN - PyTorch not available, skipping")

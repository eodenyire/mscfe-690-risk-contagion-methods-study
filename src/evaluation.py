"""
Evaluation Framework for Network Recovery
==========================================
Computes 6 key metrics comparing estimated networks to planted ground truth.

Metrics:
1. Precision: TP / (TP + FP) - fraction of recovered edges that are correct
2. Recall: TP / (TP + FN) - fraction of true edges that are recovered
3. F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
4. AUC-ROC: Area under ROC curve for edge presence (threshold sweep)
5. Edge Weight Correlation: Pearson r between estimated and true weights
6. Impulse-Response MSE: MSE between shock propagation paths

The framework thresholds edge weights and sweeps the threshold to trace ROC curves.
"""

import numpy as np
from scipy.stats import linregress, norm
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class EdgeBinarizer:
    """Convert weighted adjacency matrices to binary edge sets via threshold."""
    
    @staticmethod
    def binarize(W: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Args:
            W (n, n): Weighted adjacency matrix
            threshold: Edge weight threshold
            
        Returns:
            A (n, n): Binary adjacency (0 or 1)
        """
        return (W >= threshold).astype(int)
    
    @staticmethod
    def remove_diagonal(A: np.ndarray) -> np.ndarray:
        """Remove self-loops."""
        np.fill_diagonal(A, 0)
        return A


class ConfusionMetrics:
    """Compute precision, recall, F1 at various thresholds."""
    
    @staticmethod
    def compute_at_threshold(W_est: np.ndarray, W_true: np.ndarray, threshold: float = 0.1) -> Dict[str, float]:
        """
        Compute metrics at a single threshold.
        
        Args:
            W_est (n, n): Estimated adjacency matrix (weighted)
            W_true (n, n): Ground-truth adjacency matrix (weighted or binary)
            threshold: Edge weight threshold for binarization
            
        Returns:
            Dictionary with: precision, recall, f1, tp, fp, fn, tn
        """
        # Binarize both matrices
        A_est = EdgeBinarizer.binarize(W_est, threshold)
        A_true = EdgeBinarizer.remove_diagonal(W_true > 0).astype(int)
        
        # Flatten to 1D
        a_est_flat = A_est.flatten()
        a_true_flat = A_true.flatten()
        
        # Confusion matrix
        tp = np.sum((a_est_flat == 1) & (a_true_flat == 1))
        fp = np.sum((a_est_flat == 1) & (a_true_flat == 0))
        fn = np.sum((a_est_flat == 0) & (a_true_flat == 1))
        tn = np.sum((a_est_flat == 0) & (a_true_flat == 0))
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }


class ROCAnalysis:
    """Compute ROC curve and AUC-ROC by sweeping threshold."""
    
    @staticmethod
    def compute_roc_curve(W_est: np.ndarray, W_true: np.ndarray, num_thresholds: int = 50) -> Dict[str, Any]:
        """
        Sweep thresholds and compute ROC curve.
        
        Args:
            W_est (n, n): Estimated adjacency (weighted)
            W_true (n, n): Ground-truth adjacency (binary or weighted)
            num_thresholds: Number of threshold points to evaluate
            
        Returns:
            Dictionary with: thresholds, tpr, fpr, auc_roc, best_threshold, best_f1
        """
        # Flatten weights for threshold sweep
        w_est_flat = W_est.flatten()
        w_true_flat = (W_true > 0).astype(int).flatten()
        
        # Remove NaN and inf
        valid_idx = np.isfinite(w_est_flat)
        w_est_flat = w_est_flat[valid_idx]
        w_true_flat = w_true_flat[valid_idx]
        
        # Threshold sweep
        thresholds = np.linspace(0, 1, num_thresholds)
        tpr_list = []
        fpr_list = []
        f1_list = []
        best_threshold = 0
        best_f1 = 0
        
        for thresh in thresholds:
            a_est_flat = (w_est_flat >= thresh).astype(int)
            
            # TPR and FPR
            tp = np.sum((a_est_flat == 1) & (w_true_flat == 1))
            fp = np.sum((a_est_flat == 1) & (w_true_flat == 0))
            fn = np.sum((a_est_flat == 0) & (w_true_flat == 1))
            tn = np.sum((a_est_flat == 0) & (w_true_flat == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            f1_list.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        tpr_array = np.array(tpr_list)
        fpr_array = np.array(fpr_list)
        
        # Compute AUC
        auc_roc = auc(fpr_array, tpr_array)
        
        return {
            'thresholds': thresholds,
            'tpr': tpr_array,
            'fpr': fpr_array,
            'auc_roc': auc_roc,
            'best_threshold': best_threshold,
            'best_f1': best_f1,
        }


class WeightCorrelation:
    """Correlation between estimated and true edge weights."""
    
    @staticmethod
    def compute(W_est: np.ndarray, W_true: np.ndarray) -> Dict[str, float]:
        """
        Compute Pearson correlation between estimated and true weights.
        
        Args:
            W_est (n, n): Estimated weights
            W_true (n, n): True weights
            
        Returns:
            Dictionary with: correlation, slope, intercept, r_squared, p_value
        """
        # Flatten
        w_est_flat = W_est.flatten()
        w_true_flat = W_true.flatten()
        
        # Remove self-loops and ensure same shape
        mask = np.eye(W_true.shape[0], dtype=bool).flatten()
        w_est_clean = w_est_flat[~mask]
        w_true_clean = w_true_flat[~mask]
        
        # Remove NaN and inf
        valid_idx = np.isfinite(w_est_clean) & np.isfinite(w_true_clean)
        w_est_clean = w_est_clean[valid_idx]
        w_true_clean = w_true_clean[valid_idx]
        
        # Correlation
        if len(w_est_clean) > 2:
            corr = np.corrcoef(w_est_clean, w_true_clean)[0, 1]
            slope, intercept, r_value, p_value, std_err = linregress(w_true_clean, w_est_clean)
            r_squared = r_value ** 2
        else:
            corr = np.nan
            slope = np.nan
            intercept = np.nan
            r_squared = np.nan
            p_value = np.nan
        
        return {
            'correlation': corr,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
        }


class ImpulseResponseError:
    """Measure impulse-response errors (shock propagation)."""
    
    @staticmethod
    def compute_ir_mse(W_est: np.ndarray, W_true: np.ndarray, steps: int = 10) -> Dict[str, float]:
        """
        Compute MSE between impulse responses of estimated and true networks.
        
        Process:
        1. Define canonical impulse shocks (one per node)
        2. Propagate through both networks via matrix powers
        3. Compare resulting response paths
        
        Args:
            W_est (n, n): Estimated adjacency
            W_true (n, n): True adjacency
            steps: Number of propagation steps
            
        Returns:
            Dictionary with: mse_mean, mse_std, mse_by_node
        """
        n = W_true.shape[0]
        
        # Normalize adjacencies (column-stochastic)
        W_est_norm = W_est.copy()
        W_true_norm = W_true.copy()
        
        for i in range(n):
            col_sum_est = W_est_norm[:, i].sum()
            col_sum_true = W_true_norm[:, i].sum()
            
            if col_sum_est > 0:
                W_est_norm[:, i] /= col_sum_est
            if col_sum_true > 0:
                W_true_norm[:, i] /= col_sum_true
        
        # Compute impulse responses
        mse_per_shock = []
        
        for shock_node in range(n):
            # Canonical impulse
            impulse = np.zeros(n)
            impulse[shock_node] = 1.0
            
            # Propagate through true network
            response_true = np.zeros((steps, n))
            response_true[0] = impulse
            for t in range(1, steps):
                response_true[t] = W_true_norm @ response_true[t - 1]
            
            # Propagate through estimated network
            response_est = np.zeros((steps, n))
            response_est[0] = impulse
            for t in range(1, steps):
                response_est[t] = W_est_norm @ response_est[t - 1]
            
            # MSE for this shock
            mse = np.mean((response_true - response_est) ** 2)
            mse_per_shock.append(mse)
        
        return {
            'mse_mean': np.mean(mse_per_shock),
            'mse_std': np.std(mse_per_shock),
            'mse_by_node': np.array(mse_per_shock),
        }


class RecoveryEvaluator:
    """Comprehensive evaluation of network recovery performance."""
    
    def __init__(self, W_true: np.ndarray):
        """
        Args:
            W_true (n, n): Ground-truth planted network
        """
        self.W_true = W_true
        self.n = W_true.shape[0]
    
    def evaluate(self, W_est: np.ndarray, optimal_threshold: float = None) -> Dict[str, Any]:
        """
        Compute all metrics for a single estimated network.
        
        Args:
            W_est (n, n): Estimated network
            optimal_threshold: If None, find via ROC; else use this threshold
            
        Returns:
            Dictionary with all 6 metrics
        """
        # Ensure dimensions match: if W_est is (100, 100) and W_true is (20, 20),
        # expand W_true by block replication
        W_true_eval = self.W_true
        if W_est.shape[0] != W_true_eval.shape[0]:
            # Block replicate W_true
            ratio = W_est.shape[0] // W_true_eval.shape[0]
            W_true_eval = np.kron(W_true_eval, np.ones((ratio, ratio)))
        
        # 1. ROC analysis (includes AUC-ROC and optimal threshold/F1)
        roc = ROCAnalysis.compute_roc_curve(W_est, W_true_eval, num_thresholds=50)
        
        # Use optimal threshold from ROC if not provided
        threshold = optimal_threshold if optimal_threshold is not None else roc['best_threshold']
        
        # 2. Confusion metrics at optimal threshold
        confusion = ConfusionMetrics.compute_at_threshold(W_est, W_true_eval, threshold)
        
        # 3. Weight correlation
        corr = WeightCorrelation.compute(W_est, W_true_eval)
        
        # 4. Impulse-response MSE
        ir_mse = ImpulseResponseError.compute_ir_mse(W_est, W_true_eval, steps=10)
        
        # Compile all results
        results = {
            'precision': confusion['precision'],
            'recall': confusion['recall'],
            'f1_score': confusion['f1'],
            'auc_roc': roc['auc_roc'],
            'edge_weight_correlation': corr['correlation'],
            'impulse_response_mse': ir_mse['mse_mean'],
            'optimal_threshold': threshold,
            'roc_curve': roc,
            'confusion_matrix': confusion,
            'weight_stats': corr,
            'ir_stats': ir_mse,
        }
        
        return results
    
    def evaluate_batch(self, W_est_list: list) -> list:
        """
        Evaluate multiple estimated networks (Monte Carlo runs).
        
        Args:
            W_est_list: List of estimated adjacency matrices
            
        Returns:
            List of result dictionaries
        """
        results = []
        for W_est in W_est_list:
            result = self.evaluate(W_est)
            results.append(result)
        
        return results
    
    @staticmethod
    def aggregate_results(results_list: list) -> Dict[str, Any]:
        """
        Aggregate results across Monte Carlo replications.
        
        Args:
            results_list: List of evaluation results
            
        Returns:
            Dictionary with mean, std, percentiles for each metric
        """
        metrics = ['precision', 'recall', 'f1_score', 'auc_roc', 'edge_weight_correlation', 'impulse_response_mse']
        
        aggregated = {}
        for metric in metrics:
            values = [r[metric] for r in results_list if not np.isnan(r[metric])]
            
            if len(values) > 0:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p25': np.percentile(values, 25),
                    'p50': np.percentile(values, 50),
                    'p75': np.percentile(values, 75),
                    'n': len(values),
                }
            else:
                aggregated[metric] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'p25': np.nan,
                    'p50': np.nan,
                    'p75': np.nan,
                    'n': 0,
                }
        
        return aggregated


if __name__ == "__main__":
    # Test evaluation framework
    from synthetic_generator import DataGenerator
    from recovery_methods import VARDieboldYilmaz
    
    gen = DataGenerator("config.yaml")
    exp = gen.generate_experiment(time_steps=250, noise_level=0.1)
    X = exp['kri_panel']
    W_true = exp['W_true']
    
    # Recover network
    var_dy = VARDieboldYilmaz()
    W_est = var_dy.fit_and_recover(X)
    
    # Evaluate
    evaluator = RecoveryEvaluator(W_true)
    results = evaluator.evaluate(W_est)
    
    print("Evaluation Results:")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1 Score: {results['f1_score']:.3f}")
    print(f"  AUC-ROC: {results['auc_roc']:.3f}")
    print(f"  Edge Weight Correlation: {results['edge_weight_correlation']:.3f}")
    print(f"  Impulse-Response MSE: {results['impulse_response_mse']:.3f}")

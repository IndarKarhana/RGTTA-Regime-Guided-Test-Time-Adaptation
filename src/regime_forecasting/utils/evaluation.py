"""
Evaluation utilities for forecasting performance
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def weighted_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Standard Weighted Mean Absolute Percentage Error (wMAPE).

    Formula:  wMAPE = (Σ |y - ŷ|) / (Σ |y|) × 100

    This is the industry-standard wMAPE used in forecasting competitions
    (M-competitions, Kaggle) and papers. Unlike per-point MAPE, it is
    robust to near-zero actuals because the denominator is an aggregate sum.

    The ``weights`` argument is accepted for API compatibility but is
    ignored — standard wMAPE has no observation-level weighting.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        weights: Ignored (kept for backward compatibility).

    Returns:
        wMAPE as a percentage (0–∞, lower is better).
    """
    abs_errors = np.abs(y_true - y_pred)
    abs_actuals = np.abs(y_true)

    denom = np.sum(abs_actuals)
    if denom < 1e-10:
        # All actuals are essentially zero → undefined; return 0
        return 0.0

    return float(np.sum(abs_errors) / denom * 100.0)


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    Better handles zero values than regular MAPE
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    
    smape = np.mean(numerator / denominator) * 100
    return smape


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct trend predictions)
    """
    if len(y_true) < 2:
        return np.nan
    
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Compare directions (ignore cases where true direction is 0)
    mask = true_direction != 0
    if np.sum(mask) == 0:
        return np.nan
    
    correct_directions = true_direction[mask] == pred_direction[mask]
    return np.mean(correct_directions) * 100


def calculate_all_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        weights: Optional weights for MAPE calculation
        
    Returns:
        Dictionary with all metric values
    """
    metrics = {
        'weighted_mape': weighted_mape(y_true, y_pred, weights),
        'smape': symmetric_mape(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred)
    }
    
    # Remove NaN values
    metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
    
    return metrics


class EvaluationTracker:
    """
    Tracks evaluation metrics across different regimes and time periods
    """
    
    def __init__(self):
        self.regime_metrics = {}
        self.overall_metrics = []
        
    def add_regime_evaluation(
        self, 
        regime_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamp: Optional[str] = None
    ):
        """Add evaluation results for a specific regime"""
        metrics = calculate_all_metrics(y_true, y_pred)
        
        if regime_id not in self.regime_metrics:
            self.regime_metrics[regime_id] = []
            
        evaluation_record = {
            'timestamp': timestamp,
            'metrics': metrics,
            'n_samples': len(y_true)
        }
        
        self.regime_metrics[regime_id].append(evaluation_record)
        self.overall_metrics.append({
            'regime_id': regime_id,
            **evaluation_record
        })
        
        logger.info(f"📊 Evaluation - Regime {regime_id}: MAPE={metrics.get('weighted_mape', 0):.2f}%, RMSE={metrics.get('rmse', 0):.4f}")
    
    def get_regime_summary(self, regime_id: str) -> Optional[Dict]:
        """Get summary statistics for a specific regime"""
        if regime_id not in self.regime_metrics:
            return None
            
        regime_evals = self.regime_metrics[regime_id]
        if not regime_evals:
            return None
        
        # Calculate average metrics across all evaluations for this regime
        all_metrics = [eval_record['metrics'] for eval_record in regime_evals]
        
        summary = {}
        for metric_name in ['weighted_mape', 'smape', 'rmse', 'mae', 'directional_accuracy']:
            values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_latest'] = values[-1]
        
        summary['n_evaluations'] = len(regime_evals)
        summary['total_samples'] = sum(eval_record['n_samples'] for eval_record in regime_evals)
        
        return summary
    
    def get_overall_summary(self) -> Dict:
        """Get overall performance summary across all regimes"""
        if not self.overall_metrics:
            return {}
        
        # Latest metrics
        latest_eval = self.overall_metrics[-1]
        
        # Overall averages
        all_metrics = [eval_record['metrics'] for eval_record in self.overall_metrics]
        overall_avg = {}
        
        for metric_name in ['weighted_mape', 'smape', 'rmse', 'mae', 'directional_accuracy']:
            values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
            if values:
                overall_avg[f'{metric_name}_avg'] = np.mean(values)
        
        return {
            'latest_regime': latest_eval['regime_id'],
            'latest_metrics': latest_eval['metrics'],
            'total_evaluations': len(self.overall_metrics),
            'unique_regimes': len(self.regime_metrics),
            'overall_averages': overall_avg
        }
    
    def get_performance_trend(self, metric_name: str = 'weighted_mape') -> pd.DataFrame:
        """Get performance trend over time for visualization"""
        if not self.overall_metrics:
            return pd.DataFrame()
        
        data = []
        for eval_record in self.overall_metrics:
            if metric_name in eval_record['metrics']:
                data.append({
                    'regime_id': eval_record['regime_id'],
                    'timestamp': eval_record['timestamp'],
                    metric_name: eval_record['metrics'][metric_name],
                    'n_samples': eval_record['n_samples']
                })
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset all tracking data"""
        self.regime_metrics = {}
        self.overall_metrics = []
        logger.info("🔄 Reset evaluation tracker")
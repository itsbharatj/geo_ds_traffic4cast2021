# src/training/metrics.py
"""
Enhanced metrics for Traffic4Cast competition with comprehensive evaluation
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

class Traffic4CastMetrics:
    """
    Comprehensive metrics for Traffic4Cast competition following official evaluation criteria
    
    The Traffic4Cast competition uses several specific metrics:
    1. MSE (Mean Squared Error) - Primary metric
    2. MAE (Mean Absolute Error)
    3. Volume-specific metrics (accuracy, precision, recall)
    4. Speed-specific metrics (only where volume > 0)
    5. Sparsity-aware metrics
    6. Temporal consistency metrics
    """
    
    def __init__(self):
        # Channel indices for traffic data
        self.VOLUME_CHANNELS = [0, 2, 4, 6]  # NE, NW, SE, SW volume
        self.SPEED_CHANNELS = [1, 3, 5, 7]   # NE, NW, SE, SW speed
        
        # Competition-specific thresholds
        self.VOLUME_THRESHOLD = 1.0  # Minimum volume to consider non-zero
        self.SPEED_TOLERANCE = 10.0  # Speed tolerance for accuracy computation
        
    def compute_mse(self, pred: torch.Tensor, target: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> float:
        """Compute Mean Squared Error (primary competition metric)"""
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        mse = F.mse_loss(pred, target, reduction='mean')
        return mse.item()
    
    def compute_mae(self, pred: torch.Tensor, target: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> float:
        """Compute Mean Absolute Error"""
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        mae = F.l1_loss(pred, target, reduction='mean')
        return mae.item()
    
    def compute_rmse(self, pred: torch.Tensor, target: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> float:
        """Compute Root Mean Squared Error"""
        mse = self.compute_mse(pred, target, mask)
        return np.sqrt(mse)
    
    def compute_volume_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute volume-specific metrics
        
        Returns:
            Dict with volume_mse, volume_mae, volume_accuracy, volume_precision, volume_recall
        """
        # Extract volume channels - handle different tensor shapes
        pred_vol, target_vol = self._extract_volume_channels(pred, target)
        
        # Basic volume metrics
        volume_mse = F.mse_loss(pred_vol, target_vol).item()
        volume_mae = F.l1_loss(pred_vol, target_vol).item()
        
        # Binary classification metrics (traffic vs no traffic)
        pred_binary = (pred_vol > self.VOLUME_THRESHOLD).float()
        target_binary = (target_vol > self.VOLUME_THRESHOLD).float()
        
        # Accuracy
        volume_accuracy = torch.mean((pred_binary == target_binary).float()).item()
        
        # Precision and Recall
        tp = torch.sum(pred_binary * target_binary).item()
        fp = torch.sum(pred_binary * (1 - target_binary)).item()
        fn = torch.sum((1 - pred_binary) * target_binary).item()
        
        volume_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        volume_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        volume_f1 = 2 * volume_precision * volume_recall / (volume_precision + volume_recall) if (volume_precision + volume_recall) > 0 else 0.0
        
        return {
            'volume_mse': volume_mse,
            'volume_mae': volume_mae,
            'volume_accuracy': volume_accuracy,
            'volume_precision': volume_precision,
            'volume_recall': volume_recall,
            'volume_f1': volume_f1
        }
    
    def compute_speed_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute speed-specific metrics (only where volume > 0)
        
        Returns:
            Dict with speed_mse, speed_mae, speed_accuracy, speed_correlation
        """
        pred_speed, target_speed, pred_vol, target_vol = self._extract_speed_and_volume_channels(pred, target)
        
        # Create mask where volume > 0 (only consider speed where there's traffic)
        volume_mask = (target_vol > self.VOLUME_THRESHOLD).float()
        
        # Masked speed metrics
        if torch.sum(volume_mask) > 0:
            pred_speed_masked = pred_speed * volume_mask
            target_speed_masked = target_speed * volume_mask
            
            # MSE and MAE only where volume > 0
            speed_mse = F.mse_loss(pred_speed_masked, target_speed_masked, reduction='sum') / torch.sum(volume_mask)
            speed_mae = F.l1_loss(pred_speed_masked, target_speed_masked, reduction='sum') / torch.sum(volume_mask)
            
            # Speed accuracy (within tolerance)
            speed_diff = torch.abs(pred_speed_masked - target_speed_masked)
            speed_accuracy = torch.sum((speed_diff <= self.SPEED_TOLERANCE) * volume_mask) / torch.sum(volume_mask)
            
            # Speed correlation where volume > 0
            mask_flat = volume_mask.view(-1) > 0
            if torch.sum(mask_flat) > 1:
                pred_flat = pred_speed_masked.view(-1)[mask_flat]
                target_flat = target_speed_masked.view(-1)[mask_flat]
                speed_correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                if torch.isnan(speed_correlation):
                    speed_correlation = torch.tensor(0.0)
            else:
                speed_correlation = torch.tensor(0.0)
        else:
            speed_mse = torch.tensor(0.0)
            speed_mae = torch.tensor(0.0)
            speed_accuracy = torch.tensor(0.0)
            speed_correlation = torch.tensor(0.0)
        
        return {
            'speed_mse': speed_mse.item(),
            'speed_mae': speed_mae.item(),
            'speed_accuracy': speed_accuracy.item(),
            'speed_correlation': speed_correlation.item()
        }
    
    def compute_sparsity_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute sparsity-aware metrics for traffic data
        
        Returns:
            Dict with sparsity_ratio, zero_accuracy, non_zero_mse
        """
        pred_vol, target_vol = self._extract_volume_channels(pred, target)
        
        # Sparsity ratio in target
        zero_mask = (target_vol <= self.VOLUME_THRESHOLD).float()
        sparsity_ratio = torch.mean(zero_mask).item()
        
        # Accuracy on zero vs non-zero predictions
        pred_zero = (pred_vol <= self.VOLUME_THRESHOLD).float()
        target_zero = zero_mask
        zero_accuracy = torch.mean((pred_zero == target_zero).float()).item()
        
        # MSE only on non-zero locations
        non_zero_mask = 1 - zero_mask
        if torch.sum(non_zero_mask) > 0:
            non_zero_mse = F.mse_loss(pred_vol * non_zero_mask, target_vol * non_zero_mask, reduction='sum') / torch.sum(non_zero_mask)
        else:
            non_zero_mse = torch.tensor(0.0)
        
        return {
            'sparsity_ratio': sparsity_ratio,
            'zero_accuracy': zero_accuracy,
            'non_zero_mse': non_zero_mse.item()
        }
    
    def compute_temporal_consistency_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute temporal consistency metrics across prediction horizon
        
        Returns:
            Dict with temporal_smoothness, prediction_variance
        """
        # Reshape to get temporal dimension
        pred_temporal, target_temporal = self._reshape_for_temporal_analysis(pred, target)
        
        if pred_temporal.shape[1] > 1:  # Need at least 2 timesteps
            # Temporal smoothness (difference between consecutive timesteps)
            pred_diff = torch.diff(pred_temporal, dim=1)
            target_diff = torch.diff(target_temporal, dim=1)
            
            temporal_smoothness = F.mse_loss(pred_diff, target_diff).item()
            
            # Prediction variance across time
            pred_var = torch.var(pred_temporal, dim=1).mean().item()
            target_var = torch.var(target_temporal, dim=1).mean().item()
            variance_error = abs(pred_var - target_var)
        else:
            temporal_smoothness = 0.0
            variance_error = 0.0
        
        return {
            'temporal_smoothness': temporal_smoothness,
            'variance_error': variance_error
        }
    
    def compute_competition_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute all official Traffic4Cast competition metrics
        
        Args:
            pred: Predictions tensor
            target: Ground truth tensor
            mask: Optional mask for road locations
            
        Returns:
            Dictionary with all competition metrics
        """
        # Ensure tensors are on same device and format
        pred = pred.float()
        target = target.float()
        
        metrics = {}
        
        try:
            # Primary competition metrics
            metrics['mse'] = self.compute_mse(pred, target, mask)
            metrics['mae'] = self.compute_mae(pred, target, mask)
            metrics['rmse'] = self.compute_rmse(pred, target, mask)
            
            # Volume-specific metrics
            volume_metrics = self.compute_volume_metrics(pred, target)
            metrics.update(volume_metrics)
            
            # Speed-specific metrics
            speed_metrics = self.compute_speed_metrics(pred, target)
            metrics.update(speed_metrics)
            
            # Sparsity metrics
            sparsity_metrics = self.compute_sparsity_metrics(pred, target)
            metrics.update(sparsity_metrics)
            
            # Temporal consistency metrics
            temporal_metrics = self.compute_temporal_consistency_metrics(pred, target)
            metrics.update(temporal_metrics)
            
            # Competition score (weighted combination)
            competition_score = self._compute_competition_score(metrics)
            metrics['competition_score'] = competition_score
            
        except Exception as e:
            logging.warning(f"Error computing metrics: {e}")
            # Return basic metrics in case of error
            metrics = {
                'mse': self.compute_mse(pred, target, mask),
                'mae': self.compute_mae(pred, target, mask),
                'rmse': self.compute_rmse(pred, target, mask),
                'competition_score': self.compute_mse(pred, target, mask)
            }
        
        return metrics
    
    def _extract_volume_channels(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract volume channels from prediction and target tensors"""
        if pred.dim() == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8  # Assuming 8 channels per timestep
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.dim() == 5:  # (B, T, H, W, C)
            pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.permute(0, 1, 4, 2, 3)
        
        # Extract volume channels
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]  # (B, T, 4, H, W)
        target_vol = target[:, :, self.VOLUME_CHANNELS]
        
        return pred_vol, target_vol
    
    def _extract_speed_and_volume_channels(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract speed and volume channels from tensors"""
        if pred.dim() == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.dim() == 5:  # (B, T, H, W, C)
            pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.permute(0, 1, 4, 2, 3)
        
        # Extract channels
        pred_speed = pred[:, :, self.SPEED_CHANNELS]  # (B, T, 4, H, W)
        target_speed = target[:, :, self.SPEED_CHANNELS]
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]
        target_vol = target[:, :, self.VOLUME_CHANNELS]
        
        return pred_speed, target_speed, pred_vol, target_vol
    
    def _reshape_for_temporal_analysis(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape tensors for temporal analysis"""
        if pred.dim() == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.dim() == 5:  # (B, T, H, W, C)
            pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.permute(0, 1, 4, 2, 3)
        
        # Aggregate spatially for temporal analysis
        pred_temporal = pred.mean(dim=(3, 4))  # (B, T, C)
        target_temporal = target.mean(dim=(3, 4))
        
        return pred_temporal, target_temporal
    
    def _compute_competition_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute overall competition score as weighted combination of metrics
        
        Lower is better (following MSE convention)
        """
        # Weights based on Traffic4Cast competition emphasis
        weights = {
            'mse': 0.4,
            'volume_mse': 0.3,
            'speed_mse': 0.2,
            'temporal_smoothness': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]
                total_weight += weight
        
        if total_weight > 0:
            score /= total_weight
        else:
            score = metrics.get('mse', 0.0)
        
        return score

class MetricsTracker:
    """Track metrics across training epochs"""
    
    def __init__(self):
        self.epoch_metrics = []
        
    def update(self, epoch: int, phase: str, metrics: Dict[str, float]):
        """Update metrics for current epoch and phase"""
        record = {
            'epoch': epoch,
            'phase': phase,
            **metrics
        }
        self.epoch_metrics.append(record)
    
    def get_best_epoch(self, metric: str = 'competition_score', phase: str = 'val') -> Tuple[int, float]:
        """Get epoch with best metric value"""
        phase_metrics = [m for m in self.epoch_metrics if m['phase'] == phase and metric in m]
        if not phase_metrics:
            return 0, float('inf')
        
        best_record = min(phase_metrics, key=lambda x: x[metric])
        return best_record['epoch'], best_record[metric]
    
    def get_metrics_history(self, metric: str, phase: str = 'val') -> List[float]:
        """Get history of specific metric"""
        return [m[metric] for m in self.epoch_metrics 
                if m['phase'] == phase and metric in m]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {'epoch_metrics': self.epoch_metrics}
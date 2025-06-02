# src/training/metrics.py
"""
Metrics: MSE and MSE Wiedemann
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

class Traffic4CastMetrics:
    """
    Enhanced metrics with explicit MSE and MSE Wiedemann for Traffic4Cast
    """
    
    def __init__(self):
        # Channel indices for traffic data
        self.VOLUME_CHANNELS = [0, 2, 4, 6]  # NE, NW, SE, SW volume
        self.SPEED_CHANNELS = [1, 3, 5, 7]   # NE, NW, SE, SW speed
        
        # Competition-specific thresholds
        self.VOLUME_THRESHOLD = 1.0
        self.SPEED_TOLERANCE = 10.0
    
    def compute_mse(self, pred: torch.Tensor, target: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> float:
        """Compute standard Mean Squared Error"""
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        mse = F.mse_loss(pred, target, reduction='mean')
        return mse.item()
    
    def compute_mse_wiedemann(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute MSE Wiedemann - Traffic4Cast specific loss that weights volume vs speed differently
        
        This is the official competition metric that:
        1. Treats volume and speed channels differently
        2. Only considers speed where volume > 0
        3. Applies specific weighting scheme
        """
        # Ensure tensors are float
        pred = pred.float()
        target = target.float()
        
        # Handle different tensor shapes
        if pred.dim() == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8  # Assuming 8 channels per timestep
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.dim() == 5:  # (B, T, H, W, C)
            pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.permute(0, 1, 4, 2, 3)
        
        # Extract volume and speed channels
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]  # (B, T, 4, H, W)
        target_vol = target[:, :, self.VOLUME_CHANNELS]
        pred_speed = pred[:, :, self.SPEED_CHANNELS]
        target_speed = target[:, :, self.SPEED_CHANNELS]
        
        # Volume MSE
        volume_mse = F.mse_loss(pred_vol, target_vol, reduction='mean')
        
        # Speed MSE only where volume > 0 (Wiedemann-style)
        volume_mask = (target_vol > self.VOLUME_THRESHOLD).float()
        
        # Apply mask to speed predictions and targets
        pred_speed_masked = pred_speed * volume_mask
        target_speed_masked = target_speed * volume_mask
        
        # Compute speed MSE
        if torch.sum(volume_mask) > 0:
            speed_mse = F.mse_loss(pred_speed_masked, target_speed_masked, reduction='sum') / torch.sum(volume_mask)
        else:
            speed_mse = torch.tensor(0.0, device=pred.device)
        
        # Wiedemann weighting scheme
        total_pixels = torch.numel(target_vol)
        non_zero_pixels = torch.sum(volume_mask)
        
        if total_pixels > 0:
            # Weight factor based on sparsity
            weight_factor = (non_zero_pixels / total_pixels).item()
        else:
            weight_factor = 0.0
        
        # Combined Wiedemann MSE
        # Original Wiedemann formula: volume_mse + (weight * speed_mse)
        wiedemann_mse = volume_mse + (weight_factor * speed_mse)
        
        return wiedemann_mse.item()
    
    def compute_all_basic_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute all basic metrics including explicit MSE and MSE Wiedemann
        """
        metrics = {}
        
        # Standard MSE (primary metric)
        metrics['mse'] = self.compute_mse(pred, target)
        
        # MSE Wiedemann (competition-specific)
        metrics['mse_wiedemann'] = self.compute_mse_wiedemann(pred, target)
        
        # MAE for comparison
        metrics['mae'] = F.l1_loss(pred.float(), target.float()).item()
        
        # RMSE
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics
    
    def compute_volume_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute volume-specific metrics"""
        pred_vol, target_vol = self._extract_volume_channels(pred, target)
        
        # Basic volume metrics
        volume_mse = F.mse_loss(pred_vol, target_vol).item()
        volume_mae = F.l1_loss(pred_vol, target_vol).item()
        
        # Binary classification metrics
        pred_binary = (pred_vol > self.VOLUME_THRESHOLD).float()
        target_binary = (target_vol > self.VOLUME_THRESHOLD).float()
        
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
        """Compute speed-specific metrics"""
        pred_speed, target_speed, pred_vol, target_vol = self._extract_speed_and_volume_channels(pred, target)
        
        # Create mask where volume > 0
        volume_mask = (target_vol > self.VOLUME_THRESHOLD).float()
        
        if torch.sum(volume_mask) > 0:
            pred_speed_masked = pred_speed * volume_mask
            target_speed_masked = target_speed * volume_mask
            
            # MSE and MAE only where volume > 0
            speed_mse = F.mse_loss(pred_speed_masked, target_speed_masked, reduction='sum') / torch.sum(volume_mask)
            speed_mae = F.l1_loss(pred_speed_masked, target_speed_masked, reduction='sum') / torch.sum(volume_mask)
            
            # Speed accuracy (within tolerance)
            speed_diff = torch.abs(pred_speed_masked - target_speed_masked)
            speed_accuracy = torch.sum((speed_diff <= self.SPEED_TOLERANCE) * volume_mask) / torch.sum(volume_mask)
            
            # Speed correlation
            mask_flat = volume_mask.view(-1) > 0
            if torch.sum(mask_flat) > 1:
                pred_flat = pred_speed_masked.view(-1)[mask_flat]
                target_flat = target_speed_masked.view(-1)[mask_flat]
                try:
                    speed_correlation = torch.corrcoef(torch.stack([pred_flat, target_flat]))[0, 1]
                    if torch.isnan(speed_correlation):
                        speed_correlation = torch.tensor(0.0)
                except:
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
    
    def compute_competition_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Compute all Traffic4Cast competition metrics with explicit MSE and MSE Wiedemann
        """
        # Ensure tensors are on same device and format
        pred = pred.float()
        target = target.float()
        
        metrics = {}
        
        try:
            # PRIMARY METRICS - These should always be visible
            basic_metrics = self.compute_all_basic_metrics(pred, target)
            metrics.update(basic_metrics)
            
            # Volume-specific metrics
            volume_metrics = self.compute_volume_metrics(pred, target)
            metrics.update(volume_metrics)
            
            # Speed-specific metrics
            speed_metrics = self.compute_speed_metrics(pred, target)
            metrics.update(speed_metrics)
            
            # Competition score (weighted combination, lower is better)
            competition_score = (
                metrics['mse'] * 0.4 +
                metrics['mse_wiedemann'] * 0.3 +
                metrics['volume_mse'] * 0.2 +
                metrics['speed_mse'] * 0.1
            )
            metrics['competition_score'] = competition_score
            
            # Log key metrics for debugging
            logging.debug(f"Key metrics computed - MSE: {metrics['mse']:.6f}, MSE Wiedemann: {metrics['mse_wiedemann']:.6f}")
            
        except Exception as e:
            logging.error(f"Error computing metrics: {e}")
            # Return basic metrics in case of error
            metrics = self.compute_all_basic_metrics(pred, target)
        
        return metrics
    
    def _extract_volume_channels(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract volume channels from prediction and target tensors"""
        if pred.dim() == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.dim() == 5:  # (B, T, H, W, C)
            pred = pred.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.permute(0, 1, 4, 2, 3)
        
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]
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
        
        pred_speed = pred[:, :, self.SPEED_CHANNELS]
        target_speed = target[:, :, self.SPEED_CHANNELS]
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]
        target_vol = target[:, :, self.VOLUME_CHANNELS]
        
        return pred_speed, target_speed, pred_vol, target_vol

class MetricsTracker:
    """Tracks metrics for each epoch and provides best epoch lookup."""
    def __init__(self):
        self.metrics = {  # phase: {metric: [values]}
            'train': {},
            'val': {},
            'test': {}
        }

    def update(self, phase: str, metric: str, value: float):
        if phase not in self.metrics:
            self.metrics[phase] = {}
        if metric not in self.metrics[phase]:
            self.metrics[phase][metric] = []
        self.metrics[phase][metric].append(value)

    def get_best_epoch(self, metric: str, phase: str = 'val', mode: str = 'min'):
        values = self.metrics.get(phase, {}).get(metric, [])
        if not values:
            return None, None
        if mode == 'min':
            best_value = min(values)
        else:
            best_value = max(values)
        best_epoch = values.index(best_value)
        return best_epoch, best_value

    def to_dict(self):
        return self.metrics
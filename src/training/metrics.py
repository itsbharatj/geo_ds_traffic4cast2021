# src/training/metrics.py
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

class TrafficMetrics:
    """Metrics computation for traffic prediction"""
    
    def __init__(self):
        # Channel indices for traffic data
        self.VOLUME_CHANNELS = [0, 2, 4, 6]  # NE, NW, SE, SW volume
        self.SPEED_CHANNELS = [1, 3, 5, 7]   # NE, NW, SE, SW speed
        
    def compute_mse(self, pred: np.ndarray, target: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> float:
        """Compute Mean Squared Error"""
        pred_tensor = torch.from_numpy(pred).float()
        target_tensor = torch.from_numpy(target).float()
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
            pred_tensor = pred_tensor * mask_tensor
            target_tensor = target_tensor * mask_tensor
            
        mse = F.mse_loss(pred_tensor, target_tensor).item()
        return mse
    
    def compute_mae(self, pred: np.ndarray, target: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
        """Compute Mean Absolute Error"""
        pred_tensor = torch.from_numpy(pred).float()
        target_tensor = torch.from_numpy(target).float()
        
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).float()
            pred_tensor = pred_tensor * mask_tensor
            target_tensor = target_tensor * mask_tensor
            
        mae = F.l1_loss(pred_tensor, target_tensor).item()
        return mae
    
    def compute_rmse(self, pred: np.ndarray, target: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> float:
        """Compute Root Mean Squared Error"""
        mse = self.compute_mse(pred, target, mask)
        return np.sqrt(mse)
    
    def compute_volume_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute volume-specific metrics"""
        # Extract volume channels
        if pred.ndim == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8  # Assuming 8 channels per timestep
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.ndim == 5:  # (B, T, H, W, C)
            pred = pred.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.transpose(0, 1, 4, 2, 3)
        
        # Get volume data
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]  # (B, T, 4, H, W)
        target_vol = target[:, :, self.VOLUME_CHANNELS]
        
        # Compute metrics
        volume_mse = self.compute_mse(pred_vol, target_vol)
        volume_mae = self.compute_mae(pred_vol, target_vol)
        
        # Volume accuracy (binary classification: traffic vs no traffic)
        pred_binary = (pred_vol > 0).astype(np.float32)
        target_binary = (target_vol > 0).astype(np.float32)
        volume_accuracy = np.mean(pred_binary == target_binary)
        
        return {
            'volume_mse': volume_mse,
            'volume_mae': volume_mae,
            'volume_accuracy': volume_accuracy
        }
    
    def compute_speed_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute speed-specific metrics (only where volume > 0)"""
        # Extract speed channels
        if pred.ndim == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8
            pred = pred.reshape(B, T, 8, H, W)
            target = target.reshape(B, T, 8, H, W)
        elif pred.ndim == 5:  # (B, T, H, W, C)
            pred = pred.transpose(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            target = target.transpose(0, 1, 4, 2, 3)
        
        # Get speed and volume data
        pred_speed = pred[:, :, self.SPEED_CHANNELS]  # (B, T, 4, H, W)
        target_speed = target[:, :, self.SPEED_CHANNELS]
        pred_vol = pred[:, :, self.VOLUME_CHANNELS]
        target_vol = target[:, :, self.VOLUME_CHANNELS]
        
        # Create mask where volume > 0 (only consider speed where there's traffic)
        volume_mask = (target_vol > 0).astype(np.float32)
        
        # Compute metrics only where there's volume
        speed_mse = self.compute_mse(pred_speed, target_speed, volume_mask)
        speed_mae = self.compute_mae(pred_speed, target_speed, volume_mask)
        
        # Speed accuracy (within tolerance)
        speed_tolerance = 10.0  # Speed units
        mask_locations = volume_mask > 0
        if np.sum(mask_locations) > 0:
            speed_diff = np.abs(pred_speed[mask_locations] - target_speed[mask_locations])
            speed_accuracy = np.mean(speed_diff <= speed_tolerance)
        else:
            speed_accuracy = 0.0
        
        return {
            'speed_mse': speed_mse,
            'speed_mae': speed_mae, 
            'speed_accuracy': speed_accuracy
        }
    
    def compute_traffic_specific_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Compute traffic-specific metrics (Wiedemann-style)"""
        pred_tensor = torch.from_numpy(pred).float()
        target_tensor = torch.from_numpy(target).float()
        
        # Wiedemann MSE - gives different weights to volume/speed
        if pred.ndim == 4:  # (B, T*C, H, W)
            B, TC, H, W = pred.shape
            T = TC // 8
            pred_tensor = pred_tensor.reshape(B, T, 8, H, W)
            target_tensor = target_tensor.reshape(B, T, 8, H, W)
        
        # Calculate volume and speed contributions separately
        vol_channels = [0, 2, 4, 6]
        speed_channels = [1, 3, 5, 7]
        
        # Volume MSE
        vol_mse = F.mse_loss(pred_tensor[..., vol_channels, :, :], 
                            target_tensor[..., vol_channels, :, :])
        
        # Speed MSE (only where volume > 0)
        volume_mask = target_tensor[..., vol_channels, :, :] > 0
        speed_pred = pred_tensor[..., speed_channels, :, :] * volume_mask
        speed_target = target_tensor[..., speed_channels, :, :] * volume_mask
        
        speed_mse = F.mse_loss(speed_pred, speed_target)
        
        # Weighted combination (similar to Wiedemann loss)
        total_pixels = np.prod(pred.shape)
        non_zero_pixels = torch.sum(volume_mask).item()
        weight = non_zero_pixels / total_pixels if total_pixels > 0 else 0.0
        
        wiedemann_mse = vol_mse + weight * speed_mse
        
        return {
            'wiedemann_mse': wiedemann_mse.item(),
            'volume_contribution': vol_mse.item(),
            'speed_contribution': speed_mse.item(),
            'traffic_density': weight
        }
    
    def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all metrics for traffic prediction"""
        
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = self.compute_mse(pred, target, mask)
        metrics['mae'] = self.compute_mae(pred, target, mask)
        metrics['rmse'] = self.compute_rmse(pred, target, mask)
        
        # Volume metrics
        try:
            volume_metrics = self.compute_volume_metrics(pred, target)
            metrics.update(volume_metrics)
        except Exception as e:
            logging.warning(f"Could not compute volume metrics: {e}")
        
        # Speed metrics
        try:
            speed_metrics = self.compute_speed_metrics(pred, target)
            metrics.update(speed_metrics)
        except Exception as e:
            logging.warning(f"Could not compute speed metrics: {e}")
        
        # Traffic-specific metrics
        try:
            traffic_metrics = self.compute_traffic_specific_metrics(pred, target)
            metrics.update(traffic_metrics)
        except Exception as e:
            logging.warning(f"Could not compute traffic metrics: {e}")
        
        return metrics
# src/training/callbacks.py
"""
Enhanced callbacks with improved checkpointing and CSV logging
"""

import torch
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List
import csv
import json
import shutil
from datetime import datetime

class CheckpointManager:
    """
    Enhanced checkpoint manager that saves only the best models and maintains detailed logs
    """
    
    def __init__(self, 
                 save_dir: Path,
                 monitor_metric: str = 'competition_score',
                 mode: str = 'min',
                 save_top_k: int = 3,
                 save_last: bool = True):
        """
        Initialize enhanced checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            monitor_metric: Metric to monitor for best model selection
            mode: 'min' or 'max' for monitoring metric
            save_top_k: Number of best checkpoints to keep
            save_last: Whether to save the last checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        
        # Checkpoint tracking
        self.best_checkpoints = []  # List of (metric_value, epoch, filepath)
        self.checkpoint_history = []
        
        # CSV logging
        self.csv_path = self.save_dir / "training_log.csv"
        self.csv_initialized = False
        
        # Best metrics tracking
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        logging.info(f"Enhanced checkpoint manager initialized at {self.save_dir}")
        logging.info(f"Monitoring '{monitor_metric}' ({'minimize' if mode == 'min' else 'maximize'})")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler=None, additional_data: Optional[Dict] = None) -> bool:
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for all phases (train, val, test)
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler
            additional_data: Additional data to save (e.g., config, training history)
            
        Returns:
            True if this was a new best epoch
        """
        # Log to CSV
        self._log_to_csv(epoch, metrics)
        
        # Get the monitoring metric value
        monitor_value = self._extract_monitor_value(metrics)
        
        if monitor_value is None:
            logging.warning(f"Monitor metric '{self.monitor_metric}' not found in metrics")
            return False
        
        # Check if this is a new best
        is_best = self._is_better(monitor_value, self.best_metric_value)
        
        if is_best:
            self.best_metric_value = monitor_value
            self.best_epoch = epoch
            
            # Save best checkpoint
            checkpoint_path = self._save_checkpoint(
                epoch, model, optimizer, scheduler, metrics, additional_data, is_best=True
            )
            
            # Update best checkpoints list
            self._update_best_checkpoints(monitor_value, epoch, checkpoint_path)
            
            logging.info(f"New best {self.monitor_metric}: {monitor_value:.6f} at epoch {epoch}")
            
        # Save last checkpoint if requested
        if self.save_last:
            self._save_checkpoint(
                epoch, model, optimizer, scheduler, metrics, additional_data, is_best=False
            )
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return is_best
    
    def _extract_monitor_value(self, metrics: Dict[str, Any]) -> Optional[float]:
        """Extract the monitoring metric value from metrics dictionary"""
        
        # Try different metric locations
        locations_to_try = [
            metrics,  # Direct access
            metrics.get('val', {}),  # Validation metrics
            metrics.get('test', {}),  # Test metrics
            metrics.get('validation', {}),  # Alternative validation key
        ]
        
        for location in locations_to_try:
            if isinstance(location, dict) and self.monitor_metric in location:
                value = location[self.monitor_metric]
                if isinstance(value, torch.Tensor):
                    return value.item()
                return float(value)
        
        return None
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best"""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _save_checkpoint(self, epoch: int, model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer, scheduler,
                        metrics: Dict[str, Any], additional_data: Optional[Dict],
                        is_best: bool = False) -> Path:
        """Save model checkpoint"""
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric_value': self.best_metric_value,
            'monitor_metric': self.monitor_metric,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add scheduler state if available
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add additional data
        if additional_data:
            checkpoint.update(additional_data)
        
        # Determine filename
        if is_best:
            filename = f"best_model_epoch_{epoch:04d}.pth"
        else:
            filename = f"last_model_epoch_{epoch:04d}.pth"
        
        checkpoint_path = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Also save a copy with generic name for easy access
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_path)
        
        if not is_best and self.save_last:
            last_path = self.save_dir / "last_model.pth"
            shutil.copy2(checkpoint_path, last_path)
        
        logging.debug(f"Saved {'best' if is_best else 'last'} checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def _update_best_checkpoints(self, metric_value: float, epoch: int, checkpoint_path: Path):
        """Update the list of best checkpoints"""
        
        # Add current checkpoint
        self.best_checkpoints.append((metric_value, epoch, checkpoint_path))
        
        # Sort by metric value
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == 'max'))
        
        # Keep only top k
        if len(self.best_checkpoints) > self.save_top_k:
            # Remove excess checkpoints from disk
            for metric_val, ep, path in self.best_checkpoints[self.save_top_k:]:
                if path.exists():
                    path.unlink()
                    logging.debug(f"Removed old checkpoint: {path}")
            
            # Keep only top k in memory
            self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
    
    def _cleanup_checkpoints(self):
        """Clean up old non-best checkpoints"""
        
        # Keep only recent last checkpoints (e.g., last 5)
        last_checkpoints = sorted(
            self.save_dir.glob("last_model_epoch_*.pth"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old last checkpoints (keep only 5 most recent)
        for old_checkpoint in last_checkpoints[5:]:
            old_checkpoint.unlink()
            logging.debug(f"Removed old last checkpoint: {old_checkpoint}")
    
    def _log_to_csv(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics to CSV file"""
        
        # Flatten metrics dictionary
        flattened_metrics = self._flatten_metrics_dict(metrics)
        
        # Add epoch and timestamp
        row_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **flattened_metrics
        }
        
        # Initialize CSV if needed
        if not self.csv_initialized:
            self._initialize_csv(row_data.keys())
        
        # Append to CSV
        try:
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row_data.keys())
                writer.writerow(row_data)
            
        except Exception as e:
            logging.warning(f"Failed to log to CSV: {e}")
    
    def _initialize_csv(self, fieldnames: List[str]):
        """Initialize CSV file with headers"""
        try:
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            self.csv_initialized = True
            logging.info(f"Initialized training log CSV: {self.csv_path}")
            
        except Exception as e:
            logging.warning(f"Failed to initialize CSV: {e}")
    
    def _flatten_metrics_dict(self, metrics: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """Flatten nested metrics dictionary"""
        
        flattened = {}
        
        for key, value in metrics.items():
            new_key = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(self._flatten_metrics_dict(value, new_key))
            else:
                # Convert tensor to float if needed
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.cpu().numpy().tolist()
                
                # Only include numeric values
                if isinstance(value, (int, float)):
                    flattened[new_key] = value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # Handle lists/tuples of numbers
                    if all(isinstance(x, (int, float)) for x in value):
                        flattened[f"{new_key}_mean"] = sum(value) / len(value)
        
        return flattened
    
    def get_best_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about the best checkpoint"""
        if not self.best_checkpoints:
            return {}
        
        best_metric, best_epoch, best_path = self.best_checkpoints[0]
        
        return {
            'best_epoch': best_epoch,
            'best_metric_value': best_metric,
            'best_checkpoint_path': str(best_path),
            'monitor_metric': self.monitor_metric
        }
    
    def load_best_checkpoint(self, model: torch.nn.Module, 
                           optimizer: Optional[torch.optim.Optimizer] = None,
                           scheduler=None) -> Dict[str, Any]:
        """Load the best checkpoint"""
        
        best_path = self.save_dir / "best_model.pth"
        
        if not best_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
        
        checkpoint = torch.load(best_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logging.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def save_training_summary(self, final_metrics: Dict[str, Any]):
        """Save final training summary"""
        
        summary = {
            'training_completed': datetime.now().isoformat(),
            'best_epoch': self.best_epoch,
            'best_metric_value': self.best_metric_value,
            'monitor_metric': self.monitor_metric,
            'total_checkpoints_saved': len(self.best_checkpoints),
            'final_metrics': final_metrics,
            'best_checkpoints': [
                {
                    'epoch': epoch,
                    'metric_value': metric_val,
                    'checkpoint_path': str(path)
                }
                for metric_val, epoch, path in self.best_checkpoints
            ]
        }
        
        # Save summary as JSON
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training summary saved: {summary_path}")
        
        return summary


class CSVMetricsLogger:
    """Standalone CSV logger for metrics"""
    
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialized = False
        
    def log(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics to CSV"""
        
        # Prepare row data
        row_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        # Flatten and add metrics
        flattened = self._flatten_dict(metrics)
        row_data.update(flattened)
        
        # Initialize if needed
        if not self.initialized:
            self._initialize_csv(row_data.keys())
        
        # Write row
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            writer.writerow(row_data)
    
    def _initialize_csv(self, fieldnames):
        """Initialize CSV with headers"""
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        self.initialized = True
    
    def _flatten_dict(self, d: Dict, prefix: str = '') -> Dict:
        """Flatten nested dictionary"""
        result = {}
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                result.update(self._flatten_dict(value, new_key))
            elif isinstance(value, (int, float)):
                result[new_key] = value
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                result[new_key] = value.item()
        return result
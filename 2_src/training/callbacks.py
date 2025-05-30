# src/training/callbacks.py
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

class TrainingCallbacks:
    """Callbacks for training process"""
    
    def __init__(self, 
                 save_dir: Path,
                 save_best_only: bool = True,
                 early_stopping_patience: int = 20,
                 monitor_metric: str = 'loss',
                 mode: str = 'min'):
        """
        Initialize training callbacks
        
        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Whether to save only the best model
            early_stopping_patience: Patience for early stopping
            monitor_metric: Metric to monitor for callbacks
            mode: 'min' or 'max' for monitoring metric
        """
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.early_stopping_patience = early_stopping_patience
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Early stopping state
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        # Checkpointing state
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> bool:
        """
        Called at the end of each epoch
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            model: Model to save
            optimizer: Optimizer to save
            
        Returns:
            True if training should stop (early stopping)
        """
        current_metric = metrics.get(self.monitor_metric, None)
        
        if current_metric is None:
            logging.warning(f"Metric '{self.monitor_metric}' not found in metrics")
            return False
        
        # Check if current metric is better
        is_better = self._is_better(current_metric, self.best_metric)
        
        if is_better:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # Save best model
            self._save_checkpoint(epoch, model, optimizer, metrics, is_best=True)
            logging.info(f"New best {self.monitor_metric}: {current_metric:.6f}")
            
        else:
            self.patience_counter += 1
            
        # Save regular checkpoint
        if not self.save_best_only:
            self._save_checkpoint(epoch, model, optimizer, metrics, is_best=False)
        
        # Check early stopping
        if self.patience_counter >= self.early_stopping_patience:
            logging.info(
                f"Early stopping triggered. "
                f"Best {self.monitor_metric}: {self.best_metric:.6f} at epoch {self.best_epoch}"
            )
            return True
        
        return False
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best"""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _save_checkpoint(self, epoch: int, model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer, metrics: Dict[str, float],
                        is_best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            logging.info(f"Saved best model to {checkpoint_path}")

class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop
        
        Args:
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
            
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self, save_dir: Path, save_best_only: bool = True, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_best_only: Whether to save only best model
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, epoch: int, model: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, metrics: Dict[str, float]):
        """Save checkpoint if needed"""
        
        current_value = metrics.get(self.monitor, None)
        if current_value is None:
            return
            
        is_better = (current_value < self.best_value) if self.mode == 'min' else (current_value > self.best_value)
        
        if is_better:
            self.best_value = current_value
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'monitor_value': current_value
            }
            
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            logging.info(f"Saved best model (epoch {epoch}, {self.monitor}: {current_value:.6f})")
        
        if not self.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pth')
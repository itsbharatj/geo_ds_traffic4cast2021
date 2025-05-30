# src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import json

from .metrics import TrafficMetrics
from .callbacks import TrainingCallbacks

class Traffic4CastTrainer:
    """Main trainer class for Traffic4Cast experiments"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config,
                 device: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            config: Configuration object
            device: Device to use for training
        """
        self.config = config
        self.device = device or config.training.device
        
        # Setup model
        self.model = model.to(self.device)
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        # Setup metrics
        self.metrics = TrafficMetrics()
        
        # Setup logging and checkpointing
        self.setup_logging()
        
        # Setup callbacks
        self.callbacks = TrainingCallbacks(
            save_dir=self.save_dir,
            save_best_only=True,
            early_stopping_patience=20
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],  
            'test_loss': [],
            'learning_rate': []
        }
        
    def setup_logging(self):
        """Setup logging and experiment directory"""
        
        # Create experiment directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.logging.experiment_name}_{timestamp}"
        self.save_dir = Path(self.config.logging.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        if self.config.logging.use_tensorboard:
            self.writer = SummaryWriter(str(self.save_dir / "tensorboard"))
        else:
            self.writer = None
            
        # Save configuration
        self.config.save(str(self.save_dir / "config.yaml"))
        
        logging.info(f"Experiment directory: {self.save_dir}")
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle output format - reshape if needed
            if outputs.dim() == 4 and targets.dim() == 5:
                # outputs: (B, T*C, H, W), targets: (B, T, H, W, C)
                B, T, H, W, C = targets.shape
                targets = targets.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
                targets = targets.reshape(B, T * C, H, W)  # (B, T*C, H, W)
            elif outputs.dim() == 5 and targets.dim() == 4:
                # Convert outputs to match targets
                B, TC, H, W = outputs.shape
                T = TC // 8  # Assuming 8 channels
                C = 8
                outputs = outputs.reshape(B, T, C, H, W)
                outputs = outputs.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
                outputs = outputs.reshape(B, T * C, H * W)
                targets = targets.reshape(B, T * C, H * W)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.logging.log_interval == 0:
                logging.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx}/{len(self.train_loader)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate model on validation set"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle output format (same as training)
                if outputs.dim() == 4 and targets.dim() == 5:
                    B, T, H, W, C = targets.shape
                    targets = targets.permute(0, 1, 4, 2, 3)
                    targets = targets.reshape(B, T * C, H, W)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute detailed metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        detailed_metrics = self.metrics.compute_all_metrics(
            all_outputs.numpy(), 
            all_targets.numpy()
        )
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def test_epoch(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle output format (same as training)
                if outputs.dim() == 4 and targets.dim() == 5:
                    B, T, H, W, C = targets.shape
                    targets = targets.permute(0, 1, 4, 2, 3)
                    targets = targets.reshape(B, T * C, H, W)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute detailed metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        detailed_metrics = self.metrics.compute_all_metrics(
            all_outputs.numpy(), 
            all_targets.numpy()
        )
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def fit(self) -> Dict[str, List[float]]:
        """Main training loop"""
        
        logging.info("Starting training...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model: {self.model.__class__.__name__}")
        logging.info(f"Training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Test (optional - can be expensive)
            test_metrics = self.test_epoch()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['test_loss'].append(test_metrics['loss'])
            self.training_history['learning_rate'].append(current_lr)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logging.info(
                f'Epoch {epoch}: '
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_metrics["loss"]:.4f}, '
                f'Test Loss: {test_metrics["loss"]:.4f}, '
                f'LR: {current_lr:.2e}, '
                f'Time: {epoch_time:.1f}s'
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/Test', test_metrics['loss'], epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                # Log detailed metrics
                for key, value in val_metrics.items():
                    if key != 'loss':
                        self.writer.add_scalar(f'Metrics/Val_{key}', value, epoch)
                        
                for key, value in test_metrics.items():
                    if key != 'loss':
                        self.writer.add_scalar(f'Metrics/Test_{key}', value, epoch)
            
            # Callbacks (checkpointing, early stopping)
            should_stop = self.callbacks.on_epoch_end(
                epoch, val_metrics['loss'], self.model, self.optimizer
            )
            
            # Update best validation loss  
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
            
            if should_stop:
                logging.info("Early stopping triggered")
                break
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.1f}s")
        
        # Save final results
        self.save_results()
        
        if self.writer:
            self.writer.close()
            
        return self.training_history
    
    def save_results(self):
        """Save training results and final model"""
        
        # Save training history
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save final model
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config.__dict__,
                'training_history': self.training_history,
                'best_val_loss': self.best_val_loss
            },
            self.save_dir / "final_model.pth"
        )
        
        logging.info(f"Results saved to {self.save_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            
        logging.info(f"Loaded checkpoint from {checkpoint_path}")

# Utility function to create trainer
def create_trainer(model, train_loader, val_loader, test_loader, config) -> Traffic4CastTrainer:
    """Factory function to create trainer"""
    return Traffic4CastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        test_loader=test_loader,
        config=config
    )
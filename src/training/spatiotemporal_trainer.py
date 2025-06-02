# src/training/spatiotemporal_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json

from .metrics import TrafficMetrics
from .callbacks import TrainingCallbacks

class SpatioTemporalTransferTrainer:
    """
    Advanced trainer for spatio-temporal transfer learning with multi-task training
    and few-shot adaptation.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 adapt_loader: DataLoader,
                 test_loader: DataLoader,
                 config,
                 device: Optional[str] = None):
        """
        Initialize spatio-temporal transfer trainer
        
        Args:
            model: Multi-task model
            train_loader: Training data (multi-city, multi-year)
            adapt_loader: Adaptation data (test city, source year)
            test_loader: Test data (test city, target year)
            config: Configuration object
            device: Device to use for training
        """
        
        self.config = config
        self.device = device or config.training.device
        
        # Setup model
        self.model = model.to(self.device)
        
        # Setup data loaders
        self.train_loader = train_loader
        self.adapt_loader = adapt_loader
        self.test_loader = test_loader
        
        # Multi-task loss functions
        self.traffic_criterion = nn.MSELoss()
        self.city_criterion = nn.CrossEntropyLoss()
        self.year_criterion = nn.CrossEntropyLoss()
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'traffic': getattr(config.training, 'traffic_weight', 1.0),
            'city': getattr(config.training, 'city_weight', 0.1),
            'year': getattr(config.training, 'year_weight', 0.1),
            'enhanced_traffic': getattr(config.training, 'enhanced_traffic_weight', 0.5)
        }
        
        # Optimizers
        self.main_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Separate optimizer for meta-learning (faster learning rate)
        if hasattr(self.model, 'meta_learner'):
            self.meta_optimizer = optim.Adam(
                self.model.meta_learner.parameters(),
                lr=config.training.learning_rate * 10,  # Faster for adaptation
                weight_decay=config.training.weight_decay
            )
        else:
            self.meta_optimizer = None
        
        # Learning rate schedulers
        self.main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.main_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        if self.meta_optimizer:
            self.meta_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.meta_optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        
        # Setup metrics
        self.metrics = TrafficMetrics()
        
        # Setup logging and checkpointing
        self.setup_logging()
        
        # Setup callbacks
        self.callbacks = TrainingCallbacks(
            save_dir=self.save_dir,
            save_best_only=True,
            early_stopping_patience=30  # Longer patience for transfer learning
        )
        
        # Training state
        self.current_epoch = 0
        self.best_transfer_score = float('inf')
        self.training_history = {
            'train_loss': [],
            'adapt_loss': [],
            'test_loss': [],
            'transfer_score': [],
            'task_losses': {
                'traffic': [],
                'city': [],
                'year': [],
                'enhanced_traffic': []
            }
        }
    
    def setup_logging(self):
        """Setup logging and experiment directory"""
        
        # Create experiment directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config.logging.experiment_name}_spatiotemporal_{timestamp}"
        self.save_dir = Path(self.config.logging.save_dir) / exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        if self.config.logging.use_tensorboard:
            self.writer = SummaryWriter(str(self.save_dir / "tensorboard"))
        else:
            self.writer = None
            
        # Save configuration
        self.config.save(str(self.save_dir / "config.yaml"))
        
        logging.info(f"Spatio-temporal experiment directory: {self.save_dir}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with multi-task learning"""
        
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'traffic': 0.0,
            'city': 0.0,
            'year': 0.0,
            'enhanced_traffic': 0.0
        }
        num_batches = 0
        
        for batch_idx, (inputs, targets, metadata) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Extract labels from metadata
            city_labels = torch.tensor([m['city_label'] for m in metadata]).to(self.device)
            year_labels = torch.tensor([m['year_label'] for m in metadata]).to(self.device)
            
            # Forward pass
            self.main_optimizer.zero_grad()
            outputs = self.model(inputs, metadata, mode="train")
            
            # Handle output format for targets
            if outputs['traffic'].dim() == 4 and targets.dim() == 5:
                B, T, H, W, C = targets.shape
                targets = targets.permute(0, 1, 4, 2, 3)
                targets = targets.reshape(B, T * C, H, W)
            
            # Compute multi-task losses
            traffic_loss = self.traffic_criterion(outputs['traffic'], targets)
            city_loss = self.city_criterion(outputs['city'], city_labels)
            year_loss = self.year_criterion(outputs['year'], year_labels)
            
            # Enhanced traffic loss (if available)
            enhanced_traffic_loss = 0.0
            if 'enhanced_traffic' in outputs:
                enhanced_traffic_loss = self.traffic_criterion(outputs['enhanced_traffic'], targets)
            
            # Combined loss
            total_loss = (
                self.loss_weights['traffic'] * traffic_loss +
                self.loss_weights['city'] * city_loss +
                self.loss_weights['year'] * year_loss +
                self.loss_weights['enhanced_traffic'] * enhanced_traffic_loss
            )
            
            # Backward pass
            total_loss.backward()
            self.main_optimizer.step()
            
            # Update statistics
            total_losses['total'] += total_loss.item()
            total_losses['traffic'] += traffic_loss.item()
            total_losses['city'] += city_loss.item()
            total_losses['year'] += year_loss.item()
            if isinstance(enhanced_traffic_loss, torch.Tensor):
                total_losses['enhanced_traffic'] += enhanced_traffic_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.logging.log_interval == 0:
                logging.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx}/{len(self.train_loader)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Total Loss: {total_loss.item():.6f}, '
                    f'Traffic: {traffic_loss.item():.6f}, '
                    f'City: {city_loss.item():.6f}, '
                    f'Year: {year_loss.item():.6f}'
                )
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def adapt_epoch(self) -> Dict[str, float]:
        """Perform few-shot adaptation on target city"""
        
        self.model.train()  # Keep in training mode for adaptation
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets, metadata) in enumerate(self.adapt_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass in adaptation mode
            if self.meta_optimizer:
                self.meta_optimizer.zero_grad()
            
            outputs = self.model(inputs, metadata, mode="adapt")
            
            # Handle output format
            if outputs['traffic'].dim() == 4 and targets.dim() == 5:
                B, T, H, W, C = targets.shape
                targets = targets.permute(0, 1, 4, 2, 3)
                targets = targets.reshape(B, T * C, H, W)
            
            # Adaptation loss
            if 'adapted_traffic' in outputs:
                adapt_loss = self.traffic_criterion(outputs['adapted_traffic'], targets)
            else:
                adapt_loss = self.traffic_criterion(outputs['traffic'], targets)
            
            # Backward pass for meta-learning
            if self.meta_optimizer:
                adapt_loss.backward()
                self.meta_optimizer.step()
            
            total_loss += adapt_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        logging.info(f"Adaptation completed. Average loss: {avg_loss:.6f}")
        return {'adapt_loss': avg_loss}
    
    def test_epoch(self) -> Dict[str, float]:
        """Evaluate on test set (target city, target year)"""
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, metadata in self.test_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass in test mode
                outputs = self.model(inputs, metadata, mode="test")
                
                # Handle output format
                if outputs['traffic'].dim() == 4 and targets.dim() == 5:
                    B, T, H, W, C = targets.shape
                    targets = targets.permute(0, 1, 4, 2, 3)
                    targets = targets.reshape(B, T * C, H, W)
                
                # Compute loss
                loss = self.traffic_criterion(outputs['traffic'], targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(outputs['traffic'].cpu())
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
    
    def compute_transfer_score(self, test_metrics: Dict[str, float]) -> float:
        """
        Compute transfer learning score
        
        Lower is better. Combines multiple metrics to assess transfer quality.
        """
        
        # Weighted combination of metrics
        transfer_score = (
            test_metrics.get('mse', 0.0) * 0.4 +
            test_metrics.get('mae', 0.0) * 0.3 +
            test_metrics.get('volume_mse', 0.0) * 0.2 +
            test_metrics.get('speed_mse', 0.0) * 0.1
        )
        
        return transfer_score
    
    def fit(self) -> Dict[str, List[float]]:
        """Main training loop for spatio-temporal transfer learning"""
        
        logging.info("Starting spatio-temporal transfer learning...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model: {self.model.__class__.__name__}")
        logging.info(f"Training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Adaptation samples: {len(self.adapt_loader.dataset)}")
        logging.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 1. Multi-task training on source cities
            train_metrics = self.train_epoch()
            
            # 2. Few-shot adaptation on target city
            adapt_metrics = self.adapt_epoch()
            
            # 3. Evaluation on target city test data
            test_metrics = self.test_epoch()
            
            # 4. Compute transfer score
            transfer_score = self.compute_transfer_score(test_metrics)
            
            # Update learning rates
            self.main_scheduler.step(train_metrics['total'])
            if self.meta_scheduler:
                self.meta_scheduler.step(adapt_metrics['adapt_loss'])
            
            current_lr = self.main_optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['total'])
            self.training_history['adapt_loss'].append(adapt_metrics['adapt_loss'])
            self.training_history['test_loss'].append(test_metrics['loss'])
            self.training_history['transfer_score'].append(transfer_score)
            
            # Update task-specific losses
            for task in ['traffic', 'city', 'year', 'enhanced_traffic']:
                if task in train_metrics:
                    self.training_history['task_losses'][task].append(train_metrics[task])
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            logging.info(
                f'Epoch {epoch}: '
                f'Train: {train_metrics["total"]:.4f}, '
                f'Adapt: {adapt_metrics["adapt_loss"]:.4f}, '
                f'Test: {test_metrics["loss"]:.4f}, '
                f'Transfer Score: {transfer_score:.4f}, '
                f'LR: {current_lr:.2e}, '
                f'Time: {epoch_time:.1f}s'
            )
            
            # Detailed task losses
            logging.info(
                f'  Task Losses - Traffic: {train_metrics["traffic"]:.4f}, '
                f'City: {train_metrics["city"]:.4f}, '
                f'Year: {train_metrics["year"]:.4f}'
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/Train_Total', train_metrics['total'], epoch)
                self.writer.add_scalar('Loss/Adapt', adapt_metrics['adapt_loss'], epoch)
                self.writer.add_scalar('Loss/Test', test_metrics['loss'], epoch)
                self.writer.add_scalar('Transfer/Score', transfer_score, epoch)
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                
                # Task-specific losses
                for task, loss in train_metrics.items():
                    if task != 'total':
                        self.writer.add_scalar(f'Task_Loss/{task}', loss, epoch)
                
                # Detailed test metrics
                for key, value in test_metrics.items():
                    if key != 'loss':
                        self.writer.add_scalar(f'Test_Metrics/{key}', value, epoch)
            
            # Callbacks (checkpointing, early stopping)
            callback_metrics = {'transfer_score': transfer_score, **test_metrics}
            should_stop = self.callbacks.on_epoch_end(
                epoch, callback_metrics, self.model, self.main_optimizer
            )
            
            # Update best transfer score
            if transfer_score < self.best_transfer_score:
                self.best_transfer_score = transfer_score
                logging.info(f"New best transfer score: {transfer_score:.6f}")
            
            if should_stop:
                logging.info("Early stopping triggered")
                break
        
        total_time = time.time() - start_time
        logging.info(f"Spatio-temporal transfer training completed in {total_time:.1f}s")
        
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
        
        # Save detailed results
        results = {
            'best_transfer_score': self.best_transfer_score,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_test_loss': self.training_history['test_loss'][-1],
            'final_transfer_score': self.training_history['transfer_score'][-1],
            'config': {
                'train_cities': self.config.experiment.train_cities,
                'train_years': self.config.experiment.train_years,
                'test_city': self.config.experiment.test_city,
                'test_train_year': self.config.experiment.test_train_year,
                'test_target_year': self.config.experiment.test_target_year
            }
        }
        
        with open(self.save_dir / "transfer_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'main_optimizer_state_dict': self.main_optimizer.state_dict(),
                'meta_optimizer_state_dict': self.meta_optimizer.state_dict() if self.meta_optimizer else None,
                'config': self.config.__dict__,
                'training_history': self.training_history,
                'best_transfer_score': self.best_transfer_score
            },
            self.save_dir / "final_model.pth"
        )
        
        logging.info(f"Spatio-temporal transfer results saved to {self.save_dir}")

def create_spatiotemporal_trainer(model, train_loader, adapt_loader, test_loader, config) -> SpatioTemporalTransferTrainer:
    """Factory function to create spatio-temporal transfer trainer"""
    return SpatioTemporalTransferTrainer(
        model=model,
        train_loader=train_loader,
        adapt_loader=adapt_loader,
        test_loader=test_loader,
        config=config
    )
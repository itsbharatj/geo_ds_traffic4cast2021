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
from typing import Dict, Optional, List, Tuple, Any
import json

from .metrics import TrafficMetrics
from .callbacks import TrainingCallbacks

class Traffic4CastTrainer:
    """
    Unified trainer for all Traffic4Cast experiments including enhanced spatio-temporal transfer
    """
    
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
            test_loader: Test data loader (or adaptation loader for enhanced spatio-temporal)
            config: Configuration object
            device: Device to use for training
        """
        self.config = config
        self.device = device or self._detect_device()
        
        # Setup model
        self.model = model.to(self.device)
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Detect experiment type
        self.experiment_type = config.experiment.type
        self.is_enhanced_spatiotemporal = (self.experiment_type == "enhanced_spatiotemporal_transfer")
        
        # Setup loss functions
        self._setup_loss_functions()
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.main_optimizer, 
            mode='min', 
            factor=0.5, 
            patience=15 if self.is_enhanced_spatiotemporal else 10,
            verbose=True
        )
        
        # Setup metrics
        self.metrics = TrafficMetrics()
        
        # Setup logging and checkpointing
        self.setup_logging()
        
        # Setup callbacks
        patience = 30 if self.is_enhanced_spatiotemporal else 20
        self.callbacks = TrainingCallbacks(
            save_dir=self.save_dir,
            save_best_only=True,
            early_stopping_patience=patience
        )
        
        # Initialize training state
        self._init_training_state()
        
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _setup_loss_functions(self):
        """Setup loss functions based on experiment type"""
        self.traffic_criterion = nn.MSELoss()
        
        if self.is_enhanced_spatiotemporal:
            self.city_criterion = nn.CrossEntropyLoss()
            self.year_criterion = nn.CrossEntropyLoss()
            
            # Multi-task loss weights
            self.loss_weights = {
                'traffic': getattr(self.config.training, 'traffic_weight', 1.0),
                'city': getattr(self.config.training, 'city_weight', 0.1),
                'year': getattr(self.config.training, 'year_weight', 0.1),
                'enhanced_traffic': getattr(self.config.training, 'enhanced_traffic_weight', 0.5)
            }
            
            logging.info(f"Multi-task loss weights: {self.loss_weights}")
    
    def _setup_optimizers(self):
        """Setup optimizers"""
        self.main_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Additional optimizer for meta-learning components
        self.meta_optimizer = None
        if (self.is_enhanced_spatiotemporal and 
            hasattr(self.model, 'meta_adapter') and 
            self.model.meta_adapter is not None):
            
            self.meta_optimizer = optim.Adam(
                self.model.meta_adapter.parameters(),
                lr=self.config.training.learning_rate * 5,  # Faster learning for adaptation
                weight_decay=self.config.training.weight_decay
            )
            logging.info("Meta-learning optimizer created")
    
    def _init_training_state(self):
        """Initialize training state variables"""
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_transfer_score = float('inf') if self.is_enhanced_spatiotemporal else None
        
        # Initialize training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],  
            'test_loss': [],
            'learning_rate': []
        }
        
        if self.is_enhanced_spatiotemporal:
            self.training_history.update({
                'transfer_score': [],
                'adaptation_loss': [],
                'task_losses': {
                    'traffic': [],
                    'city': [],
                    'year': [],
                    'enhanced_traffic': []
                }
            })
    
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
    
    def _handle_batch_data(self, batch_data):
        """Handle different batch data formats"""
        if len(batch_data) == 3:  # Enhanced format with metadata
            inputs, targets, metadata = batch_data
            return inputs, targets, metadata
        else:  # Standard format
            inputs, targets = batch_data
            return inputs, targets, None
    
    def _extract_labels_from_metadata(self, metadata):
        """Extract city and year labels from metadata (supports list-of-dicts or dict-of-lists/tensors)"""
        if metadata is None:
            return None, None
        # If metadata is a list of dicts (standard)
        if isinstance(metadata, list):
            city_labels = torch.tensor([m['city_label'] for m in metadata]).to(self.device)
            year_labels = torch.tensor([m['year_label'] for m in metadata]).to(self.device)
            return city_labels, year_labels
        # If metadata is a dict (as in enhanced spatio-temporal loader)
        elif isinstance(metadata, dict):
            # Accept both lists and tensors
            city_labels = metadata['city_label']
            year_labels = metadata['year_label']
            # Convert to tensor if needed
            if not torch.is_tensor(city_labels):
                city_labels = torch.tensor(city_labels)
            if not torch.is_tensor(year_labels):
                year_labels = torch.tensor(year_labels)
            city_labels = city_labels.to(self.device)
            year_labels = year_labels.to(self.device)
            return city_labels, year_labels
        else:
            raise TypeError(f"Unsupported metadata format: {type(metadata)}")
    
    def _handle_output_format(self, outputs, targets):
        """Handle different output formats and reshape targets if needed"""
        # Get the traffic prediction output
        if isinstance(outputs, dict):
            traffic_output = outputs['traffic']
        else:
            traffic_output = outputs
        
        # Handle format mismatch between outputs and targets
        if traffic_output.dim() == 4 and targets.dim() == 5:
            # targets: (B, T, H, W, C) -> (B, T*C, H, W)
            B, T, H, W, C = targets.shape
            targets = targets.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            targets = targets.reshape(B, T * C, H, W)  # (B, T*C, H, W)
        elif traffic_output.dim() == 5 and targets.dim() == 4:
            # outputs: (B, T, C, H, W) -> (B, T*C, H, W)
            B, T, C, H, W = traffic_output.shape
            traffic_output = traffic_output.reshape(B, T * C, H, W)
            if isinstance(outputs, dict):
                outputs['traffic'] = traffic_output
            else:
                outputs = traffic_output
        
        return outputs, targets
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        if self.is_enhanced_spatiotemporal:
            return self._train_epoch_enhanced()
        else:
            return self._train_epoch_standard()
    
    def _train_epoch_standard(self) -> Dict[str, float]:
        """Standard training epoch for spatial/temporal transfer"""
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            inputs, targets, _ = self._handle_batch_data(batch_data)
            
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.main_optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle output format
            outputs, targets = self._handle_output_format(outputs, targets)
            
            # Compute loss
            loss = self.traffic_criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.main_optimizer.step()
            
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
    
    def _train_epoch_enhanced(self) -> Dict[str, float]:
        """Enhanced spatio-temporal training epoch with multi-task learning"""
        total_losses = {
            'total': 0.0,
            'traffic': 0.0,
            'city': 0.0,
            'year': 0.0,
            'enhanced_traffic': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            inputs, targets, metadata = self._handle_batch_data(batch_data)
            city_labels, year_labels = self._extract_labels_from_metadata(metadata)
            
            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.main_optimizer.zero_grad()
            outputs = self.model(inputs, metadata, mode="train")
            
            # Handle output format
            outputs, targets = self._handle_output_format(outputs, targets)
            
            # Compute multi-task losses
            traffic_loss = self.traffic_criterion(outputs['traffic'], targets)
            
            city_loss = torch.tensor(0.0).to(self.device)
            year_loss = torch.tensor(0.0).to(self.device)
            enhanced_traffic_loss = torch.tensor(0.0).to(self.device)
            
            if city_labels is not None and 'city' in outputs:
                city_loss = self.city_criterion(outputs['city'], city_labels)
            
            if year_labels is not None and 'year' in outputs:
                year_loss = self.year_criterion(outputs['year'], year_labels)
            
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
            total_losses['enhanced_traffic'] += enhanced_traffic_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.logging.log_interval == 0:
                logging.info(
                    f'Train Epoch: {self.current_epoch} '
                    f'[{batch_idx}/{len(self.train_loader)} '
                    f'({100. * batch_idx / len(self.train_loader):.0f}%)]\t'
                    f'Total: {total_loss.item():.6f}, '
                    f'Traffic: {traffic_loss.item():.6f}, '
                    f'City: {city_loss.item():.4f}, '
                    f'Year: {year_loss.item():.4f}'
                )
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate model on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                inputs, targets, metadata = self._handle_batch_data(batch_data)
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.is_enhanced_spatiotemporal:
                    outputs = self.model(inputs, metadata, mode="train")
                    traffic_output = outputs['traffic']
                else:
                    traffic_output = self.model(inputs)
                
                # Handle output format
                _, targets = self._handle_output_format({'traffic': traffic_output}, targets)
                
                # Compute loss
                loss = self.traffic_criterion(traffic_output, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(traffic_output.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute detailed metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics.compute_all_metrics(
                all_outputs.numpy(), 
                all_targets.numpy()
            )
        else:
            detailed_metrics = {}
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def test_epoch(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        if self.is_enhanced_spatiotemporal:
            return self._test_epoch_enhanced()
        else:
            return self._test_epoch_standard()
    
    def _test_epoch_standard(self) -> Dict[str, float]:
        """Standard test epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                inputs, targets, _ = self._handle_batch_data(batch_data)
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle output format
                outputs, targets = self._handle_output_format(outputs, targets)
                
                # Compute loss
                loss = self.traffic_criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute detailed metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics.compute_all_metrics(
                all_outputs.numpy(), 
                all_targets.numpy()
            )
        else:
            detailed_metrics = {}
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def _test_epoch_enhanced(self) -> Dict[str, float]:
        """Enhanced test epoch with adaptation and transfer evaluation"""
        
        # Phase 1: Few-shot adaptation (using val_loader as adaptation data)
        adaptation_loss = self._perform_adaptation()
        
        # Phase 2: Transfer evaluation (using test_loader as target domain test)
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                inputs, targets, metadata = self._handle_batch_data(batch_data)
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass in test mode (uses adapted features if available)
                outputs = self.model(inputs, metadata, mode="test")
                traffic_output = outputs['traffic']
                
                # Handle output format
                _, targets = self._handle_output_format({'traffic': traffic_output}, targets)
                
                # Compute loss
                loss = self.traffic_criterion(traffic_output, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(traffic_output.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute detailed metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics.compute_all_metrics(
                all_outputs.numpy(), 
                all_targets.numpy()
            )
        else:
            detailed_metrics = {}
        
        metrics = {'loss': avg_loss, 'adaptation_loss': adaptation_loss}
        metrics.update(detailed_metrics)
        
        # Compute transfer score
        transfer_score = self._compute_transfer_score(metrics)
        metrics['transfer_score'] = transfer_score
        
        return metrics
    
    def _perform_adaptation(self) -> float:
        """Perform few-shot adaptation using validation data"""
        if not (hasattr(self.model, 'meta_adapter') and 
                self.model.meta_adapter is not None and 
                self.meta_optimizer is not None):
            return 0.0
        
        self.model.train()  # Enable training for adaptation
        
        adapt_loss = 0.0
        num_batches = 0
        max_adapt_batches = 5  # Limit adaptation to prevent overfitting
        
        for batch_data in self.val_loader:
            inputs, targets, metadata = self._handle_batch_data(batch_data)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Adaptation forward pass
            self.meta_optimizer.zero_grad()
            
            outputs = self.model(inputs, metadata, mode="adapt")
            
            # Handle output format
            if 'adapted_traffic' in outputs:
                traffic_output = outputs['adapted_traffic']
            else:
                traffic_output = outputs['traffic']
            
            _, targets = self._handle_output_format({'traffic': traffic_output}, targets)
            
            # Adaptation loss
            loss = self.traffic_criterion(traffic_output, targets)
            
            # Meta-learning update
            loss.backward()
            self.meta_optimizer.step()
            
            adapt_loss += loss.item()
            num_batches += 1
            
            # Limit adaptation steps
            if num_batches >= max_adapt_batches:
                break
        
        avg_adapt_loss = adapt_loss / num_batches if num_batches > 0 else 0.0
        logging.info(f"Few-shot adaptation completed. Loss: {avg_adapt_loss:.6f}")
        
        return avg_adapt_loss
    
    def _compute_transfer_score(self, test_metrics: Dict[str, float]) -> float:
        """Compute transfer learning score (lower is better)"""
        transfer_score = (
            test_metrics.get('mse', 0.0) * 0.4 +
            test_metrics.get('mae', 0.0) * 0.3 +
            test_metrics.get('volume_mse', 0.0) * 0.2 +
            test_metrics.get('speed_mse', 0.0) * 0.1
        )
        return transfer_score
    
    def fit(self) -> Dict[str, List[float]]:
        """Main training loop"""
        
        logging.info("Starting training...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model: {self.model.__class__.__name__}")
        logging.info(f"Experiment type: {self.experiment_type}")
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
            
            # Test
            test_metrics = self.test_epoch()
            
            # Update learning rate
            if self.is_enhanced_spatiotemporal and 'transfer_score' in test_metrics:
                monitor_metric = test_metrics['transfer_score']
            else:
                monitor_metric = val_metrics['loss']
            
            self.scheduler.step(monitor_metric)
            current_lr = self.main_optimizer.param_groups[0]['lr']
            
            # Update training history
            self._update_training_history(train_metrics, val_metrics, test_metrics, current_lr)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            self._log_epoch_results(epoch, train_metrics, val_metrics, test_metrics, current_lr, epoch_time)
            
            # TensorBoard logging
            self._log_tensorboard(epoch, train_metrics, val_metrics, test_metrics, current_lr)
            
            # Callbacks (checkpointing, early stopping)
            callback_metrics = test_metrics if self.is_enhanced_spatiotemporal else val_metrics
            should_stop = self.callbacks.on_epoch_end(
                epoch, callback_metrics, self.model, self.main_optimizer
            )
            
            # Update best metrics
            self._update_best_metrics(val_metrics, test_metrics)
            
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
    
    def _update_training_history(self, train_metrics, val_metrics, test_metrics, current_lr):
        """Update training history with current epoch metrics"""
        if self.is_enhanced_spatiotemporal:
            self.training_history['train_loss'].append(train_metrics['total'])
            
            # Update task-specific losses
            for task in ['traffic', 'city', 'year', 'enhanced_traffic']:
                if task in train_metrics:
                    self.training_history['task_losses'][task].append(train_metrics[task])
            
            if 'transfer_score' in test_metrics:
                self.training_history['transfer_score'].append(test_metrics['transfer_score'])
            
            if 'adaptation_loss' in test_metrics:
                self.training_history['adaptation_loss'].append(test_metrics['adaptation_loss'])
        else:
            self.training_history['train_loss'].append(train_metrics['loss'])
        
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['test_loss'].append(test_metrics['loss'])
        self.training_history['learning_rate'].append(current_lr)
    
    def _log_epoch_results(self, epoch, train_metrics, val_metrics, test_metrics, current_lr, epoch_time):
        """Log epoch results"""
        if self.is_enhanced_spatiotemporal:
            transfer_score = test_metrics.get('transfer_score', 0.0)
            adapt_loss = test_metrics.get('adaptation_loss', 0.0)
            
            logging.info(
                f'Epoch {epoch}: '
                f'Train: {train_metrics["total"]:.4f}, '
                f'Val: {val_metrics["loss"]:.4f}, '
                f'Test: {test_metrics["loss"]:.4f}, '
                f'Transfer: {transfer_score:.4f}, '
                f'Adapt: {adapt_loss:.4f}, '
                f'LR: {current_lr:.2e}, '
                f'Time: {epoch_time:.1f}s'
            )
            
            # Log task-specific losses
            task_logs = []
            for task in ['traffic', 'city', 'year']:
                if task in train_metrics:
                    task_logs.append(f'{task.capitalize()}: {train_metrics[task]:.4f}')
            if task_logs:
                logging.info(f'  Task Losses - {", ".join(task_logs)}')
        else:
            logging.info(
                f'Epoch {epoch}: '
                f'Train: {train_metrics["loss"]:.4f}, '
                f'Val: {val_metrics["loss"]:.4f}, '
                f'Test: {test_metrics["loss"]:.4f}, '
                f'LR: {current_lr:.2e}, '
                f'Time: {epoch_time:.1f}s'
            )
    
    def _log_tensorboard(self, epoch, train_metrics, val_metrics, test_metrics, current_lr):
        """Log metrics to TensorBoard"""
        if not self.writer:
            return
        
        # Basic metrics
        if self.is_enhanced_spatiotemporal:
            self.writer.add_scalar('Loss/Train_Total', train_metrics['total'], epoch)
            
            # Task-specific losses
            for task, loss in train_metrics.items():
                if task != 'total':
                    self.writer.add_scalar(f'Task_Loss/{task}', loss, epoch)
            
            # Transfer metrics
            if 'transfer_score' in test_metrics:
                self.writer.add_scalar('Transfer/Score', test_metrics['transfer_score'], epoch)
            if 'adaptation_loss' in test_metrics:
                self.writer.add_scalar('Transfer/Adaptation_Loss', test_metrics['adaptation_loss'], epoch)
        else:
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        
        self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
        self.writer.add_scalar('Loss/Test', test_metrics['loss'], epoch)
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Detailed test metrics
        for key, value in test_metrics.items():
            if key not in ['loss', 'transfer_score', 'adaptation_loss']:
                self.writer.add_scalar(f'Test_Metrics/{key}', value, epoch)
    
    def _update_best_metrics(self, val_metrics, test_metrics):
        """Update best metrics"""
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
        
        if (self.is_enhanced_spatiotemporal and 
            'transfer_score' in test_metrics and 
            test_metrics['transfer_score'] < self.best_transfer_score):
            self.best_transfer_score = test_metrics['transfer_score']
    
    def save_results(self):
        """Save training results and final model"""
        
        # Save training history
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Defensive: handle empty lists
        train_loss = self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None
        test_loss = self.training_history['test_loss'][-1] if self.training_history['test_loss'] else None
        
        # Prepare results summary
        results = {
            'experiment_type': self.experiment_type,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_loss,
            'final_test_loss': test_loss,
            'training_epochs': len(self.training_history['train_loss']),
            'config': {
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.training.batch_size,
                'model_features': self.config.model.features
            }
        }
        
        if self.is_enhanced_spatiotemporal:
            results.update({
                'best_transfer_score': self.best_transfer_score,
                'final_transfer_score': (self.training_history['transfer_score'][-1] 
                                       if self.training_history.get('transfer_score') and self.training_history['transfer_score'] else None),
                'experiment_config': {
                    'train_cities': self.config.experiment.train_cities,
                    'train_years': self.config.experiment.train_years,
                    'test_city': self.config.experiment.test_city,
                    'test_train_year': self.config.experiment.test_train_year,
                    'test_target_year': self.config.experiment.test_target_year,
                }
            })
        else:
            # Robustly handle train_city vs train_cities
            experiment_config = {}
            if hasattr(self.config.experiment, 'train_city'):
                experiment_config['train_city'] = self.config.experiment.train_city
            elif hasattr(self.config.experiment, 'train_cities'):
                experiment_config['train_cities'] = self.config.experiment.train_cities
            if hasattr(self.config.experiment, 'test_city'):
                experiment_config['test_city'] = self.config.experiment.test_city
            results.update({
                'final_transfer_score': None,
                'experiment_config': experiment_config
            })
        
        # Save results summary
        with open(self.save_dir / "results_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save final model
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'main_optimizer_state_dict': self.main_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'experiment_type': self.experiment_type
        }
        
        if self.is_enhanced_spatiotemporal:
            save_dict.update({
                'best_transfer_score': self.best_transfer_score,
                'meta_optimizer_state_dict': (self.meta_optimizer.state_dict() 
                                            if self.meta_optimizer else None),
                'loss_weights': self.loss_weights
            })
        
        torch.save(save_dict, self.save_dir / "final_model.pth")
        
        logging.info(f"Results saved to {self.save_dir}")
        
        # Log final summary
        self._log_final_summary()
    
    def _log_final_summary(self):
        """Log final training summary"""
        logging.info("\n" + "="*60)
        logging.info("TRAINING SUMMARY")
        logging.info("="*60)
        # Defensive: handle empty lists
        train_loss = self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None
        test_loss = self.training_history['test_loss'][-1] if self.training_history['test_loss'] else None
        best_val_loss = self.best_val_loss if hasattr(self, 'best_val_loss') else None
        if self.is_enhanced_spatiotemporal:
            logging.info(f"Enhanced Spatio-Temporal Transfer Results:")
            logging.info(f"  Best transfer score: {getattr(self, 'best_transfer_score', None):.6f}")
            logging.info(f"  Best validation loss: {best_val_loss:.6f}")
            logging.info(f"  Final test loss: {test_loss:.6f}" if test_loss is not None else "  Final test loss: N/A")
            if self.training_history.get('transfer_score') and self.training_history['transfer_score']:
                logging.info(f"  Final transfer score: {self.training_history['transfer_score'][-1]:.6f}")
            logging.info(f"Transfer Setup:")
            logging.info(f"  Training cities: {self.config.experiment.train_cities}")
            logging.info(f"  Training years: {self.config.experiment.train_years}")
            logging.info(f"  Test city: {self.config.experiment.test_city}")
            logging.info(f"  Transfer: {self.config.experiment.test_train_year} → {self.config.experiment.test_target_year}")
        else:
            logging.info(f"{self.experiment_type.replace('_', ' ').title()} Results:")
            logging.info(f"  Best validation loss: {best_val_loss:.6f}")
            logging.info(f"  Final training loss: {train_loss:.6f}" if train_loss is not None else "  Final training loss: N/A")
            logging.info(f"  Final test loss: {test_loss:.6f}" if test_loss is not None else "  Final test loss: N/A")
        logging.info(f"Training completed in {len(self.training_history['train_loss'])} epochs")
        logging.info("="*60)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if (self.meta_optimizer and 
            'meta_optimizer_state_dict' in checkpoint and 
            checkpoint['meta_optimizer_state_dict']):
            self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            
        if 'best_transfer_score' in checkpoint:
            self.best_transfer_score = checkpoint['best_transfer_score']
        
        if 'loss_weights' in checkpoint:
            self.loss_weights = checkpoint['loss_weights']
            
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Resume from the epoch where we left off
        self.current_epoch = len(self.training_history['train_loss'])
    
    def evaluate_model(self, data_loader: DataLoader, dataset_name: str = "Dataset") -> Dict[str, float]:
        """
        Evaluate model on a given dataset
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        logging.info(f"Evaluating on {dataset_name}...")
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                inputs, targets, metadata = self._handle_batch_data(batch_data)
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.is_enhanced_spatiotemporal:
                    outputs = self.model(inputs, metadata, mode="test")
                    traffic_output = outputs['traffic']
                else:
                    traffic_output = self.model(inputs)
                
                # Handle output format
                _, targets = self._handle_output_format({'traffic': traffic_output}, targets)
                
                # Compute loss
                loss = self.traffic_criterion(traffic_output, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Store for detailed metrics
                all_outputs.append(traffic_output.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Compute detailed metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics.compute_all_metrics(
                all_outputs.numpy(), 
                all_targets.numpy()
            )
        else:
            detailed_metrics = {}
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        # Log results
        logging.info(f"{dataset_name} Evaluation Results:")
        for key, value in metrics.items():
            logging.info(f"  {key}: {value:.6f}")
        
        return metrics
    
    def get_model_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions for analysis
        
        Args:
            data_loader: DataLoader for prediction
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        self.model.eval()
        
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                inputs, targets, metadata = self._handle_batch_data(batch_data)
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.is_enhanced_spatiotemporal:
                    outputs = self.model(inputs, metadata, mode="test")
                    traffic_output = outputs['traffic']
                else:
                    traffic_output = self.model(inputs)
                
                # Handle output format
                _, targets = self._handle_output_format({'traffic': traffic_output}, targets)
                
                # Store results
                all_outputs.append(traffic_output.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_outputs, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return predictions, targets

# Factory function to create trainer
def create_trainer(model, train_loader, val_loader, test_loader, config) -> Traffic4CastTrainer:
    """
    Factory function to create appropriate trainer
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration object
        
    Returns:
        Traffic4CastTrainer instance
    """
    return Traffic4CastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        test_loader=test_loader,
        config=config
    )
# src/training/enhanced_trainer.py
"""
Enhanced trainer with TQDM progress bars, comprehensive metrics, wandb integration, 
and improved checkpointing
"""

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
from tqdm import tqdm

# Import our enhanced components
from .metrics import Traffic4CastMetrics, MetricsTracker
from .callbacks import CheckpointManager
from utils.wandb_utils import WandbLogger

class EnhancedTraffic4CastTrainer:
    """
    Enhanced trainer with comprehensive metrics, TQDM progress bars, wandb integration,
    and improved checkpointing for Traffic4Cast experiments
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 config,
                 device: Optional[str] = None,
                 use_wandb: bool = True,
                 wandb_project: str = "traffic4cast"):
        """
        Initialize enhanced trainer
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader (or adaptation loader for enhanced spatio-temporal)
            config: Configuration object
            device: Device to use for training
            use_wandb: Whether to use wandb logging
            wandb_project: Wandb project name
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
        
        # Setup enhanced metrics
        self.metrics_calculator = Traffic4CastMetrics()
        self.metrics_tracker = MetricsTracker()
        
        # Setup logging and checkpointing
        self.setup_logging()
        
        # Setup enhanced checkpoint manager
        monitor_metric = 'competition_score' if self.is_enhanced_spatiotemporal else 'val_competition_score'
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir / "checkpoints",
            monitor_metric=monitor_metric,
            mode='min',
            save_top_k=3,
            save_last=True
        )
        
        # Setup wandb
        self.use_wandb = use_wandb
        self.wandb_logger = None
        if use_wandb:
            try:
                self.wandb_logger = WandbLogger(
                    config=config,
                    project_name=wandb_project,
                    experiment_name=config.logging.experiment_name
                )
                self.wandb_logger.watch_model(self.model, log_freq=50)
                self.wandb_logger.log_config_summary()
                
                # Log dataset info
                self.wandb_logger.log_dataset_info(
                    train_size=len(train_loader.dataset),
                    val_size=len(val_loader.dataset),
                    test_size=len(test_loader.dataset)
                )
                
            except Exception as e:
                logging.warning(f"Failed to initialize wandb: {e}")
                self.use_wandb = False
        
        # Initialize training state
        self._init_training_state()
        
        logging.info("Enhanced trainer initialized successfully")
        
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
        """Extract city and year labels from metadata"""
        if metadata is None:
            return None, None
        
        if isinstance(metadata, list):
            city_labels = torch.tensor([m['city_label'] for m in metadata]).to(self.device)
            year_labels = torch.tensor([m['year_label'] for m in metadata]).to(self.device)
            return city_labels, year_labels
        elif isinstance(metadata, dict):
            city_labels = metadata['city_label']
            year_labels = metadata['year_label']
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
        """Train for one epoch with TQDM progress bar"""
        self.model.train()
        
        if self.is_enhanced_spatiotemporal:
            return self._train_epoch_enhanced()
        else:
            return self._train_epoch_standard()
    
    def _train_epoch_standard(self) -> Dict[str, float]:
        """Standard training epoch with progress bar"""
        total_loss = 0.0
        accumulated_metrics = {}
        
        # Setup progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f"Train Epoch {self.current_epoch}",
            leave=False,
            ncols=120
        )
        
        for batch_idx, batch_data in enumerate(pbar):
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
            
            # Compute detailed metrics every few batches
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    batch_metrics = self.metrics_calculator.compute_competition_metrics(
                        outputs.detach(), targets.detach()
                    )
                    
                    # Accumulate metrics
                    for key, value in batch_metrics.items():
                        if key not in accumulated_metrics:
                            accumulated_metrics[key] = []
                        accumulated_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
            
            # Log to wandb
            if self.wandb_logger and batch_idx % 50 == 0:
                self.wandb_logger.log_metrics({
                    'train_batch_loss': loss.item(),
                    'train_lr': self.main_optimizer.param_groups[0]['lr']
                }, step=self.current_epoch * len(self.train_loader) + batch_idx, commit=False)
        
        # Calculate average metrics
        avg_metrics = {}
        for key, values in accumulated_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def _train_epoch_enhanced(self) -> Dict[str, float]:
        """Enhanced spatio-temporal training epoch with multi-task learning"""
        total_losses = {
            'total': 0.0,
            'traffic': 0.0,
            'city': 0.0,
            'year': 0.0,
            'enhanced_traffic': 0.0
        }
        accumulated_metrics = {}
        
        # Setup progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Enhanced Train Epoch {self.current_epoch}",
            leave=False,
            ncols=140
        )
        
        for batch_idx, batch_data in enumerate(pbar):
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
            
            # Compute detailed metrics every few batches
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    batch_metrics = self.metrics_calculator.compute_competition_metrics(
                        outputs['traffic'].detach(), targets.detach()
                    )
                    
                    # Accumulate metrics
                    for key, value in batch_metrics.items():
                        if key not in accumulated_metrics:
                            accumulated_metrics[key] = []
                        accumulated_metrics[key].append(value)
            
            # Update progress bar
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'traffic': f"{traffic_loss.item():.4f}",
                'city': f"{city_loss.item():.3f}",
                'year': f"{year_loss.item():.3f}"
            })
            
            # Log to wandb
            if self.wandb_logger and batch_idx % 50 == 0:
                self.wandb_logger.log_metrics({
                    'train_batch_total_loss': total_loss.item(),
                    'train_batch_traffic_loss': traffic_loss.item(),
                    'train_batch_city_loss': city_loss.item(),
                    'train_batch_year_loss': year_loss.item(),
                    'train_lr': self.main_optimizer.param_groups[0]['lr']
                }, step=self.current_epoch * len(self.train_loader) + batch_idx, commit=False)
        
        # Calculate average losses and metrics
        avg_losses = {k: v / len(self.train_loader) for k, v in total_losses.items()}
        
        # Add averaged detailed metrics
        for key, values in accumulated_metrics.items():
            avg_losses[key] = np.mean(values)
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate model on validation set with progress bar"""
        self.model.eval()
        total_loss = 0.0
        
        all_outputs = []
        all_targets = []
        
        # Setup progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Val Epoch {self.current_epoch}",
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
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
                
                # Store for detailed metrics
                all_outputs.append(traffic_output.cpu())
                all_targets.append(targets.cpu())
                
                # Update progress bar
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute comprehensive metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics_calculator.compute_competition_metrics(
                all_outputs, all_targets
            )
        else:
            detailed_metrics = {}
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def test_epoch(self) -> Dict[str, float]:
        """Evaluate model on test set with progress bar"""
        if self.is_enhanced_spatiotemporal:
            return self._test_epoch_enhanced()
        else:
            return self._test_epoch_standard()
    
    def _test_epoch_standard(self) -> Dict[str, float]:
        """Standard test epoch with progress bar"""
        self.model.eval()
        total_loss = 0.0
        
        all_outputs = []
        all_targets = []
        
        # Setup progress bar
        pbar = tqdm(
            self.test_loader,
            desc=f"Test Epoch {self.current_epoch}",
            leave=False,
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
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
                
                # Store for detailed metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                
                # Update progress bar
                pbar.set_postfix({'test_loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.test_loader)
        
        # Compute comprehensive metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics_calculator.compute_competition_metrics(
                all_outputs, all_targets
            )
        else:
            detailed_metrics = {}
        
        metrics = {'loss': avg_loss}
        metrics.update(detailed_metrics)
        
        return metrics
    
    def _test_epoch_enhanced(self) -> Dict[str, float]:
        """Enhanced test epoch with adaptation and transfer evaluation"""
        
        # Phase 1: Few-shot adaptation
        adaptation_loss = self._perform_adaptation()
        
        # Phase 2: Transfer evaluation
        self.model.eval()
        total_loss = 0.0
        
        all_outputs = []
        all_targets = []
        
        # Setup progress bar
        pbar = tqdm(
            self.test_loader,
            desc=f"Transfer Test Epoch {self.current_epoch}",
            leave=False,
            ncols=120
        )
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                inputs, targets, metadata = self._handle_batch_data(batch_data)
                
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass in test mode
                outputs = self.model(inputs, metadata, mode="test")
                traffic_output = outputs['traffic']
                
                # Handle output format
                _, targets = self._handle_output_format({'traffic': traffic_output}, targets)
                
                # Compute loss
                loss = self.traffic_criterion(traffic_output, targets)
                total_loss += loss.item()
                
                # Store for detailed metrics
                all_outputs.append(traffic_output.cpu())
                all_targets.append(targets.cpu())
                
                # Update progress bar
                pbar.set_postfix({
                    'transfer_loss': f"{loss.item():.4f}",
                    'adapt_loss': f"{adaptation_loss:.4f}"
                })
        
        avg_loss = total_loss / len(self.test_loader)
        
        # Compute comprehensive metrics
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            detailed_metrics = self.metrics_calculator.compute_competition_metrics(
                all_outputs, all_targets
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
        
        self.model.train()
        
        adapt_loss = 0.0
        num_batches = 0
        max_adapt_batches = 5
        
        # Setup progress bar for adaptation
        adapt_pbar = tqdm(
            self.val_loader,
            desc="Adaptation",
            leave=False,
            total=min(max_adapt_batches, len(self.val_loader)),
            ncols=100
        )
        
        for batch_data in adapt_pbar:
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
            
            # Update progress bar
            adapt_pbar.set_postfix({'adapt_loss': f"{loss.item():.4f}"})
            
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
        """Main training loop with enhanced progress tracking"""
        
        logging.info("Starting enhanced training with comprehensive metrics...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Model: {self.model.__class__.__name__}")
        logging.info(f"Experiment type: {self.experiment_type}")
        logging.info(f"Training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        # Main training progress bar
        epoch_pbar = tqdm(
            range(self.config.training.epochs),
            desc="Training Progress",
            ncols=150
        )
        
        start_time = time.time()
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Test
            test_metrics = self.test_epoch()
            
            # Update learning rate
            monitor_metric = val_metrics.get('competition_score', val_metrics['loss'])
            self.scheduler.step(monitor_metric)
            current_lr = self.main_optimizer.param_groups[0]['lr']
            
            # Update training history
            self._update_training_history(train_metrics, val_metrics, test_metrics, current_lr)
            
            # Track metrics
            self.metrics_tracker.update(epoch, 'train', train_metrics)
            self.metrics_tracker.update(epoch, 'val', val_metrics)
            self.metrics_tracker.update(epoch, 'test', test_metrics)
            
            # Enhanced checkpointing
            all_metrics = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
            
            is_best = self.checkpoint_manager.on_epoch_end(
                epoch, all_metrics, self.model, self.main_optimizer, 
                self.scheduler, {'config': self.config.__dict__}
            )
            
            # Wandb logging
            if self.wandb_logger:
                self._log_to_wandb(epoch, train_metrics, val_metrics, test_metrics, current_lr)
                
                # Log predictions occasionally
                if epoch % 10 == 0:
                    self._log_predictions_to_wandb(epoch)
                
                # Log system metrics
                self.wandb_logger.log_system_metrics()
            
            # Update epoch progress bar
            epoch_time = time.time() - epoch_start_time
            
            # Prepare postfix for progress bar
            postfix = {
                'train_loss': f"{train_metrics.get('loss', 0):.4f}",
                'val_loss': f"{val_metrics.get('loss', 0):.4f}",
                'test_loss': f"{test_metrics.get('loss', 0):.4f}",
                'lr': f"{current_lr:.2e}",
                'time': f"{epoch_time:.1f}s"
            }
            
            if self.is_enhanced_spatiotemporal:
                transfer_score = test_metrics.get('transfer_score', 0)
                postfix['transfer'] = f"{transfer_score:.4f}"
                
            epoch_pbar.set_postfix(postfix)
            
            # Update best metrics
            self._update_best_metrics(val_metrics, test_metrics)
            
            # Detailed logging every few epochs
            if epoch % 5 == 0 or is_best:
                self._log_detailed_metrics(epoch, train_metrics, val_metrics, test_metrics)
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.1f}s")
        
        # Save final results
        self.save_results()
        
        # Finish wandb
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        if self.writer:
            self.writer.close()
            
        return self.training_history
    
    def _update_training_history(self, train_metrics, val_metrics, test_metrics, current_lr):
        """Update training history with current epoch metrics"""
        if self.is_enhanced_spatiotemporal:
            self.training_history['train_loss'].append(train_metrics.get('total', train_metrics.get('loss', 0)))
            
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
    
    def _log_to_wandb(self, epoch, train_metrics, val_metrics, test_metrics, current_lr):
        """Log metrics to wandb"""
        # Aggregate all metrics for wandb
        wandb_metrics = {
            'epoch': epoch,
            'learning_rate': current_lr
        }
        
        # Add train metrics with prefix
        for key, value in train_metrics.items():
            wandb_metrics[f'train_{key}'] = value
        
        # Add val metrics with prefix
        for key, value in val_metrics.items():
            wandb_metrics[f'val_{key}'] = value
        
        # Add test metrics with prefix
        for key, value in test_metrics.items():
            wandb_metrics[f'test_{key}'] = value
        
        self.wandb_logger.log_metrics(wandb_metrics, step=epoch, commit=True)
    
    def _log_predictions_to_wandb(self, epoch):
        """Log model predictions to wandb"""
        try:
            # Get a batch from validation set
            batch_data = next(iter(self.val_loader))
            inputs, targets, metadata = self._handle_batch_data(batch_data)
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                if self.is_enhanced_spatiotemporal:
                    outputs = self.model(inputs, metadata, mode="test")
                    predictions = outputs['traffic']
                else:
                    predictions = self.model(inputs)
            
            # Handle output format
            predictions, targets = self._handle_output_format({'traffic': predictions}, targets)
            
            # Convert to numpy for logging
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            
            self.wandb_logger.log_model_predictions(pred_np, target_np, epoch, max_samples=2)
            
        except Exception as e:
            logging.warning(f"Failed to log predictions to wandb: {e}")
    
    def _log_detailed_metrics(self, epoch, train_metrics, val_metrics, test_metrics):
        """Log detailed metrics"""
        logging.info(f"\n{'='*80}")
        logging.info(f"EPOCH {epoch} DETAILED METRICS")
        logging.info(f"{'='*80}")
        
        # Training metrics
        logging.info("Training Metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {key}: {value:.6f}")
        
        # Validation metrics
        logging.info("Validation Metrics:")
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {key}: {value:.6f}")
        
        # Test metrics
        logging.info("Test Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f"  {key}: {value:.6f}")
        
        logging.info(f"{'='*80}")
    
    def _update_best_metrics(self, val_metrics, test_metrics):
        """Update best metrics"""
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
        
        if (self.is_enhanced_spatiotemporal and 
            'transfer_score' in test_metrics and 
            test_metrics['transfer_score'] < self.best_transfer_score):
            self.best_transfer_score = test_metrics['transfer_score']
    
    def save_results(self):
        """Save comprehensive training results"""
        
        # Get checkpoint info
        checkpoint_info = self.checkpoint_manager.get_best_checkpoint_info()
        
        # Prepare final results
        final_metrics = {
            'train': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
            'val': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'test': self.training_history['test_loss'][-1] if self.training_history['test_loss'] else None
        }
        
        # Save checkpoint manager summary
        self.checkpoint_manager.save_training_summary(final_metrics)
        
        # Save training history
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save metrics tracker
        with open(self.save_dir / "metrics_tracker.json", 'w') as f:
            json.dump(self.metrics_tracker.to_dict(), f, indent=2)
        
        logging.info(f"Enhanced results saved to {self.save_dir}")


# Factory function
def create_enhanced_trainer(model, train_loader, val_loader, test_loader, config, 
                           use_wandb=True, wandb_project="traffic4cast"):
    """Factory function to create enhanced trainer"""
    return EnhancedTraffic4CastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )
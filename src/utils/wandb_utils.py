# src/utils/wandb_utils.py
"""
Weights & Biases integration for Traffic4Cast experiments
"""

import wandb
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class WandbLogger:
    """
    Wandb logger for Traffic4Cast experiments with comprehensive tracking
    """
    
    def __init__(self, config, project_name: str = "traffic4cast", 
                 entity: Optional[str] = None, experiment_name: Optional[str] = None):
        """
        Initialize Wandb logger
        
        Args:
            config: Configuration object
            project_name: Wandb project name
            entity: Wandb entity (optional)
            experiment_name: Custom experiment name
        """
        self.config = config
        self.project_name = project_name
        self.entity = entity
        self.experiment_name = experiment_name or config.logging.experiment_name
        
        # Initialize wandb
        self.run = None
        self.is_enabled = True
        
        try:
            self._init_wandb()
            logging.info(f"Wandb initialized: {self.run.url}")
        except Exception as e:
            logging.warning(f"Failed to initialize wandb: {e}")
            self.is_enabled = False
    
    def _init_wandb(self):
        """Initialize wandb run"""
        
        # Convert config to dict for wandb
        config_dict = self._config_to_dict()
        
        # Add system info
        config_dict.update({
            'device': str(self.config.training.device),
            'pytorch_version': torch.__version__,
        })
        
        # Initialize run
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=self.experiment_name,
            config=config_dict,
            reinit=True,
            tags=self._generate_tags()
        )
        
        # Watch model if available
        # (Will be called separately in trainer)
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Watch model for gradient and parameter tracking"""
        if self.is_enabled and self.run:
            try:
                wandb.watch(model, log="all", log_freq=log_freq)
                logging.info("Model watching enabled in wandb")
            except Exception as e:
                logging.warning(f"Failed to watch model: {e}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, 
                   prefix: str = "", commit: bool = True):
        """
        Log metrics to wandb
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (epoch)
            prefix: Prefix for metric names
            commit: Whether to commit the metrics
        """
        if not self.is_enabled or not self.run:
            return
            
        try:
            # Add prefix to metric names
            log_dict = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.cpu().numpy()
                    
                    log_key = f"{prefix}/{key}" if prefix else key
                    log_dict[log_key] = value
            
            # Add step if provided
            if step is not None:
                log_dict['epoch'] = step
            
            wandb.log(log_dict, step=step, commit=commit)
            
        except Exception as e:
            logging.warning(f"Failed to log metrics to wandb: {e}")
    
    def log_model_predictions(self, predictions: np.ndarray, targets: np.ndarray, 
                            step: int, max_samples: int = 4):
        """
        Log model predictions as images
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            step: Current step/epoch
            max_samples: Maximum number of samples to log
        """
        if not self.is_enabled or not self.run:
            return
            
        try:
            # Select random samples
            n_samples = min(max_samples, predictions.shape[0])
            indices = np.random.choice(predictions.shape[0], n_samples, replace=False)
            
            images = []
            
            for i, idx in enumerate(indices):
                pred = predictions[idx]
                target = targets[idx]
                
                # Create comparison image
                fig = self._create_prediction_comparison(pred, target, idx)
                
                # Convert to wandb image
                images.append(wandb.Image(fig, caption=f"Sample {idx}"))
                plt.close(fig)
            
            # Log images
            wandb.log({
                "predictions": images,
                "epoch": step
            }, step=step)
            
        except Exception as e:
            logging.warning(f"Failed to log predictions: {e}")
    
    def log_learning_curves(self, train_metrics: List[float], val_metrics: List[float], 
                           metric_name: str = "loss"):
        """Log learning curves"""
        if not self.is_enabled or not self.run:
            return
            
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(train_metrics) + 1)
            ax.plot(epochs, train_metrics, label=f'Train {metric_name}', color='blue')
            ax.plot(epochs, val_metrics, label=f'Val {metric_name}', color='red')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.capitalize())
            ax.set_title(f'{metric_name.capitalize()} Learning Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            wandb.log({f"learning_curves/{metric_name}": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            logging.warning(f"Failed to log learning curves: {e}")
    
    def log_model_checkpoint(self, model_path: str, step: int, is_best: bool = False):
        """Log model checkpoint to wandb"""
        if not self.is_enabled or not self.run:
            return
            
        try:
            artifact_name = f"model-{self.experiment_name}"
            if is_best:
                artifact_name += "-best"
            
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Model checkpoint at epoch {step}"
            )
            
            artifact.add_file(model_path)
            
            wandb.log_artifact(artifact)
            logging.info(f"Model checkpoint logged to wandb: {artifact_name}")
            
        except Exception as e:
            logging.warning(f"Failed to log model checkpoint: {e}")
    
    def log_config_summary(self):
        """Log configuration summary table"""
        if not self.is_enabled or not self.run:
            return
            
        try:
            config_dict = self._config_to_dict()
            
            # Create summary table
            table_data = []
            for section, params in config_dict.items():
                if isinstance(params, dict):
                    for key, value in params.items():
                        table_data.append([section, key, str(value)])
                else:
                    table_data.append(["general", section, str(params)])
            
            table = wandb.Table(
                columns=["Section", "Parameter", "Value"],
                data=table_data
            )
            
            wandb.log({"config_summary": table})
            
        except Exception as e:
            logging.warning(f"Failed to log config summary: {e}")
    
    def log_dataset_info(self, train_size: int, val_size: int, test_size: int,
                        dataset_distribution: Optional[Dict] = None):
        """Log dataset information"""
        if not self.is_enabled or not self.run:
            return
            
        try:
            dataset_info = {
                "dataset/train_size": train_size,
                "dataset/val_size": val_size,
                "dataset/test_size": test_size,
                "dataset/total_size": train_size + val_size + test_size
            }
            
            if dataset_distribution:
                for key, value in dataset_distribution.items():
                    dataset_info[f"dataset/distribution_{key}"] = value
            
            wandb.log(dataset_info)
            
        except Exception as e:
            logging.warning(f"Failed to log dataset info: {e}")
    
    def log_system_metrics(self):
        """Log system metrics (GPU, memory, etc.)"""
        if not self.is_enabled or not self.run:
            return
            
        try:
            import psutil
            
            # CPU and memory
            system_metrics = {
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent,
                "system/memory_available_gb": psutil.virtual_memory().available / (1024**3)
            }
            
            # GPU metrics if available
            if torch.cuda.is_available():
                system_metrics.update({
                    "system/gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),
                    "system/gpu_memory_reserved": torch.cuda.memory_reserved() / (1024**3)
                })
            
            wandb.log(system_metrics)
            
        except Exception as e:
            logging.warning(f"Failed to log system metrics: {e}")
    
    def finish(self, exit_code: int = 0):
        """Finish wandb run"""
        if self.is_enabled and self.run:
            try:
                wandb.finish(exit_code=exit_code)
                logging.info("Wandb run finished")
            except Exception as e:
                logging.warning(f"Error finishing wandb run: {e}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        config_dict = {}
        
        if hasattr(self.config, 'model'):
            config_dict['model'] = self.config.model.__dict__
        if hasattr(self.config, 'training'):
            config_dict['training'] = self.config.training.__dict__
        if hasattr(self.config, 'data'):
            config_dict['data'] = self.config.data.__dict__
        if hasattr(self.config, 'experiment'):
            config_dict['experiment'] = self.config.experiment.__dict__
        if hasattr(self.config, 'logging'):
            config_dict['logging'] = self.config.logging.__dict__
        
        return config_dict
    
    def _generate_tags(self) -> List[str]:
        """Generate tags for the wandb run"""
        tags = ["traffic4cast"]
        
        if hasattr(self.config, 'experiment'):
            experiment_type = getattr(self.config.experiment, 'type', None)
            if experiment_type:
                tags.append(experiment_type)
        
        if hasattr(self.config, 'model'):
            model_name = getattr(self.config.model, 'name', None)
            if model_name:
                tags.append(model_name)
        
        return tags
    
    def _create_prediction_comparison(self, pred: np.ndarray, target: np.ndarray, 
                                    sample_idx: int) -> plt.Figure:
        """Create comparison visualization of predictions vs targets"""
        
        # Handle different tensor shapes
        if pred.ndim == 3:  # (C, H, W)
            # Take first channel (volume)
            pred_vis = pred[0] if pred.shape[0] > 0 else pred
            target_vis = target[0] if target.shape[0] > 0 else target
        elif pred.ndim == 4:  # (T, C, H, W) or (T, H, W, C)
            if pred.shape[-1] == 8:  # (T, H, W, C)
                pred_vis = pred[0, :, :, 0]  # First timestep, first channel
                target_vis = target[0, :, :, 0]
            else:  # (T, C, H, W)
                pred_vis = pred[0, 0]  # First timestep, first channel
                target_vis = target[0, 0]
        else:
            # Fallback: take mean
            pred_vis = np.mean(pred, axis=tuple(range(pred.ndim-2)))
            target_vis = np.mean(target, axis=tuple(range(target.ndim-2)))
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target
        im1 = axes[0].imshow(target_vis, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Ground Truth (Sample {sample_idx})')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Prediction
        im2 = axes[1].imshow(pred_vis, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Prediction (Sample {sample_idx})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Error
        error = np.abs(pred_vis - target_vis)
        im3 = axes[2].imshow(error, cmap='Reds', aspect='auto')
        axes[2].set_title(f'Absolute Error (Sample {sample_idx})')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig
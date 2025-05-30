# src/experiments/spatiotemporal_transfer.py
"""
Spatio-Temporal Transfer Learning Experiment Runner
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import numpy as np

from ..utils.config import Config
from ..data.splitters import ExperimentDataManager
from ..data.dataset import create_data_loaders, get_dataset_stats
from ..models.unet import create_unet_model
from ..training.trainer import create_trainer

class SpatioTemporalTransferExperiment:
    """Handles spatio-temporal transfer learning experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate experiment configuration"""
        if self.config.experiment.type != "spatiotemporal_transfer":
            raise ValueError("Config must be for spatiotemporal_transfer experiment")
            
        if not self.config.experiment.train_years:
            raise ValueError("train_years must be specified")
            
        if not self.config.experiment.test_years:
            raise ValueError("test_years must be specified")
    
    def setup_data(self) -> Tuple[Any, Any, Any]:
        """Setup data for spatio-temporal transfer experiment"""
        logging.info("Setting up spatio-temporal transfer data...")
        
        data_manager = ExperimentDataManager(self.config)
        train_dataset, val_dataset, test_dataset = data_manager.setup_spatiotemporal_transfer()
        
        # Log dataset statistics
        train_stats = get_dataset_stats(train_dataset)
        val_stats = get_dataset_stats(val_dataset)
        test_stats = get_dataset_stats(test_dataset)
        
        logging.info("Spatio-Temporal Transfer Data Setup:")
        logging.info(f"  Train years: {self.config.experiment.train_years}")
        logging.info(f"  Test years: {self.config.experiment.test_years}")
        logging.info(f"  Cities: {self.config.data.cities}")
        logging.info(f"  Train samples: {train_stats['total_samples']} from {train_stats['year_counts']}")
        logging.info(f"  Val samples: {val_stats['total_samples']} from {val_stats['year_counts']}")
        logging.info(f"  Test samples: {test_stats['total_samples']} from {test_stats['year_counts']}")
        
        return train_dataset, val_dataset, test_dataset
    
    def run_domain_adaptation_analysis(self) -> Dict[str, Any]:
        """Analyze domain shift between train and test years"""
        logging.info("Analyzing domain shift...")
        
        # Setup data
        train_dataset, val_dataset, test_dataset = self.setup_data()
        
        # Sample some data for analysis
        def sample_dataset_statistics(dataset, num_samples=100):
            """Sample statistics from dataset"""
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
            
            volume_stats = []
            speed_stats = []
            
            for idx in indices:
                inputs, targets = dataset[idx]
                
                # Convert to numpy if tensor
                if torch.is_tensor(inputs):
                    inputs = inputs.numpy()
                if torch.is_tensor(targets):
                    targets = targets.numpy()
                
                # Extract volume and speed channels
                if inputs.ndim == 4:  # (T, H, W, C)
                    volume_data = inputs[:, :, :, [0, 2, 4, 6]]  # Volume channels
                    speed_data = inputs[:, :, :, [1, 3, 5, 7]]   # Speed channels
                else:  # Handle other formats
                    continue
                
                volume_stats.append({
                    'mean': np.mean(volume_data),
                    'std': np.std(volume_data),
                    'max': np.max(volume_data),
                    'sparsity': np.mean(volume_data == 0)
                })
                
                speed_stats.append({
                    'mean': np.mean(speed_data[volume_data > 0]) if np.any(volume_data > 0) else 0,
                    'std': np.std(speed_data[volume_data > 0]) if np.any(volume_data > 0) else 0,
                    'max': np.max(speed_data)
                })
            
            return {
                'volume': {k: np.mean([s[k] for s in volume_stats]) for k in volume_stats[0].keys()},
                'speed': {k: np.mean([s[k] for s in speed_stats]) for k in speed_stats[0].keys()}
            }
        
        train_stats = sample_dataset_statistics(train_dataset)
        test_stats = sample_dataset_statistics(test_dataset)
        
        # Calculate domain shift metrics
        domain_shift = {
            'volume_mean_shift': abs(train_stats['volume']['mean'] - test_stats['volume']['mean']),
            'volume_sparsity_shift': abs(train_stats['volume']['sparsity'] - test_stats['volume']['sparsity']),
            'speed_mean_shift': abs(train_stats['speed']['mean'] - test_stats['speed']['mean']),
            'train_stats': train_stats,
            'test_stats': test_stats
        }
        
        logging.info("Domain Shift Analysis:")
        logging.info(f"  Volume mean shift: {domain_shift['volume_mean_shift']:.4f}")
        logging.info(f"  Volume sparsity shift: {domain_shift['volume_sparsity_shift']:.4f}")
        logging.info(f"  Speed mean shift: {domain_shift['speed_mean_shift']:.4f}")
        
        return domain_shift
    
    def run_progressive_transfer(self) -> Dict[str, Any]:
        """Run progressive transfer: train on increasing amounts of target domain data"""
        
        # First, train on source domain only
        logging.info("Step 1: Training on source domain only...")
        source_results = self.run_single_experiment()
        
        # Then, fine-tune on increasing amounts of target domain data
        target_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        progressive_results = []
        
        for fraction in target_fractions:
            logging.info(f"Step 2: Fine-tuning on {fraction*100:.0f}% of target domain...")
            
            # Modify config to include some target domain data in training
            modified_config = self.config
            modified_config.experiment.target_fraction = fraction
            
            # TODO: Implement mixed training with source + target data
            # This would require modifying the data splitter
            
            # For now, just log the intent
            logging.info(f"Would fine-tune with {fraction*100:.0f}% target data")
        
        return {
            'source_only_results': source_results,
            'progressive_results': progressive_results
        }
    
    def run_single_experiment(self) -> Dict[str, Any]:
        """Run a single spatio-temporal transfer experiment"""
        
        # Setup data
        train_dataset, val_dataset, test_dataset = self.setup_data()
        train_loader, val_loader, test_loader = create_data_loaders(
            self.config, train_dataset, val_dataset, test_dataset
        )
        
        # Create model
        model = create_unet_model(self.config)
        
        # Create trainer
        trainer = create_trainer(model, train_loader, val_loader, test_loader, self.config)
        
        # Train model
        training_history = trainer.fit()
        
        # Collect results
        results = {
            'train_years': self.config.experiment.train_years,
            'test_years': self.config.experiment.test_years,
            'cities': self.config.data.cities,
            'best_val_loss': trainer.best_val_loss,
            'final_train_loss': training_history['train_loss'][-1],
            'final_test_loss': training_history['test_loss'][-1],
            'training_history': training_history,
            'model_path': str(trainer.save_dir / "final_model.pth"),
            'experiment_dir': str(trainer.save_dir)
        }
        
        return results
    
    def run(self) -> Dict[str, Any]:
        """Run the spatio-temporal transfer experiment"""
        logging.info("Starting spatio-temporal transfer learning experiment...")
        
        # Run domain analysis
        domain_analysis = self.run_domain_adaptation_analysis()
        
        # Run main experiment
        main_results = self.run_single_experiment()
        
        # Combine results
        results = {
            'experiment_type': 'spatiotemporal_transfer',
            'domain_analysis': domain_analysis,
            'main_results': main_results
        }
        
        return results
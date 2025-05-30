# src/experiments/spatial_transfer.py
"""
Spatial Transfer Learning Experiment Runner
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import numpy as np

from ..utils.config import Config
from ..data.splitter import ExperimentDataManager
from ..data.dataset import create_data_loaders, get_dataset_stats
from ..models.unet import create_unet_model
from ..training.trainer import create_trainer

class SpatialTransferExperiment:
    """Handles spatial transfer learning experiments"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate experiment configuration"""
        if self.config.experiment.type != "spatial_transfer":
            raise ValueError("Config must be for spatial_transfer experiment")
            
        if not self.config.experiment.train_cities:
            raise ValueError("train_cities must be specified")
            
        if not self.config.experiment.test_cities:
            raise ValueError("test_cities must be specified")
            
        # Check for overlap
        train_set = set(self.config.experiment.train_cities)
        test_set = set(self.config.experiment.test_cities)
        
        if train_set & test_set:
            overlap = train_set & test_set
            raise ValueError(f"Cities cannot be in both train and test: {overlap}")
    
    def setup_data(self) -> Tuple[Any, Any, Any]:
        """Setup data for spatial transfer experiment"""
        logging.info("Setting up spatial transfer data...")
        
        data_manager = ExperimentDataManager(self.config)
        train_dataset, val_dataset, test_dataset = data_manager.setup_spatial_transfer()
        
        # Log dataset statistics
        train_stats = get_dataset_stats(train_dataset)
        val_stats = get_dataset_stats(val_dataset)
        test_stats = get_dataset_stats(test_dataset)
        
        logging.info("Spatial Transfer Data Setup:")
        logging.info(f"  Train cities: {self.config.experiment.train_cities}")
        logging.info(f"  Test cities: {self.config.experiment.test_cities}")
        logging.info(f"  Train samples: {train_stats['total_samples']} from {train_stats['city_counts']}")
        logging.info(f"  Val samples: {val_stats['total_samples']} from {val_stats['city_counts']}")
        logging.info(f"  Test samples: {test_stats['total_samples']} from {test_stats['city_counts']}")
        
        return train_dataset, val_dataset, test_dataset
    
    def run_single_experiment(self, train_cities: List[str], test_cities: List[str]) -> Dict[str, Any]:
        """Run a single spatial transfer experiment"""
        
        # Update config for this experiment
        self.config.experiment.train_cities = train_cities
        self.config.experiment.test_cities = test_cities
        
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
            'train_cities': train_cities,
            'test_cities': test_cities,
            'best_val_loss': trainer.best_val_loss,
            'final_train_loss': training_history['train_loss'][-1],
            'final_test_loss': training_history['test_loss'][-1],
            'training_history': training_history,
            'model_path': str(trainer.save_dir / "final_model.pth"),
            'experiment_dir': str(trainer.save_dir)
        }
        
        return results
    
    def run_cross_validation(self) -> Dict[str, Any]:
        """Run cross-validation across different city splits"""
        
        all_cities = self.config.data.cities
        results = []
        
        # Define different train/test splits
        splits = [
            # Split 1: Use provided config
            (self.config.experiment.train_cities, self.config.experiment.test_cities),
            
            # Split 2: Reverse the split
            (self.config.experiment.test_cities, self.config.experiment.train_cities),
            
            # Additional splits if enough cities
            # Add more strategic splits here
        ]
        
        for i, (train_cities, test_cities) in enumerate(splits):
            logging.info(f"\n{'='*50}")
            logging.info(f"CROSS-VALIDATION FOLD {i+1}/{len(splits)}")
            logging.info(f"Train: {train_cities}")
            logging.info(f"Test: {test_cities}")
            logging.info(f"{'='*50}")
            
            try:
                fold_results = self.run_single_experiment(train_cities, test_cities)
                fold_results['fold'] = i + 1
                results.append(fold_results)
                
            except Exception as e:
                logging.error(f"Fold {i+1} failed: {e}")
                continue
        
        # Aggregate results
        if results:
            avg_val_loss = np.mean([r['best_val_loss'] for r in results])
            avg_test_loss = np.mean([r['final_test_loss'] for r in results])
            std_val_loss = np.std([r['best_val_loss'] for r in results])
            std_test_loss = np.std([r['final_test_loss'] for r in results])
            
            summary = {
                'experiment_type': 'spatial_transfer_cv',
                'num_folds': len(results),
                'avg_val_loss': avg_val_loss,
                'std_val_loss': std_val_loss,
                'avg_test_loss': avg_test_loss,
                'std_test_loss': std_test_loss,
                'fold_results': results
            }
            
            logging.info(f"\nCross-Validation Summary:")
            logging.info(f"  Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
            logging.info(f"  Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
            
            return summary
        else:
            raise RuntimeError("All cross-validation folds failed")
    
    def run(self) -> Dict[str, Any]:
        """Run the spatial transfer experiment"""
        logging.info("Starting spatial transfer learning experiment...")
        
        # Run single experiment or cross-validation
        if hasattr(self.config.experiment, 'cross_validation') and self.config.experiment.cross_validation:
            return self.run_cross_validation()
        else:
            return self.run_single_experiment(
                self.config.experiment.train_cities,
                self.config.experiment.test_cities
            )
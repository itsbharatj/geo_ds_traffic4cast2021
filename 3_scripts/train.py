# scripts/train.py
"""
Main training script for Traffic4Cast experiments

Usage:
    python scripts/train.py --config config/spatial_transfer.yaml
    python scripts/train.py --config config/spatiotemporal_transfer.yaml
    python scripts/train.py --config config/debug.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.reproducibility import set_random_seeds
from data.splitters import ExperimentDataManager
from data.dataset import create_data_loaders
from models.unet import create_unet_model
from training.trainer import create_trainer

def main():
    parser = argparse.ArgumentParser(description="Traffic4Cast Training")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--experiment-name", help="Override experiment name")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.device:
        config.training.device = args.device
    if args.debug:
        config.data.limit_per_split = 50
        config.training.epochs = 2
        config.training.batch_size = 2
        config.logging.log_interval = 5
    
    # Setup logging
    setup_logging()
    
    # Set random seeds for reproducibility
    set_random_seeds(config.experiment.random_seed)
    
    # Log configuration
    logging.info("=" * 60)
    logging.info("TRAFFIC4CAST EXPERIMENT")
    logging.info("=" * 60)
    logging.info(f"Experiment type: {config.experiment.type}")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Device: {config.training.device}")
    logging.info(f"Random seed: {config.experiment.random_seed}")
    
    # Setup data
    logging.info("Setting up data...")
    data_manager = ExperimentDataManager(config)
    train_dataset, val_dataset, test_dataset = data_manager.setup_experiment_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config, train_dataset, val_dataset, test_dataset
    )
    
    logging.info(f"Data loaded:")
    logging.info(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    logging.info(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    logging.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    # Create model
    logging.info("Creating model...")
    model = create_unet_model(config)
    model_info = model.get_model_info()
    
    logging.info(f"Model created:")
    logging.info(f"  Architecture: {model_info['model_name']}")
    logging.info(f"  Parameters: {model_info['trainable_parameters']:,}")
    logging.info(f"  Model size: {model_info['model_size_mb']:.1f} MB")
    logging.info(f"  Input channels: {model_info['in_channels']}")
    logging.info(f"  Output channels: {model_info['out_channels']}")
    
    # Create trainer
    logging.info("Setting up trainer...")
    trainer = create_trainer(model, train_loader, val_loader, test_loader, config)
    
    # Start training
    logging.info("Starting training...")
    try:
        training_history = trainer.fit()
        
        # Print final results
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETED")
        logging.info("=" * 60)
        logging.info(f"Best validation loss: {trainer.best_val_loss:.6f}")
        logging.info(f"Final training loss: {training_history['train_loss'][-1]:.6f}")
        logging.info(f"Final test loss: {training_history['test_loss'][-1]:.6f}")
        logging.info(f"Results saved to: {trainer.save_dir}")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        trainer.save_results()
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
# scripts/train.py
"""
Main training script for Traffic4Cast experiments including enhanced spatio-temporal transfer

Usage:
    python scripts/train.py --config config/spatial_config.yaml
    python scripts/train.py --config config/spatiotemporal_config.yaml
    python scripts/train.py --config config/debug.yaml --debug
    python scripts/train.py --config config/enhanced_spatiotemporal.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

import torch
import numpy as np
from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.reproducibility import set_random_seeds
from data.splitter import ExperimentDataManager
from data.dataset import create_data_loaders
from models.unet import create_unet_model
from models.multitask_unet import create_multitask_unet
from training.trainer import create_trainer

def create_model_from_config(config):
    """Create appropriate model based on configuration"""
    
    if config.experiment.type == "enhanced_spatiotemporal_transfer":
        # Use multi-task UNet for enhanced spatio-temporal transfer
        return create_multitask_unet(config)
    else:
        # Use standard UNet for other experiments
        return create_unet_model(config)

def main():
    parser = argparse.ArgumentParser(description="Traffic4Cast Training")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--experiment-name", help="Override experiment name")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--test-city", help="Override test city (for spatio-temporal)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.device:
        config.training.device = args.device
    if args.test_city:
        # Handle different experiment types
        if hasattr(config.experiment, 'test_cities'):
            config.experiment.test_cities = [args.test_city]
        if hasattr(config.experiment, 'test_city'):
            config.experiment.test_city = args.test_city
    if args.debug:
        config.data.limit_per_split = 50
        config.training.epochs = 3
        config.training.batch_size = 2
        config.logging.log_interval = 5
        # Reduce adaptation samples for debug
        if hasattr(config.experiment, 'adaptation_samples'):
            config.experiment.adaptation_samples = 10
        
    # Force num_workers=0 for macOS/conda compatibility [REMOVE ON THE SERVER]
    if config.training.num_workers is None:
        config.training.num_workers = 0
    
    # Setup logging
    setup_logging()
    
    # Set random seeds for reproducibility
    set_random_seeds(config.experiment.random_seed)
    
    # Log configuration
    logging.info("=" * 80)
    logging.info("TRAFFIC4CAST EXPERIMENT")
    logging.info("=" * 80)
    logging.info(f"Experiment type: {config.experiment.type}")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Device: {config.training.device}")
    logging.info(f"Random seed: {config.experiment.random_seed}")
    
    # Log experiment-specific details
    if config.experiment.type == "spatial_transfer":
        logging.info(f"Train cities: {config.experiment.train_cities}")
        logging.info(f"Test cities: {config.experiment.test_cities}")
        
    elif config.experiment.type == "spatiotemporal_transfer":
        logging.info(f"Cities: {config.data.cities}")
        logging.info(f"Train years: {config.experiment.train_years}")
        logging.info(f"Test years: {config.experiment.test_years}")
        
    elif config.experiment.type == "enhanced_spatiotemporal_transfer":
        logging.info("🚀 Enhanced Spatio-Temporal Transfer Learning Mode")
        logging.info(f"Train cities: {config.experiment.train_cities}")
        logging.info(f"Train years: {config.experiment.train_years}")
        logging.info(f"Test city: {config.experiment.test_city}")
        logging.info(f"Test source year: {config.experiment.test_train_year}")
        logging.info(f"Test target year: {config.experiment.test_target_year}")
        logging.info(f"Adaptation samples: {getattr(config.experiment, 'adaptation_samples', 100)}")
        
        # Validate spatio-temporal setup
        if config.experiment.test_city in config.experiment.train_cities:
            raise ValueError(
                f"❌ Test city '{config.experiment.test_city}' cannot be in training cities "
                f"for spatio-temporal transfer learning. This violates the transfer learning setup!"
            )
        
        logging.info("✅ Spatio-temporal transfer setup validated")
    
    # Setup data
    logging.info("Setting up data...")
    data_manager = ExperimentDataManager(config)
    
    if config.experiment.type == "enhanced_spatiotemporal_transfer":
        train_dataset, adapt_dataset, test_dataset = data_manager.setup_experiment_data()
        
        # Create data loaders with the adaptation dataset as validation
        train_loader, val_loader, test_loader = create_data_loaders(
            config, train_dataset, adapt_dataset, test_dataset
        )
        
        logging.info(f"Enhanced spatio-temporal data loaded:")
        logging.info(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
        logging.info(f"  Adaptation: {len(adapt_dataset)} samples, {len(val_loader)} batches")
        logging.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
        
        # Update config with dataset information for multi-task learning
        if hasattr(train_dataset, 'get_num_cities'):
            config.experiment.num_cities = train_dataset.get_num_cities()
            config.experiment.num_years = train_dataset.get_num_years()
            logging.info(f"  Multi-task setup: {config.experiment.num_cities} cities, {config.experiment.num_years} years")
    else:
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
    model = create_model_from_config(config)
    
    # Log model information
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        logging.info(f"Model created:")
        logging.info(f"  Architecture: {model_info['model_name']}")
        logging.info(f"  Parameters: {model_info['trainable_parameters']:,}")
        logging.info(f"  Model size: {model_info['model_size_mb']:.1f} MB")
        logging.info(f"  Input channels: {model_info['in_channels']}")
        logging.info(f"  Output channels: {model_info['out_channels']}")
        
        if config.experiment.type == "enhanced_spatiotemporal_transfer":
            logging.info(f"  Multi-task heads: Traffic + City + Year classification")
            logging.info(f"  Use attention: {model_info.get('use_attention', False)}")
            logging.info(f"  Use meta-learning: {model_info.get('use_meta_learning', False)}")
    
    # Create trainer
    logging.info("Setting up trainer...")
    trainer = create_trainer(model, train_loader, val_loader, test_loader, config)
    
    # Start training
    logging.info("Starting training...")
    try:
        training_history = trainer.fit()
        
        # Print final results
        logging.info("=" * 80)
        logging.info("TRAINING COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)
        
        if config.experiment.type == "enhanced_spatiotemporal_transfer":
            logging.info(f"🎯 Enhanced Spatio-Temporal Transfer Results:")
            if hasattr(trainer, 'best_transfer_score'):
                logging.info(f"  Best transfer score: {trainer.best_transfer_score:.6f}")
            logging.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logging.info(f"  Final training loss: {training_history['train_loss'][-1]:.6f}")
            logging.info(f"  Final test loss: {training_history['test_loss'][-1]:.6f}")
            
            # Log research insights
            logging.info(f"\nResearch Insights:")
            logging.info(f"  ✓ Trained on {len(config.experiment.train_cities)} cities with {len(config.experiment.train_years)} years")
            logging.info(f"  ✓ Transferred to unseen city: {config.experiment.test_city}")
            logging.info(f"  ✓ Temporal transfer: {config.experiment.test_train_year} → {config.experiment.test_target_year}")
            logging.info(f"  ✓ Multi-task learning with spatial and temporal auxiliary tasks")
            
        else:
            logging.info(f"Training Results:")
            logging.info(f"  Best validation loss: {trainer.best_val_loss:.6f}")
            logging.info(f"  Final training loss: {training_history['train_loss'][-1]:.6f}")
            logging.info(f"  Final test loss: {training_history['test_loss'][-1]:.6f}")
        
        logging.info(f"\nResults saved to: {trainer.save_dir}")
        
        # Success message
        logging.info("\nTraining completed successfully!")
        
        if config.experiment.type == "enhanced_spatiotemporal_transfer":
            logging.info("\nResearch Contribution Summary:")
            logging.info("• Implemented true spatio-temporal transfer learning")
            logging.info("• Multi-task learning with auxiliary spatial and temporal tasks")
            logging.info("• Attention mechanisms for adaptive feature fusion")
            logging.info("• Meta-learning for few-shot adaptation to new cities")
            logging.info("• Evaluation on completely unseen city with temporal distribution shift")
            logging.info("\nThis addresses the research question:")
            logging.info("'Can models transfer to predict COVID-era traffic in unseen cities?'")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        if hasattr(trainer, 'save_results'):
            trainer.save_results()
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
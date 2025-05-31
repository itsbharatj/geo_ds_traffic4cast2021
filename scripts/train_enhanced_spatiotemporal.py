# scripts/train_enhanced_spatiotemporal.py
"""
Enhanced spatio-temporal transfer learning training script

Usage:
    python scripts/train_enhanced_spatiotemporal.py --config config/enhanced_spatiotemporal_config.yaml
    python scripts/train_enhanced_spatiotemporal.py --config config/enhanced_spatiotemporal_config.yaml --comprehensive
"""

import sys
import argparse
import logging
from pathlib import Path

# # Add src to path
# src_path = Path(__file__).parent.parent / "src"
# sys.path.append(str(src_path))

import torch
import numpy as np
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.reproducibility import set_random_seeds
from src.experiments.spatiotemporal_transfer import EnhancedSpatioTemporalTransferExperiment

def main():
    parser = argparse.ArgumentParser(description="Enhanced Spatio-Temporal Transfer Learning")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--experiment-name", help="Override experiment name")
    parser.add_argument("--test-city", help="Override test city")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive analysis")
    parser.add_argument("--device", help="Override device")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.test_city:
        config.experiment.test_city = args.test_city
    if args.comprehensive:
        config.experiment.comprehensive_analysis = True
    if args.device:
        config.training.device = args.device
    if args.debug:
        config.data.limit_per_split = 50
        config.training.epochs = 3
        config.training.batch_size = 2
        config.logging.log_interval = 5
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
    logging.info("ENHANCED SPATIO-TEMPORAL TRANSFER LEARNING EXPERIMENT")
    logging.info("=" * 80)
    logging.info(f"Experiment type: {config.experiment.type}")
    logging.info(f"Config file: {args.config}")
    logging.info(f"Device: {config.training.device}")
    logging.info(f"Random seed: {config.experiment.random_seed}")
    logging.info(f"Comprehensive analysis: {getattr(config.experiment, 'comprehensive_analysis', False)}")
    
    # Log experiment setup
    logging.info("\nExperiment Setup:")
    logging.info(f"  Training cities: {config.experiment.train_cities}")
    logging.info(f"  Training years: {config.experiment.train_years}")
    logging.info(f"  Test city: {config.experiment.test_city}")
    logging.info(f"  Test source year: {config.experiment.test_train_year}")
    logging.info(f"  Test target year: {config.experiment.test_target_year}")
    logging.info(f"  Adaptation samples: {getattr(config.experiment, 'adaptation_samples', 100)}")
    
    # Log model configuration
    logging.info(f"\nModel Configuration:")
    logging.info(f"  Multi-task UNet with {config.model.features}")
    logging.info(f"  Use attention: {getattr(config.model, 'use_attention', True)}")
    logging.info(f"  Use meta-learning: {getattr(config.model, 'use_meta_learning', True)}")
    logging.info(f"  Loss weights: Traffic={getattr(config.training, 'traffic_weight', 1.0)}, "
                f"City={getattr(config.training, 'city_weight', 0.1)}, "
                f"Year={getattr(config.training, 'year_weight', 0.1)}")
    
    # Validate experiment setup
    if config.experiment.test_city in config.experiment.train_cities:
        raise ValueError(
            f"Test city '{config.experiment.test_city}' cannot be in training cities. "
            f"This violates the spatio-temporal transfer learning setup."
        )
    
    # Create and run experiment
    logging.info("\nInitializing enhanced spatio-temporal transfer experiment...")
    experiment = EnhancedSpatioTemporalTransferExperiment(config)
    
    try:
        logging.info("Starting experiment...")
        results = experiment.run()
        
        # Print results summary
        logging.info("=" * 80)
        logging.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)
        
        if 'main_experiment' in results:
            # Comprehensive analysis results
            main_results = results['main_experiment']
            logging.info(f"Main Experiment Results:")
            logging.info(f"  Best transfer score: {main_results['best_transfer_score']:.6f}")
            logging.info(f"  Final test loss: {main_results['final_test_loss']:.6f}")
            logging.info(f"  Domain shifts:")
            logging.info(f"    Spatial: {main_results['domain_analysis']['spatial_shift']}")
            logging.info(f"    Temporal: {main_results['domain_analysis']['temporal_shift']}")
            
            if 'ablation_study' in results and results['ablation_study']:
                logging.info(f"\nAblation Study Results:")
                for variant, result in results['ablation_study'].items():
                    if 'transfer_score' in result:
                        logging.info(f"  {variant}: {result['transfer_score']:.6f}")
            
            if 'cross_city_validation' in results and results['cross_city_validation']:
                cv_results = results['cross_city_validation']
                logging.info(f"\nCross-City Validation:")
                logging.info(f"  Average transfer score: {cv_results.get('avg_transfer_score', 'N/A'):.6f}")
                logging.info(f"  Best target city: {cv_results.get('best_city', 'N/A')}")
            
            logging.info(f"\nResults saved to: {main_results['experiment_dir']}")
            
        else:
            # Single experiment results
            logging.info(f"Transfer Learning Results:")
            logging.info(f"  Best transfer score: {results['best_transfer_score']:.6f}")
            logging.info(f"  Final train loss: {results['final_train_loss']:.6f}")
            logging.info(f"  Final test loss: {results['final_test_loss']:.6f}")
            
            # Domain analysis
            domain_analysis = results.get('domain_analysis', {})
            if domain_analysis:
                logging.info(f"  Domain Analysis:")
                spatial_shift = domain_analysis.get('spatial_shift', {})
                temporal_shift = domain_analysis.get('temporal_shift', {})
                logging.info(f"    Spatial shift: mean={spatial_shift.get('mean_shift', 0):.4f}, "
                           f"sparsity={spatial_shift.get('sparsity_shift', 0):.4f}")
                logging.info(f"    Temporal shift: mean={temporal_shift.get('mean_shift', 0):.4f}, "
                           f"sparsity={temporal_shift.get('sparsity_shift', 0):.4f}")
            
            logging.info(f"\nResults saved to: {results['experiment_dir']}")
        
        # Research insights
        logging.info("\n" + "=" * 80)
        logging.info("RESEARCH INSIGHTS")
        logging.info("=" * 80)
        logging.info("This experiment addresses the research question:")
        logging.info("'Can a model trained on spatio-temporal patterns from multiple cities")
        logging.info("transfer to predict COVID-era traffic in a completely unseen city,")
        logging.info("given only that city's pre-COVID data for adaptation?'")
        logging.info("")
        logging.info("Key innovations implemented:")
        logging.info("1. Multi-task learning with city and year classification")
        logging.info("2. Spatial and temporal attention mechanisms")
        logging.info("3. Meta-learning for few-shot adaptation")
        logging.info("4. True spatio-temporal transfer (unseen city + temporal shift)")
        
    except KeyboardInterrupt:
        logging.info("Experiment interrupted by user")
    except Exception as e:
        logging.error(f"Experiment failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
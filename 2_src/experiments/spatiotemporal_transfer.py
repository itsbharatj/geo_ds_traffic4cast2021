# src/experiments/enhanced_spatiotemporal_transfer.py
"""
Enhanced Spatio-Temporal Transfer Learning Experiment
Implements true spatio-temporal transfer with multi-task learning and few-shot adaptation
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import torch
import numpy as np

from ..utils.config import Config
from ..data.spatiotemporal_dataset import SpatioTemporalDataManager
from ..models.multitask_unet import create_multitask_unet
from ..training.spatiotemporal_trainer import create_spatiotemporal_trainer

class SpatioTemporalTransferExperiment:
    """Enhanced spatio-temporal transfer learning experiment with multi-task learning"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate experiment configuration"""
        if self.config.experiment.type != "enhanced_spatiotemporal_transfer":
            raise ValueError("Config must be for enhanced_spatiotemporal_transfer experiment")
            
        required_fields = [
            'train_cities', 'train_years', 'test_city', 
            'test_train_year', 'test_target_year'
        ]
        
        for field in required_fields:
            if not hasattr(self.config.experiment, field):
                raise ValueError(f"Missing required field: {field}")
        
        # Ensure test city is not in training cities
        if self.config.experiment.test_city in self.config.experiment.train_cities:
            raise ValueError("Test city cannot be in training cities for spatio-temporal transfer")
    
    def setup_data(self) -> Tuple[Any, Any, Any]:
        """Setup data for enhanced spatio-temporal transfer"""
        
        logging.info("Setting up enhanced spatio-temporal transfer data...")
        
        data_manager = SpatioTemporalDataManager(self.config)
        train_dataset, adapt_dataset, test_dataset = data_manager.create_datasets()
        
        # Log dataset statistics
        logging.info("Enhanced Spatio-Temporal Transfer Data Setup:")
        logging.info(f"  Training Cities: {self.config.experiment.train_cities}")
        logging.info(f"  Training Years: {self.config.experiment.train_years}")
        logging.info(f"  Test City: {self.config.experiment.test_city}")
        logging.info(f"  Test Train Year: {self.config.experiment.test_train_year}")
        logging.info(f"  Test Target Year: {self.config.experiment.test_target_year}")
        
        logging.info(f"  Train samples: {len(train_dataset)}")
        logging.info(f"    City distribution: {train_dataset.get_city_distribution()}")
        logging.info(f"    Year distribution: {train_dataset.get_year_distribution()}")
        
        logging.info(f"  Adaptation samples: {len(adapt_dataset)}")
        logging.info(f"  Test samples: {len(test_dataset)}")
        
        return train_dataset, adapt_dataset, test_dataset
    
    def analyze_domain_shift(self, train_dataset, adapt_dataset, test_dataset) -> Dict[str, Any]:
        """Analyze domain shift between training and target domains"""
        
        logging.info("Analyzing domain shift...")
        
        def sample_statistics(dataset, num_samples=50):
            """Sample statistics from dataset"""
            if len(dataset) == 0:
                return {'mean': 0, 'std': 0, 'sparsity': 0}
                
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
            
            stats = []
            for idx in indices:
                inputs, targets, metadata = dataset[idx]
                
                if torch.is_tensor(inputs):
                    data = inputs.numpy()
                else:
                    data = inputs
                
                stats.append({
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'sparsity': np.mean(data == 0)
                })
            
            return {
                'mean': np.mean([s['mean'] for s in stats]),
                'std': np.mean([s['std'] for s in stats]),
                'sparsity': np.mean([s['sparsity'] for s in stats])
            }
        
        # Compute statistics for each domain
        train_stats = sample_statistics(train_dataset)
        adapt_stats = sample_statistics(adapt_dataset)
        test_stats = sample_statistics(test_dataset)
        
        # Calculate domain shifts
        spatial_shift = {
            'mean_shift': abs(train_stats['mean'] - test_stats['mean']),
            'sparsity_shift': abs(train_stats['sparsity'] - test_stats['sparsity'])
        }
        
        temporal_shift = {
            'mean_shift': abs(adapt_stats['mean'] - test_stats['mean']),
            'sparsity_shift': abs(adapt_stats['sparsity'] - test_stats['sparsity'])
        }
        
        domain_analysis = {
            'train_stats': train_stats,
            'adapt_stats': adapt_stats,
            'test_stats': test_stats,
            'spatial_shift': spatial_shift,
            'temporal_shift': temporal_shift
        }
        
        logging.info("Domain Shift Analysis:")
        logging.info(f"  Spatial shift (train→test): mean={spatial_shift['mean_shift']:.4f}, sparsity={spatial_shift['sparsity_shift']:.4f}")
        logging.info(f"  Temporal shift (adapt→test): mean={temporal_shift['mean_shift']:.4f}, sparsity={temporal_shift['sparsity_shift']:.4f}")
        
        return domain_analysis
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run ablation study on different components"""
        
        logging.info("Running ablation study...")
        
        ablation_configs = [
            {"name": "baseline", "use_attention": False, "use_meta_learning": False},
            {"name": "with_attention", "use_attention": True, "use_meta_learning": False},
            {"name": "with_meta", "use_attention": False, "use_meta_learning": True},
            {"name": "full_model", "use_attention": True, "use_meta_learning": True},
        ]
        
        ablation_results = {}
        
        for config_variant in ablation_configs:
            logging.info(f"Testing variant: {config_variant['name']}")
            
            # Modify config for this variant
            original_attention = getattr(self.config.model, 'use_attention', True)
            original_meta = getattr(self.config.model, 'use_meta_learning', True)
            
            self.config.model.use_attention = config_variant['use_attention']
            self.config.model.use_meta_learning = config_variant['use_meta_learning']
            
            try:
                # Run experiment with this configuration
                results = self.run_single_experiment()
                ablation_results[config_variant['name']] = {
                    'transfer_score': results['best_transfer_score'],
                    'final_test_loss': results['final_test_loss'],
                    'config': config_variant
                }
                
            except Exception as e:
                logging.error(f"Ablation variant {config_variant['name']} failed: {e}")
                ablation_results[config_variant['name']] = {'error': str(e)}
            
            finally:
                # Restore original config
                self.config.model.use_attention = original_attention
                self.config.model.use_meta_learning = original_meta
        
        return ablation_results
    
    def run_single_experiment(self) -> Dict[str, Any]:
        """Run a single enhanced spatio-temporal transfer experiment"""
        
        # Setup data
        train_dataset, adapt_dataset, test_dataset = self.setup_data()
        
        # Analyze domain shift
        domain_analysis = self.analyze_domain_shift(train_dataset, adapt_dataset, test_dataset)
        
        # Create data loaders
        data_manager = SpatioTemporalDataManager(self.config)
        train_loader, adapt_loader, test_loader = data_manager.create_data_loaders(
            train_dataset, adapt_dataset, test_dataset
        )
        
        # Update config with dataset information
        self.config.experiment.num_cities = train_dataset.get_num_cities()
        self.config.experiment.num_years = train_dataset.get_num_years()
        
        # Create model
        model = create_multitask_unet(self.config)
        
        # Log model information
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            logging.info(f"Model created:")
            logging.info(f"  Architecture: Multi-task UNet")
            logging.info(f"  Parameters: {model_info.get('trainable_parameters', 'N/A')}")
            logging.info(f"  Input channels: {model_info.get('in_channels', 'N/A')}")
            logging.info(f"  Output channels: {model_info.get('out_channels', 'N/A')}")
            logging.info(f"  Use attention: {getattr(model, 'use_attention', 'N/A')}")
            logging.info(f"  Use meta-learning: {getattr(model, 'use_meta_learning', 'N/A')}")
        
        # Create trainer
        trainer = create_spatiotemporal_trainer(
            model, train_loader, adapt_loader, test_loader, self.config
        )
        
        # Train model
        training_history = trainer.fit()
        
        # Collect results
        results = {
            'experiment_type': 'enhanced_spatiotemporal_transfer',
            'config': {
                'train_cities': self.config.experiment.train_cities,
                'train_years': self.config.experiment.train_years,
                'test_city': self.config.experiment.test_city,
                'test_train_year': self.config.experiment.test_train_year,
                'test_target_year': self.config.experiment.test_target_year,
                'adaptation_samples': getattr(self.config.experiment, 'adaptation_samples', 100)
            },
            'domain_analysis': domain_analysis,
            'best_transfer_score': trainer.best_transfer_score,
            'final_train_loss': training_history['train_loss'][-1],
            'final_test_loss': training_history['test_loss'][-1],
            'final_transfer_score': training_history['transfer_score'][-1],
            'training_history': training_history,
            'model_path': str(trainer.save_dir / "final_model.pth"),
            'experiment_dir': str(trainer.save_dir)
        }
        
        return results
    
    def run_cross_city_validation(self) -> Dict[str, Any]:
        """Run cross-validation across different target cities"""
        
        available_cities = ["BARCELONA", "MELBOURNE", "NEWYORK", "CHICAGO", "ANTWERP", "VIENNA", "BERLIN", "BANGKOK", "MOSCOW"]
        
        # Define different target cities for validation
        target_cities = [city for city in available_cities 
                        if city not in self.config.experiment.train_cities]
        
        if len(target_cities) < 2:
            logging.warning("Not enough cities for cross-validation")
            return {}
        
        results = []
        original_test_city = self.config.experiment.test_city
        
        for target_city in target_cities[:3]:  # Limit to 3 for computational efficiency
            logging.info(f"\n{'='*60}")
            logging.info(f"CROSS-VALIDATION WITH TARGET CITY: {target_city}")
            logging.info(f"{'='*60}")
            
            # Update config for this target city
            self.config.experiment.test_city = target_city
            
            try:
                city_results = self.run_single_experiment()
                city_results['target_city'] = target_city
                results.append(city_results)
                
            except Exception as e:
                logging.error(f"Cross-validation for city {target_city} failed: {e}")
                continue
        
        # Restore original config
        self.config.experiment.test_city = original_test_city
        
        # Aggregate results
        if results:
            avg_transfer_score = np.mean([r['best_transfer_score'] for r in results])
            std_transfer_score = np.std([r['best_transfer_score'] for r in results])
            
            summary = {
                'experiment_type': 'cross_city_validation',
                'num_cities': len(results),
                'avg_transfer_score': avg_transfer_score,
                'std_transfer_score': std_transfer_score,
                'city_results': results,
                'best_city': min(results, key=lambda x: x['best_transfer_score'])['target_city'],
                'worst_city': max(results, key=lambda x: x['best_transfer_score'])['target_city']
            }
            
            logging.info(f"\nCross-City Validation Summary:")
            logging.info(f"  Average transfer score: {avg_transfer_score:.4f} ± {std_transfer_score:.4f}")
            logging.info(f"  Best target city: {summary['best_city']}")
            logging.info(f"  Worst target city: {summary['worst_city']}")
            
            return summary
        else:
            raise RuntimeError("All cross-validation experiments failed")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis including ablation and cross-validation"""
        
        logging.info("Starting comprehensive enhanced spatio-temporal transfer analysis...")
        
        # Main experiment
        main_results = self.run_single_experiment()
        
        # Ablation study
        try:
            ablation_results = self.run_ablation_study()
        except Exception as e:
            logging.error(f"Ablation study failed: {e}")
            ablation_results = {}
        
        # Cross-city validation
        try:
            cross_city_results = self.run_cross_city_validation()
        except Exception as e:
            logging.error(f"Cross-city validation failed: {e}")
            cross_city_results = {}
        
        # Comprehensive results
        comprehensive_results = {
            'experiment_type': 'comprehensive_enhanced_spatiotemporal_transfer',
            'main_experiment': main_results,
            'ablation_study': ablation_results,
            'cross_city_validation': cross_city_results,
            'summary': {
                'best_transfer_score': main_results['best_transfer_score'],
                'spatial_shift': main_results['domain_analysis']['spatial_shift'],
                'temporal_shift': main_results['domain_analysis']['temporal_shift']
            }
        }
        
        return comprehensive_results
    
    def run(self) -> Dict[str, Any]:
        """Run the enhanced spatio-temporal transfer experiment"""
        logging.info("Starting enhanced spatio-temporal transfer learning experiment...")
        
        # Check if comprehensive analysis is requested
        if hasattr(self.config.experiment, 'comprehensive_analysis') and self.config.experiment.comprehensive_analysis:
            return self.run_comprehensive_analysis()
        else:
            return self.run_single_experiment()
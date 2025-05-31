# scripts/evaluate.py
"""
Evaluation script for trained Traffic4Cast models

Usage:
    python scripts/evaluate.py --model-path experiments/model.pth --config config/test.yaml
    python scripts/evaluate.py --experiment-dir experiments/spatial_transfer_20240101_120000/
"""

import sys
import argparse
import logging
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging_utils import setup_logging
from src.utils.config import load_config, Config
from src.data.splitter import ExperimentDataManager
from src.data.dataset import create_data_loaders
from src.models.unet import create_unet_model
from src.training.metrics import TrafficMetrics

class ModelEvaluator:
    """Evaluate trained Traffic4Cast models"""
    
    def __init__(self, model_path: str, config: Config):
        self.model_path = Path(model_path)
        self.config = config
        self.device = config.training.device
        
        # Load model
        self.model = self._load_model()
        self.metrics = TrafficMetrics()
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        
        # Create model architecture
        model = create_unet_model(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        logging.info(f"Loaded model from {self.model_path}")
        return model
    
    def evaluate_dataset(self, data_loader, dataset_name="Dataset"):
        """Evaluate model on a dataset"""
        
        logging.info(f"Evaluating on {dataset_name}...")
        
        all_predictions = []
        all_targets = []
        batch_losses = []
        
        criterion = torch.nn.MSELoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Handle output format
                if predictions.dim() == 4 and targets.dim() == 5:
                    B, T, H, W, C = targets.shape
                    targets = targets.permute(0, 1, 4, 2, 3)
                    targets = targets.reshape(B, T * C, H, W)
                
                # Compute loss
                loss = criterion(predictions, targets)
                batch_losses.append(loss.item())
                
                # Store predictions and targets
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logging.info(f"  Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.6f}")
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets)
        metrics['avg_loss'] = np.mean(batch_losses)
        
        logging.info(f"{dataset_name} Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logging.info(f"  {key}: {value:.6f}")
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'batch_losses': batch_losses
        }
    
    def evaluate_all_datasets(self):
        """Evaluate on train, validation, and test sets"""
        
        # Setup data
        data_manager = ExperimentDataManager(self.config)
        train_dataset, val_dataset, test_dataset = data_manager.setup_experiment_data()
        
        train_loader, val_loader, test_loader = create_data_loaders(
            self.config, train_dataset, val_dataset, test_dataset
        )
        
        # Evaluate on all datasets
        results = {}
        results['train'] = self.evaluate_dataset(train_loader, "Training Set")
        results['validation'] = self.evaluate_dataset(val_loader, "Validation Set")  
        results['test'] = self.evaluate_dataset(test_loader, "Test Set")
        
        return results
    
    def create_visualizations(self, results, save_dir):
        """Create evaluation visualizations"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Metrics comparison across datasets
        datasets = ['train', 'validation', 'test']
        metrics_to_plot = ['mse', 'mae', 'volume_mse', 'speed_mse']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [results[ds]['metrics'].get(metric, 0) for ds in datasets]
            axes[i].bar(datasets, values)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Value')
            
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_comparison.png')
        plt.close()
        
        # 2. Prediction examples
        test_predictions = results['test']['predictions']
        test_targets = results['test']['targets']
        
        # Select a few examples
        num_examples = min(4, test_predictions.shape[0])
        indices = np.random.choice(test_predictions.shape[0], num_examples, replace=False)
        
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)
            
        for i, idx in enumerate(indices):
            pred = test_predictions[idx]
            target = test_targets[idx]
            
            # Show first channel of first timestep
            if pred.ndim == 3:  # (C, H, W)
                pred_vis = pred[0]
                target_vis = target[0]
            else:  # Handle other formats
                pred_vis = pred[0, 0] if pred.ndim == 4 else pred.mean(axis=0)
                target_vis = target[0, 0] if target.ndim == 4 else target.mean(axis=0)
            
            # Target
            axes[i, 0].imshow(target_vis, cmap='viridis')
            axes[i, 0].set_title(f'Ground Truth {i+1}')
            axes[i, 0].axis('off')
            
            # Prediction
            axes[i, 1].imshow(pred_vis, cmap='viridis')
            axes[i, 1].set_title(f'Prediction {i+1}')
            axes[i, 1].axis('off')
            
            # Error
            error = np.abs(pred_vis - target_vis)
            axes[i, 2].imshow(error, cmap='Reds')
            axes[i, 2].set_title(f'Error {i+1}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'prediction_examples.png')
        plt.close()
        
        logging.info(f"Visualizations saved to {save_dir}")
    
    def save_results(self, results, save_path):
        """Save evaluation results"""
        
        save_path = Path(save_path)
        
        # Save metrics only (predictions are too large)
        metrics_results = {}
        for dataset_name, dataset_results in results.items():
            metrics_results[dataset_name] = dataset_results['metrics']
        
        with open(save_path, 'w') as f:
            json.dump(metrics_results, f, indent=2)
            
        logging.info(f"Results saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Traffic4Cast Model")
    parser.add_argument("--model-path", help="Path to model checkpoint")
    parser.add_argument("--experiment-dir", help="Path to experiment directory")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Determine model path and config
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
        model_path = exp_dir / "final_model.pth"
        config_path = exp_dir / "config.yaml"
        
        if not model_path.exists():
            model_path = exp_dir / "checkpoints" / "best_model.pth"
            
    elif args.model_path:
        model_path = args.model_path
        config_path = args.config
        
    else:
        raise ValueError("Must specify either --model-path or --experiment-dir")
    
    # Load config
    if not config_path or not Path(config_path).exists():
        raise ValueError(f"Config file not found: {config_path}")
        
    config = load_config(str(config_path))
    
    # Create evaluator
    evaluator = ModelEvaluator(str(model_path), config)
    
    # Run evaluation
    logging.info("Starting model evaluation...")
    results = evaluator.evaluate_all_datasets()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    evaluator.save_results(results, output_dir / "evaluation_results.json")
    
    # Create visualizations
    evaluator.create_visualizations(results, output_dir / "visualizations")
    
    # Print summary
    logging.info("=" * 60)
    logging.info("EVALUATION SUMMARY")
    logging.info("=" * 60)
    
    for dataset_name in ['train', 'validation', 'test']:
        metrics = results[dataset_name]['metrics']
        logging.info(f"{dataset_name.capitalize()} Set:")
        logging.info(f"  MSE: {metrics['mse']:.6f}")
        logging.info(f"  MAE: {metrics['mae']:.6f}")
        if 'volume_accuracy' in metrics:
            logging.info(f"  Volume Accuracy: {metrics['volume_accuracy']:.4f}")
        if 'speed_accuracy' in metrics:
            logging.info(f"  Speed Accuracy: {metrics['speed_accuracy']:.4f}")
    
    logging.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
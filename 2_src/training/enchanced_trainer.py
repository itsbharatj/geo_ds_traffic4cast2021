#!/usr/bin/env python3
"""
Usage example and training script for enhanced Traffic4Cast model with extra channels.

This script demonstrates how to use the enhanced dataset and model with the additional
temporal channels (time of day/day of week statistics).
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from pathlib import Path
import argparse
from datetime import datetime

import sys
sys.path.append('/Users/bharatjain/Desktop/Sem-6/GeoDS/project/geo_ds_traffic4cast2021/2_src')

# Import our enhanced components
from data.enhanced_dataset import (
    EnhancedTrafficDataset, 
    EnhancedTrafficDataTransform,
    create_enhanced_data_loaders,
    get_enhanced_dataset_stats
)
from models.enhanced_unet import TrafficUNetModel, create_traffic_model


class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        # Data parameters
        self.data_root = "/path/to/your/traffic4cast/data"
        self.cities = ["BANGKOK", "MELBOURNE"]  # List of cities to include
        self.years = ["2019"]  # List of years to include
        
        # Model parameters
        self.use_enhanced_channels = True
        self.input_timesteps = 12  # 1 hour of 5-minute intervals
        self.output_timesteps = 6  # 30 minutes prediction
        self.stats_window_days = 30  # Days to look back for statistics
        
        # Training parameters
        self.batch_size = 4  # Adjust based on your GPU memory
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_workers = 4
        self.pin_memory = True
        
        # Model architecture
        self.unet_depth = 5
        self.unet_wf = 6
        self.padding = True
        self.batch_norm = True
        
        # Data transforms
        self.normalize = True
        self.stack_channels_on_time = True
        
        # Validation split
        self.val_split = 0.2
        self.test_split = 0.1
        
        # Logging and checkpoints
        self.log_interval = 10
        self.save_interval = 10
        self.checkpoint_dir = "./checkpoints"
        self.log_dir = "./logs"


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_datasets(config: TrainingConfig):
    """Create train, validation, and test datasets"""
    
    # Create transform
    num_channels = 10 if config.use_enhanced_channels else 8
    transform = EnhancedTrafficDataTransform(
        stack_channels_on_time=config.stack_channels_on_time,
        normalize=config.normalize,
        num_channels=num_channels
    )
    
    # Create full dataset
    full_dataset = EnhancedTrafficDataset(
        root_dir=config.data_root,
        cities=config.cities,
        years=config.years,
        use_enhanced_channels=config.use_enhanced_channels,
        stats_window_days=config.stats_window_days,
        input_timesteps=config.input_timesteps,
        output_timesteps=config.output_timesteps,
        transform=transform
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * config.test_split)
    val_size = int(total_size * config.val_split)
    train_size = total_size - test_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset, full_dataset


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, val_loader, criterion, device, epoch, logger):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    logger.info(f'Validation Epoch {epoch}, Average Loss: {avg_loss:.6f}')
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Train enhanced Traffic4Cast model')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory of Traffic4Cast data')
    parser.add_argument('--cities', nargs='+', default=["BANGKOK", "MELBOURNE"],
                        help='Cities to include in training')
    parser.add_argument('--years', nargs='+', default=["2019"],
                        help='Years to include in training')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--no-enhanced', action='store_true',
                        help='Disable enhanced channels')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory for logs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting enhanced Traffic4Cast training")
    
    # Create configuration
    config = TrainingConfig()
    config.data_root = args.data_root
    config.cities = args.cities
    config.years = args.years
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.use_enhanced_channels = not args.no_enhanced
    config.checkpoint_dir = args.checkpoint_dir
    config.log_dir = args.log_dir
    
    logger.info(f"Configuration: Enhanced channels = {config.use_enhanced_channels}")
    logger.info(f"Cities: {config.cities}")
    logger.info(f"Years: {config.years}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset, test_dataset, full_dataset = create_datasets(config)
    
    # Log dataset statistics
    stats = get_enhanced_dataset_stats(full_dataset.dataset)
    logger.info(f"Dataset statistics: {stats}")
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=config.pin_memory
    )
    
    # Create model
    logger.info("Creating model...")
    model_type = "enhanced" if config.use_enhanced_channels else "original"
    model = create_traffic_model(
        model_type,
        input_timesteps=config.input_timesteps,
        output_timesteps=config.output_timesteps,
        unet_depth=config.unet_depth,
        unet_wf=config.unet_wf,
        padding=config.padding,
        batch_norm=config.batch_norm
    )
    
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, logger)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch, logger)
        
        logger.info(f'Epoch {epoch}/{config.num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_checkpoint(model, optimizer, epoch, val_loss, config.checkpoint_dir)
            logger.info(f'New best model saved: {checkpoint_path}')
        
        # Save periodic checkpoint
        if epoch % config.save_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch, val_loss, config.checkpoint_dir)
            logger.info(f'Periodic checkpoint saved: {checkpoint_path}')
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
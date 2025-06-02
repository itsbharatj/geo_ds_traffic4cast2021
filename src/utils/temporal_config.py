# src/utils/config.py - Enhanced with temporal channels support
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    name: str = "unet"
    in_channels: int = 96  # 12 timesteps * 8 channels
    out_channels: int = 48  # 6 timesteps * 8 channels
    features: list = None
    use_attention: bool = False
    use_meta_learning: bool = False
    bilinear: bool = False
    
    def __post_init__(self):
        if self.features is None:
            self.features = [64, 128, 256, 512]

@dataclass 
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    device: str = "auto"
    num_workers: Optional[int] = None
    pin_memory: bool = True
    # Multi-task loss weights
    traffic_weight: float = 1.0
    city_weight: float = 0.1
    year_weight: float = 0.1
    enhanced_traffic_weight: float = 0.5
    
@dataclass
class DataConfig:
    root_dir: str = "./data/raw"
    cities: list = None
    input_timesteps: int = 12
    output_timesteps: int = 6
    channels: int = 8
    height: int = 495
    width: int = 436
    limit_per_split: Optional[int] = None
    # Enhanced temporal channels configuration
    use_enhanced_channels: bool = False
    stats_window_days: int = 30
    enhanced_channels: int = 2  # Number of additional channels (time_dow, dow)
    
    def __post_init__(self):
        if self.cities is None:
            self.cities = ["BARCELONA", "MELBOURNE", "NEWYORK", "CHICAGO", 
                          "ANTWERP", "VIENNA", "BERLIN", "BANGKOK", "MOSCOW"]
        
        # Update total channels if enhanced channels are enabled
        if self.use_enhanced_channels:
            self.total_channels = self.channels + self.enhanced_channels
        else:
            self.total_channels = self.channels

@dataclass
class ExperimentConfig:
    type: str = "spatial_transfer"
    train_cities: list = None
    test_cities: list = None
    train_years: list = None
    test_years: list = None
    years: list = None  # Backward compatibility
    test_city: str = None
    test_train_year: str = None
    test_target_year: str = None
    adaptation_samples: int = 100
    num_cities: int = 1
    num_years: int = 1
    val_fraction: float = 0.1
    random_seed: int = 42

    def __init__(self, **kwargs):
        # Set all known fields from kwargs, ignore unknown
        for field in self.__dataclass_fields__:
            setattr(self, field, kwargs.pop(field, None))
        # Backward compatibility
        if self.years is not None and self.train_years is None:
            self.train_years = self.years
        if self.type == "spatial_transfer":
            if self.train_cities is None:
                self.train_cities = ["ANTWERP", "BANGKOK", "MOSCOW"]
            if self.test_cities is None:
                self.test_cities = ["BARCELONA", "MELBOURNE"]
        elif self.type == "spatiotemporal_transfer":
            if self.train_years is None:
                self.train_years = ["2019"]
            if self.test_years is None:
                self.test_years = ["2020"]

@dataclass
class LoggingConfig:
    experiment_name: str = "traffic4cast_experiment"
    save_dir: str = "./experiments"
    log_interval: int = 100
    save_checkpoints: bool = True
    use_tensorboard: bool = True
    
class Config:
    """Main configuration class with enhanced temporal channels support"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # Load from YAML if provided
        config_dict = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Initialize sub-configs
        self.data = DataConfig(**config_dict.get('data', {}))
        self.model = ModelConfig(**config_dict.get('model', {}))
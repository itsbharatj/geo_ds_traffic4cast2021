# src/utils/config.py
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
    num_workers: Optional[int] = None  # Set to None to use default
    pin_memory: bool = True
    
@dataclass
class DataConfig:
    root_dir: str = "./data/raw"
    cities: list = None
    input_timesteps: int = 12
    output_timesteps: int = 6
    channels: int = 8
    height: int = 495
    width: int = 436
    limit_per_split: Optional[int] = None  # Add this line
    
    def __post_init__(self):
        if self.cities is None:
            self.cities = ["BARCELONA", "MELBOURNE", "NEWYORK", "CHICAGO", 
                          "ANTWERP", "VIENNA", "BERLIN", "BANGKOK", "MOSCOW"]

@dataclass
class ExperimentConfig:
    type: str = "spatial_transfer"  # or "spatiotemporal_transfer"
    train_cities: list = None
    test_cities: list = None
    train_years: list = None
    test_years: list = None
    val_fraction: float = 0.1
    random_seed: int = 42
    
    def __post_init__(self):
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
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        # Load from YAML if provided
        config_dict = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        
        # Override with kwargs
        config_dict.update(kwargs)
        
        # Initialize sub-configs
        self.model = ModelConfig(**config_dict.get('model', {}))
        self.training = TrainingConfig(**config_dict.get('training', {}))
        # Ensure num_workers is int if not None
        if self.training.num_workers is not None and not isinstance(self.training.num_workers, int):
            self.training.num_workers = int(self.training.num_workers)
        self.data = DataConfig(**config_dict.get('data', {}))
        self.experiment = ExperimentConfig(**config_dict.get('experiment', {}))
        self.logging = LoggingConfig(**config_dict.get('logging', {}))
        
        # Auto-detect device
        if self.training.device == "auto":
            self.training.device = self._detect_device()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'experiment': self.experiment.__dict__,
            'logging': self.logging.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create config from command line arguments"""
        config_path = getattr(args, 'config', None)
        
        # Convert args to dict, excluding None values
        kwargs = {k: v for k, v in vars(args).items() 
                 if v is not None and k != 'config'}
        
        return cls(config_path, **kwargs)

def load_config(config_path: str, **overrides) -> Config:
    """Load configuration from YAML file with optional overrides"""
    return Config(config_path, **overrides)

def create_experiment_config(experiment_type: str, **kwargs) -> Config:
    """Create configuration for specific experiment type"""
    base_config = {
        'experiment': {'type': experiment_type}
    }
    base_config.update(kwargs)
    return Config(**base_config)
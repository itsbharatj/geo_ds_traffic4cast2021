# src/data/spatiotemporal_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
from collections import defaultdict

from .dataset import EnhancedTrafficDataset, TrafficDataTransform
from ..utils.constants import MAX_TEST_SLOT_INDEX

class SpatioTemporalTransferDataset(Dataset):
    """
    Enhanced dataset for spatio-temporal transfer learning with multi-task learning.
    
    This dataset supports:
    1. Multi-city, multi-year training data
    2. Few-shot adaptation data for target city
    3. Target city temporal transfer prediction
    """
    
    def __init__(self,
                 root_dir: str,
                 mode: str = "train",  # "train", "adapt", "test"
                 train_cities: List[str] = None,
                 train_years: List[str] = None,
                 test_city: str = None,
                 test_train_year: str = None,
                 test_target_year: str = None,
                 adaptation_samples: int = 100,
                 transform: Optional[TrafficDataTransform] = None,
                 limit: Optional[int] = None,
                 input_timesteps: int = 12,
                 output_timesteps: int = 6):
        """
        Args:
            root_dir: Root directory containing traffic data
            mode: Dataset mode - "train", "adapt", or "test"
            train_cities: Cities used for multi-task training
            train_years: Years used for training (both pre-COVID and COVID)
            test_city: Target city for spatio-temporal transfer
            test_train_year: Year from test city used for adaptation
            test_target_year: Target year to predict in test city
            adaptation_samples: Number of samples from test city for adaptation
            transform: Data transform
            limit: Limit on dataset size
            input_timesteps: Number of input time steps
            output_timesteps: Number of output time steps
        """
        
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.train_cities = train_cities or []
        self.train_years = train_years or []
        self.test_city = test_city
        self.test_train_year = test_train_year
        self.test_target_year = test_target_year
        self.adaptation_samples = adaptation_samples
        self.transform = transform
        self.limit = limit
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        
        # Load data based on mode
        self.samples = []
        self.city_labels = {}  # For multi-task learning
        self.year_labels = {}  # For temporal pattern learning
        
        self._load_data()
        
        logging.info(f"SpatioTemporalTransferDataset ({mode}): {len(self.samples)} samples")
        
    def _load_data(self):
        """Load data based on the dataset mode"""
        
        if self.mode == "train":
            self._load_training_data()
        elif self.mode == "adapt":
            self._load_adaptation_data()
        elif self.mode == "test":
            self._load_test_data()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _load_training_data(self):
        """Load multi-city, multi-year training data"""
        
        city_counter = 0
        year_counter = 0
        
        for city in self.train_cities:
            if city not in self.city_labels:
                self.city_labels[city] = city_counter
                city_counter += 1
                
            for year in self.train_years:
                if year not in self.year_labels:
                    self.year_labels[year] = year_counter
                    year_counter += 1
                
                # Load data for this city-year combination
                dataset = EnhancedTrafficDataset(
                    root_dir=str(self.root_dir),
                    cities=[city],
                    years=[year],
                    transform=self.transform,
                    input_timesteps=self.input_timesteps,
                    output_timesteps=self.output_timesteps
                )
                
                # Add samples with multi-task labels
                for i in range(len(dataset)):
                    if self.limit and len(self.samples) >= self.limit:
                        break
                        
                    sample = {
                        'dataset': dataset,
                        'index': i,
                        'city': city,
                        'year': year,
                        'city_label': self.city_labels[city],
                        'year_label': self.year_labels[year]
                    }
                    self.samples.append(sample)
                    
                if self.limit and len(self.samples) >= self.limit:
                    break
            
            if self.limit and len(self.samples) >= self.limit:
                break
    
    def _load_adaptation_data(self):
        """Load adaptation data from test city's training year"""
        
        if not self.test_city or not self.test_train_year:
            raise ValueError("test_city and test_train_year required for adaptation mode")
        
        # Load test city's training year data
        dataset = EnhancedTrafficDataset(
            root_dir=str(self.root_dir),
            cities=[self.test_city],
            years=[self.test_train_year],
            transform=self.transform,
            input_timesteps=self.input_timesteps,
            output_timesteps=self.output_timesteps
        )
        
        # Sample limited number of adaptation samples
        total_samples = min(len(dataset), self.adaptation_samples)
        indices = np.random.choice(len(dataset), total_samples, replace=False)
        
        self.city_labels[self.test_city] = 0  # New city
        self.year_labels[self.test_train_year] = 0
        
        for i in indices:
            sample = {
                'dataset': dataset,
                'index': i,
                'city': self.test_city,
                'year': self.test_train_year,
                'city_label': 0,  # New city
                'year_label': 0
            }
            self.samples.append(sample)
    
    def _load_test_data(self):
        """Load test data from test city's target year"""
        
        if not self.test_city or not self.test_target_year:
            raise ValueError("test_city and test_target_year required for test mode")
        
        # Load test city's target year data
        dataset = EnhancedTrafficDataset(
            root_dir=str(self.root_dir),
            cities=[self.test_city],
            years=[self.test_target_year],
            transform=self.transform,
            input_timesteps=self.input_timesteps,
            output_timesteps=self.output_timesteps
        )
        
        self.city_labels[self.test_city] = 0  # New city
        self.year_labels[self.test_target_year] = 0
        
        for i in range(len(dataset)):
            if self.limit and len(self.samples) >= self.limit:
                break
                
            sample = {
                'dataset': dataset,
                'index': i,
                'city': self.test_city,
                'year': self.test_target_year,
                'city_label': 0,  # New city
                'year_label': 0
            }
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a sample with multi-task labels
        
        Returns:
            inputs: Input tensor
            targets: Target tensor
            metadata: Dictionary with city_label, year_label, city, year
        """
        
        sample_info = self.samples[idx]
        dataset = sample_info['dataset']
        sample_idx = sample_info['index']
        
        # Get the actual data
        inputs, targets = dataset[sample_idx]
        
        # Prepare metadata
        metadata = {
            'city_label': sample_info['city_label'],
            'year_label': sample_info['year_label'],
            'city': sample_info['city'],
            'year': sample_info['year']
        }
        
        return inputs, targets, metadata
    
    def get_num_cities(self) -> int:
        """Get number of unique cities"""
        return len(self.city_labels)
    
    def get_num_years(self) -> int:
        """Get number of unique years"""
        return len(self.year_labels)
    
    def get_city_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per city"""
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['city']] += 1
        return dict(distribution)
    
    def get_year_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per year"""
        distribution = defaultdict(int)
        for sample in self.samples:
            distribution[sample['year']] += 1
        return dict(distribution)

class SpatioTemporalDataManager:
    """Manager for spatio-temporal transfer learning datasets"""
    
    def __init__(self, config):
        self.config = config
        
        # Create transform
        self.transform = TrafficDataTransform(
            stack_channels_on_time=True,
            normalize=False,
            add_padding=None
        )
    
    def create_datasets(self) -> Tuple[SpatioTemporalTransferDataset, 
                                     SpatioTemporalTransferDataset,
                                     SpatioTemporalTransferDataset]:
        """
        Create training, adaptation, and test datasets
        
        Returns:
            train_dataset: Multi-city, multi-year training data
            adapt_dataset: Few-shot adaptation data from test city
            test_dataset: Test data from test city's target year
        """
        
        # Training dataset: Multi-city, multi-year
        train_dataset = SpatioTemporalTransferDataset(
            root_dir=self.config.data.root_dir,
            mode="train",
            train_cities=self.config.experiment.train_cities,
            train_years=self.config.experiment.train_years,
            transform=self.transform,
            limit=getattr(self.config.data, 'limit_per_split', None),
            input_timesteps=self.config.data.input_timesteps,
            output_timesteps=self.config.data.output_timesteps
        )
        
        # Adaptation dataset: Few-shot data from test city
        adapt_dataset = SpatioTemporalTransferDataset(
            root_dir=self.config.data.root_dir,
            mode="adapt",
            test_city=self.config.experiment.test_city,
            test_train_year=self.config.experiment.test_train_year,
            adaptation_samples=getattr(self.config.experiment, 'adaptation_samples', 100),
            transform=self.transform,
            input_timesteps=self.config.data.input_timesteps,
            output_timesteps=self.config.data.output_timesteps
        )
        
        # Test dataset: Target predictions
        test_dataset = SpatioTemporalTransferDataset(
            root_dir=self.config.data.root_dir,
            mode="test",
            test_city=self.config.experiment.test_city,
            test_target_year=self.config.experiment.test_target_year,
            transform=self.transform,
            limit=getattr(self.config.data, 'limit_per_split', None),
            input_timesteps=self.config.data.input_timesteps,
            output_timesteps=self.config.data.output_timesteps
        )
        
        return train_dataset, adapt_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, adapt_dataset, test_dataset):
        """Create data loaders for all datasets"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=True
        )
        
        adapt_loader = DataLoader(
            adapt_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=self.config.training.pin_memory,
            drop_last=False
        )
        
        return train_loader, adapt_loader, test_loader
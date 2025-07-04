# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Any, Dict
import logging
from collections import defaultdict

# Import existing utilities (modify paths as needed)
from utils.constants import MAX_TEST_SLOT_INDEX
from utils.prepare_test_data import prepare_test
from utils.h5_util import load_h5_file

class EnhancedTrafficDataset(Dataset):
    """Enhanced traffic dataset with flexible filtering and transforms"""
    
    def __init__(self,
                 root_dir: str,
                 cities: Optional[List[str]] = None,
                 years: Optional[List[str]] = None,
                 file_pattern: str = "**/training/*8ch.h5",
                 transform: Optional[Callable] = None,
                 limit: Optional[int] = None,
                 input_timesteps: int = 12,
                 output_timesteps: int = 6,
                 return_metadata: bool = False):
        """
        Enhanced dataset for Traffic4Cast data
        
        Args:
            root_dir: Root directory containing traffic data
            cities: List of cities to include (None = all cities)
            years: List of years to include (None = all years)  
            file_pattern: Pattern to match data files
            transform: Optional transform to apply to data
            limit: Maximum number of samples (None = no limit)
            input_timesteps: Number of input time steps (default: 12)
            output_timesteps: Number of output time steps (default: 6)
            return_metadata: Whether to return metadata for multi-task learning
        """
        self.root_dir = Path(root_dir)
        self.cities = cities
        self.years = years
        self.file_pattern = file_pattern
        self.transform = transform
        self.limit = limit
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.return_metadata = return_metadata
        
        # Load and filter files
        self.files = self._load_and_filter_files()
        logging.info(f"Found {len(self.files)} files matching criteria")
        
        # Calculate total samples
        self._calculate_dataset_size()
        
        # For multi-task learning
        self.city_labels = {}
        self.year_labels = {}
        if self.return_metadata:
            self._build_label_mappings()
        
    def _load_and_filter_files(self) -> List[Path]:
        """Load and filter files based on cities and years"""
        all_files = list(self.root_dir.rglob(self.file_pattern))
        filtered_files = []
        
        for file_path in all_files:
            # Check city filter
            if self.cities is not None:
                city_match = any(city in str(file_path) for city in self.cities)
                if not city_match:
                    continue
            
            # Check year filter  
            if self.years is not None:
                year_match = any(year in str(file_path) for year in self.years)
                if not year_match:
                    continue
                    
            filtered_files.append(file_path)
            
        return sorted(filtered_files)
    
    def _calculate_dataset_size(self):
        """Calculate total number of samples in dataset"""
        # Each file contains MAX_TEST_SLOT_INDEX possible samples
        total_samples = len(self.files) * MAX_TEST_SLOT_INDEX
        
        if self.limit is not None:
            total_samples = min(total_samples, self.limit)
            
        self.total_samples = total_samples
    
    def _build_label_mappings(self):
        """Build label mappings for multi-task learning"""
        cities = set()
        years = set()
        
        for file_path in self.files:
            city = self.get_city_from_file(file_path)
            year = self.get_year_from_file(file_path)
            cities.add(city)
            years.add(year)
        
        # Create label mappings
        self.city_labels = {city: idx for idx, city in enumerate(sorted(cities))}
        self.year_labels = {year: idx for idx, year in enumerate(sorted(years))}
        
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int):
        """Get a single sample"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Calculate which file and time slot
        file_idx = idx // MAX_TEST_SLOT_INDEX
        time_slot = idx % MAX_TEST_SLOT_INDEX
        
        # Load data for this sample
        file_path = self.files[file_idx]
        
        # Load enough timesteps for input + output
        total_timesteps = self.input_timesteps + self.output_timesteps
        data = load_h5_file(str(file_path), 
                           sl=slice(time_slot, time_slot + total_timesteps))
        
        # Check if we have enough data
        if data.shape[0] < total_timesteps:
            # Handle edge case - might need to pad or skip
            logging.warning(f"Not enough timesteps in {file_path} at slot {time_slot}")
            # For now, just repeat the last frame
            needed = total_timesteps - data.shape[0]
            padding = np.repeat(data[-1:], needed, axis=0)
            data = np.concatenate([data, padding], axis=0)
        
        # Split into input and output
        input_data = data[:self.input_timesteps]
        output_data = data[self.input_timesteps:self.input_timesteps + self.output_timesteps]
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            output_tensor = self.transform(output_tensor)
        
        # Return with or without metadata
        if self.return_metadata:
            city = self.get_city_from_file(file_path)
            year = self.get_year_from_file(file_path)
            metadata = {
                'city': city,
                'year': year,
                'city_label': self.city_labels.get(city, 0),
                'year_label': self.year_labels.get(year, 0),
                'file_path': str(file_path)
            }
            return input_tensor, output_tensor, metadata
        else:
            return input_tensor, output_tensor
    
    def get_city_from_file(self, file_path: Path) -> str:
        """Extract city name from file path"""
        # Assuming file structure like: .../CITY/training/date_CITY_8ch.h5
        parts = file_path.parts
        for part in parts:
            if part in ['BARCELONA', 'MELBOURNE', 'NEWYORK', 'CHICAGO', 
                       'ANTWERP', 'VIENNA', 'BERLIN', 'BANGKOK', 'MOSCOW']:
                return part
        return "UNKNOWN"
    
    def get_year_from_file(self, file_path: Path) -> str:
        """Extract year from file path"""
        # Assuming filename like: 2019-01-01_CITY_8ch.h5
        filename = file_path.name
        if filename.startswith('2019'):
            return '2019'
        elif filename.startswith('2020'):
            return '2020'
        return "UNKNOWN"
    
    def get_file_info(self) -> List[dict]:
        """Get information about all files in dataset"""
        info = []
        for file_path in self.files:
            info.append({
                'path': str(file_path),
                'city': self.get_city_from_file(file_path),
                'year': self.get_year_from_file(file_path),
                'samples': MAX_TEST_SLOT_INDEX
            })
        return info
    
    def get_num_cities(self) -> int:
        """Get number of unique cities"""
        return len(self.city_labels)
    
    def get_num_years(self) -> int:
        """Get number of unique years"""
        return len(self.year_labels)
    
    def get_city_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per city"""
        distribution = defaultdict(int)
        file_info = self.get_file_info()
        for info in file_info:
            distribution[info['city']] += info['samples']
        return dict(distribution)
    
    def get_year_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per year"""
        distribution = defaultdict(int)
        file_info = self.get_file_info()
        for info in file_info:
            distribution[info['year']] += info['samples']
        return dict(distribution)

class TrafficDataTransform:
    """Standard transforms for traffic data"""
    
    def __init__(self, stack_channels_on_time: bool = True, 
                 normalize: bool = False,
                 add_padding: Optional[Tuple[int, int, int, int]] = None):
        """
        Args:
            stack_channels_on_time: Whether to stack channels and time
            normalize: Whether to normalize data to [0, 1]
            add_padding: Padding to add (left, right, top, bottom)
        """
        self.stack_channels_on_time = stack_channels_on_time
        self.normalize = normalize
        self.add_padding = add_padding
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply transforms to data"""
        # Normalize if requested
        if self.normalize:
            data = data / 255.0
            
        # Stack channels on time dimension for UNet compatibility
        if self.stack_channels_on_time:
            # Input shape: (T, H, W, C) -> (T*C, H, W)
            T, H, W, C = data.shape
            data = data.permute(0, 3, 1, 2)  # (T, C, H, W)
            data = data.reshape(T * C, H, W)
            
        # Add padding if requested
        if self.add_padding is not None:
            left, right, top, bottom = self.add_padding
            data = torch.nn.functional.pad(data, (left, right, top, bottom))
            
        return data

def create_data_loaders(config, train_dataset, val_dataset, test_dataset):
    """Create data loaders for training, validation, and testing"""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def get_dataset_stats(dataset: EnhancedTrafficDataset) -> dict:
    """Get statistics about the dataset"""
    file_info = dataset.get_file_info()
    
    cities = [info['city'] for info in file_info]
    years = [info['year'] for info in file_info]
    
    stats = {
        'total_files': len(file_info),
        'total_samples': len(dataset),
        'cities': list(set(cities)),
        'years': list(set(years)),
        'city_counts': {city: cities.count(city) for city in set(cities)},
        'year_counts': {year: years.count(year) for year in set(years)}
    }
    
    return stats
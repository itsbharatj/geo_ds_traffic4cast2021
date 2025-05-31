# src/data/enhanced_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Any
import logging
from datetime import datetime

# Import existing utilities (modify paths as needed)
from utils.constants import MAX_TEST_SLOT_INDEX
from utils.prepare_test_data import prepare_test
from utils.h5_util import load_h5_file

# Import our enhanced data layout
from data.enhanced_data_layout import TrafficStatisticsCalculator, create_enhanced_data_array

class EnhancedTrafficDataset(Dataset):
    """Enhanced traffic dataset with extra temporal channels"""
    
    def __init__(self,
                 root_dir: str,
                 cities: Optional[List[str]] = None,
                 years: Optional[List[str]] = None,
                 file_pattern: str = "**/training/*8ch.h5",
                 transform: Optional[Callable] = None,
                 limit: Optional[int] = None,
                 input_timesteps: int = 12,
                 output_timesteps: int = 6,
                 use_enhanced_channels: bool = True,
                 stats_window_days: int = 30):
        """
        Enhanced dataset for Traffic4Cast data with extra temporal channels
        
        Args:
            root_dir: Root directory containing traffic data
            cities: List of cities to include (None = all cities)
            years: List of years to include (None = all years)  
            file_pattern: Pattern to match data files
            transform: Optional transform to apply to data
            limit: Maximum number of samples (None = no limit)
            input_timesteps: Number of input time steps (default: 12)
            output_timesteps: Number of output time steps (default: 6)
            use_enhanced_channels: Whether to add enhanced temporal channels
            stats_window_days: Days to look back for computing statistics
        """
        self.root_dir = Path(root_dir)
        self.cities = cities
        self.years = years
        self.file_pattern = file_pattern
        self.transform = transform
        self.limit = limit
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.use_enhanced_channels = use_enhanced_channels
        self.stats_window_days = stats_window_days
        
        # Initialize statistics calculator if using enhanced channels
        if self.use_enhanced_channels:
            self.stats_calculator = TrafficStatisticsCalculator(str(self.root_dir))
        
        # Load and filter files
        self.files = self._load_and_filter_files()
        logging.info(f"Found {len(self.files)} files matching criteria")
        
        # Calculate total samples
        self._calculate_dataset_size()
        
        # Determine number of channels
        self.num_channels = 10 if self.use_enhanced_channels else 8
        
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
        
    def __len__(self) -> int:
        return self.total_samples
    
    def _parse_file_metadata(self, file_path: Path) -> Tuple[str, datetime]:
        """Parse city and date from file path"""
        filename = file_path.name
        parts = filename.split('_')
        
        # Extract date and city
        date_str = parts[0]
        city = parts[1] if len(parts) > 1 else "UNKNOWN"
        
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            logging.warning(f"Could not parse date from filename: {filename}")
            date = datetime(2019, 1, 1)  # Default date
            
        return city, date
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with enhanced channels"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Calculate which file and time slot
        file_idx = idx // MAX_TEST_SLOT_INDEX
        time_slot = idx % MAX_TEST_SLOT_INDEX
        
        # Load data for this sample
        file_path = self.files[file_idx]
        
        # Parse metadata from filename
        city, date = self._parse_file_metadata(file_path)
        
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
        
        # Add enhanced channels if requested
        if self.use_enhanced_channels:
            try:
                # Get enhanced channels for this time slot
                enhanced_channels = self.stats_calculator.get_enhanced_channels_for_slot(
                    city, date, time_slot, self.stats_window_days)
                
                # Create enhanced data array
                data = create_enhanced_data_array(data, enhanced_channels)
                
            except Exception as e:
                logging.warning(f"Could not compute enhanced channels for {file_path}: {e}")
                # Fallback: add zero channels
                T, H, W, C = data.shape
                enhanced_data = np.zeros((T, H, W, C + 2), dtype=data.dtype)
                enhanced_data[:, :, :, :C] = data
                data = enhanced_data
        
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
            
        return input_tensor, output_tensor
    
    def get_city_from_file(self, file_path: Path) -> str:
        """Extract city name from file path"""
        # Parse from filename
        city, _ = self._parse_file_metadata(file_path)
        return city
    
    def get_year_from_file(self, file_path: Path) -> str:
        """Extract year from file path"""
        # Parse from filename
        _, date = self._parse_file_metadata(file_path)
        return str(date.year)
    
    def get_file_info(self) -> List[dict]:
        """Get information about all files in dataset"""
        info = []
        for file_path in self.files:
            city, date = self._parse_file_metadata(file_path)
            info.append({
                'path': str(file_path),
                'city': city,
                'year': str(date.year),
                'date': date.strftime('%Y-%m-%d'),
                'samples': MAX_TEST_SLOT_INDEX
            })
        return info

class EnhancedTrafficDataTransform:
    """Enhanced transforms for traffic data with extra channels"""
    
    def __init__(self, stack_channels_on_time: bool = True, 
                 normalize: bool = False,
                 add_padding: Optional[Tuple[int, int, int, int]] = None,
                 num_channels: int = 10):
        """
        Args:
            stack_channels_on_time: Whether to stack channels and time
            normalize: Whether to normalize data to [0, 1]
            add_padding: Padding to add (left, right, top, bottom)
            num_channels: Number of channels (8 for original, 10 for enhanced)
        """
        self.stack_channels_on_time = stack_channels_on_time
        self.normalize = normalize
        self.add_padding = add_padding
        self.num_channels = num_channels
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply transforms to data"""
        # Normalize if requested (only the original 8 channels, not the enhanced ones)
        if self.normalize:
            if self.num_channels == 10:
                # Normalize only the first 8 channels
                data[:, :, :, :8] = data[:, :, :, :8] / 255.0
                # Enhanced channels are already normalized
            else:
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

def create_enhanced_data_loaders(config, train_dataset, val_dataset, test_dataset):
    """Create data loaders for training, validation, and testing with enhanced data"""
    
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

def get_enhanced_dataset_stats(dataset: EnhancedTrafficDataset) -> dict:
    """Get statistics about the enhanced dataset"""
    file_info = dataset.get_file_info()
    
    cities = [info['city'] for info in file_info]
    years = [info['year'] for info in file_info]
    
    stats = {
        'total_files': len(file_info),
        'total_samples': len(dataset),
        'cities': list(set(cities)),
        'years': list(set(years)),
        'city_counts': {city: cities.count(city) for city in set(cities)},
        'year_counts': {year: years.count(year) for year in set(years)},
        'num_channels': dataset.num_channels,
        'enhanced_channels': dataset.use_enhanced_channels,
        'stats_window_days': dataset.stats_window_days if dataset.use_enhanced_channels else None
    }
    
    return stats

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced dataset
    dataset = EnhancedTrafficDataset(
        root_dir="/path/to/traffic4cast/data",
        cities=["BANGKOK", "MELBOURNE"],
        years=["2019"],
        use_enhanced_channels=True,
        stats_window_days=30,
        limit=100  # For testing
    )
    
    # Create transform
    transform = EnhancedTrafficDataTransform(
        stack_channels_on_time=True,
        normalize=True,
        num_channels=10
    )
    dataset.transform = transform
    
    # Get a sample
    input_data, output_data = dataset[0]
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    
    # Get dataset statistics
    stats = get_enhanced_dataset_stats(dataset)
    print(f"Dataset stats: {stats}")
# src/data/splitter.py
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from .dataset import EnhancedTrafficDataset, TrafficDataTransform, get_dataset_stats

class TrafficDataSplitter:
    """Handles data splitting for Traffic4Cast experiments"""
    
    def __init__(self, root_dir: str, random_seed: int = 42):
        self.root_dir = root_dir
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def spatial_transfer_split(self, 
                              train_cities: List[str],
                              test_cities: List[str], 
                              val_fraction: float = 0.1,
                              years: Optional[List[str]] = None,
                              transform: Optional[TrafficDataTransform] = None,
                              limit_per_split: Optional[int] = None) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """
        Create spatial transfer split: train on some cities, test on others
        
        Args:
            train_cities: Cities to use for training
            test_cities: Cities to use for testing
            val_fraction: Fraction of training cities to use for validation
            years: Years to include (None = all years)
            transform: Transform to apply to data
            limit_per_split: Limit samples per split for debugging
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logging.info(f"Creating spatial transfer split:")
        logging.info(f"  Train cities: {train_cities}")
        logging.info(f"  Test cities: {test_cities}")
        logging.info(f"  Years: {years}")
        
        # Create test dataset
        test_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=test_cities,
            years=years,
            transform=transform,
            limit=limit_per_split
        )
        
        # Create training dataset with all training cities
        full_train_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=train_cities,
            years=years,
            transform=transform,
            limit=limit_per_split
        )
        
        # Split training cities into train/val if multiple cities
        if len(train_cities) > 1 and val_fraction > 0:
            # Calculate how many cities for validation
            n_val_cities = max(1, int(len(train_cities) * val_fraction))
            
            # Randomly select validation cities
            val_cities = np.random.choice(train_cities, n_val_cities, replace=False).tolist()
            actual_train_cities = [city for city in train_cities if city not in val_cities]
            
            logging.info(f"  Actual train cities: {actual_train_cities}")
            logging.info(f"  Validation cities: {val_cities}")
            
            # Create separate datasets
            train_dataset = EnhancedTrafficDataset(
                root_dir=self.root_dir,
                cities=actual_train_cities,
                years=years,
                transform=transform,
                limit=limit_per_split
            )
            
            val_dataset = EnhancedTrafficDataset(
                root_dir=self.root_dir,
                cities=val_cities,
                years=years,
                transform=transform,
                limit=limit_per_split
            )
        else:
            # Use same cities for train/val but split samples
            train_dataset, val_dataset = self._split_dataset_samples(
                full_train_dataset, val_fraction
            )
        
        # Log dataset statistics
        self._log_split_stats(train_dataset, val_dataset, test_dataset)
        
        return train_dataset, val_dataset, test_dataset
    
    def spatiotemporal_transfer_split(self,
                                    cities: List[str],
                                    train_years: List[str],
                                    test_years: List[str],
                                    val_fraction: float = 0.1,
                                    transform: Optional[TrafficDataTransform] = None,
                                    limit_per_split: Optional[int] = None) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """
        Create spatio-temporal transfer split: train on certain years, test on others
        
        Args:
            cities: Cities to include
            train_years: Years to use for training
            test_years: Years to use for testing
            val_fraction: Fraction of training data to use for validation
            transform: Transform to apply to data
            limit_per_split: Limit samples per split for debugging
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logging.info(f"Creating spatio-temporal transfer split:")
        logging.info(f"  Cities: {cities}")
        logging.info(f"  Train years: {train_years}")
        logging.info(f"  Test years: {test_years}")
        
        # Create test dataset
        test_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=cities,
            years=test_years,
            transform=transform,
            limit=limit_per_split
        )
        
        # Create full training dataset
        full_train_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=cities,
            years=train_years,
            transform=transform,
            limit=limit_per_split
        )
        
        # Split training data into train/val
        train_dataset, val_dataset = self._split_dataset_samples(
            full_train_dataset, val_fraction
        )
        
        # Log dataset statistics
        self._log_split_stats(train_dataset, val_dataset, test_dataset)
        
        return train_dataset, val_dataset, test_dataset
    
    def enhanced_spatiotemporal_transfer_split(self,
                                             train_cities: List[str],
                                             train_years: List[str],
                                             test_city: str,
                                             test_train_year: str,
                                             test_target_year: str,
                                             adaptation_samples: int = 100,
                                             val_fraction: float = 0.1,
                                             transform: Optional[TrafficDataTransform] = None,
                                             limit_per_split: Optional[int] = None) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """
        Create enhanced spatio-temporal transfer split for few-shot adaptation
        
        Args:
            train_cities: Cities to use for training (multi-task)
            train_years: Years to use for training
            test_city: Target city for transfer (not in train_cities)
            test_train_year: Year from test city for adaptation
            test_target_year: Target year to predict in test city
            adaptation_samples: Number of samples from test city for adaptation
            val_fraction: Fraction of training data for validation
            transform: Transform to apply to data
            limit_per_split: Limit samples per split for debugging
            
        Returns:
            Tuple of (train_dataset, adapt_dataset, test_dataset)
        """
        logging.info(f"Creating enhanced spatio-temporal transfer split:")
        logging.info(f"  Train cities: {train_cities}")
        logging.info(f"  Train years: {train_years}")
        logging.info(f"  Test city: {test_city}")
        logging.info(f"  Test train year: {test_train_year}")
        logging.info(f"  Test target year: {test_target_year}")
        logging.info(f"  Adaptation samples: {adaptation_samples}")
        
        # Validate that test city is not in training cities
        if test_city in train_cities:
            raise ValueError(f"Test city '{test_city}' cannot be in training cities for spatio-temporal transfer")
        
        # 1. Multi-task training dataset (multiple cities and years)
        train_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=train_cities,
            years=train_years,
            transform=transform,
            limit=limit_per_split,
            return_metadata=True  # Enable multi-task learning
        )
        
        # 2. Adaptation dataset (test city, source year - limited samples)
        full_adapt_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=[test_city],
            years=[test_train_year],
            transform=transform,
            return_metadata=True
        )
        
        # Sample limited adaptation data
        adapt_dataset = self._sample_limited_dataset(full_adapt_dataset, adaptation_samples)
        
        # 3. Test dataset (test city, target year)
        test_dataset = EnhancedTrafficDataset(
            root_dir=self.root_dir,
            cities=[test_city],
            years=[test_target_year],
            transform=transform,
            limit=limit_per_split,
            return_metadata=True
        )
        
        # 4. Validation dataset (subset of training data)
        if val_fraction > 0:
            train_dataset, val_dataset = self._split_dataset_samples(train_dataset, val_fraction)
        else:
            val_dataset = adapt_dataset  # Use adaptation data as validation
        
        # Log enhanced dataset statistics
        self._log_enhanced_split_stats(train_dataset, adapt_dataset, test_dataset, val_dataset)
        
        return train_dataset, adapt_dataset, test_dataset
    
    def _sample_limited_dataset(self, dataset: EnhancedTrafficDataset, num_samples: int):
        """Create a limited sample dataset for few-shot learning"""
        if len(dataset) <= num_samples:
            return dataset
        
        # Create a new dataset with limited samples
        limited_dataset = EnhancedTrafficDataset(
            root_dir=dataset.root_dir,
            cities=dataset.cities,
            years=dataset.years,
            transform=dataset.transform,
            limit=num_samples,
            return_metadata=dataset.return_metadata
        )
        
        return limited_dataset
    
    def _split_dataset_samples(self, dataset: EnhancedTrafficDataset, 
                              val_fraction: float) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """Split a single dataset into train/val by sampling"""
        
        if val_fraction <= 0:
            # No validation split
            return dataset, dataset
        
        # Get all file paths
        all_files = dataset.files
        
        # Split files into train/val
        train_files, val_files = train_test_split(
            all_files, 
            test_size=val_fraction,
            random_state=self.random_seed
        )
        
        # Create new datasets with split files
        train_dataset = EnhancedTrafficDataset(
            root_dir=dataset.root_dir,
            cities=dataset.cities,
            years=dataset.years,
            transform=dataset.transform,
            limit=dataset.limit,
            return_metadata=dataset.return_metadata
        )
        train_dataset.files = train_files
        train_dataset._calculate_dataset_size()
        
        val_dataset = EnhancedTrafficDataset(
            root_dir=dataset.root_dir,
            cities=dataset.cities,
            years=dataset.years,
            transform=dataset.transform,
            limit=dataset.limit,
            return_metadata=dataset.return_metadata
        )
        val_dataset.files = val_files
        val_dataset._calculate_dataset_size()
        
        return train_dataset, val_dataset
    
    def _log_split_stats(self, train_dataset, val_dataset, test_dataset):
        """Log statistics about the data splits"""
        
        train_stats = get_dataset_stats(train_dataset)
        val_stats = get_dataset_stats(val_dataset)
        test_stats = get_dataset_stats(test_dataset)
        
        logging.info("Dataset Split Statistics:")
        logging.info(f"  Training: {train_stats['total_samples']} samples from {train_stats['total_files']} files")
        logging.info(f"    Cities: {train_stats['city_counts']}")
        logging.info(f"    Years: {train_stats['year_counts']}")
        
        logging.info(f"  Validation: {val_stats['total_samples']} samples from {val_stats['total_files']} files")
        logging.info(f"    Cities: {val_stats['city_counts']}")
        logging.info(f"    Years: {val_stats['year_counts']}")
        
        logging.info(f"  Testing: {test_stats['total_samples']} samples from {test_stats['total_files']} files")
        logging.info(f"    Cities: {test_stats['city_counts']}")  
        logging.info(f"    Years: {test_stats['year_counts']}")
    
    def _log_enhanced_split_stats(self, train_dataset, adapt_dataset, test_dataset, val_dataset):
        """Log statistics for enhanced spatio-temporal splits"""
        
        train_stats = get_dataset_stats(train_dataset)
        adapt_stats = get_dataset_stats(adapt_dataset)
        test_stats = get_dataset_stats(test_dataset)
        val_stats = get_dataset_stats(val_dataset)
        
        logging.info("Enhanced Spatio-Temporal Split Statistics:")
        logging.info(f"  Training (Multi-task): {train_stats['total_samples']} samples")
        logging.info(f"    Cities: {train_stats['city_counts']}")
        logging.info(f"    Years: {train_stats['year_counts']}")
        
        logging.info(f"  Validation: {val_stats['total_samples']} samples")
        
        logging.info(f"  Adaptation (Few-shot): {adapt_stats['total_samples']} samples")
        logging.info(f"    Cities: {adapt_stats['city_counts']}")
        logging.info(f"    Years: {adapt_stats['year_counts']}")
        
        logging.info(f"  Testing (Transfer): {test_stats['total_samples']} samples")
        logging.info(f"    Cities: {test_stats['city_counts']}")
        logging.info(f"    Years: {test_stats['year_counts']}")

class ExperimentDataManager:
    """High-level manager for experiment data splits"""
    
    def __init__(self, config):
        self.config = config
        self.splitter = TrafficDataSplitter(
            root_dir=config.data.root_dir,
            random_seed=config.experiment.random_seed
        )
        
        # Create transform
        self.transform = TrafficDataTransform(
            stack_channels_on_time=True,
            normalize=False,  # Keep as uint8 for now
            add_padding=None   # No padding by default
        )
    
    def setup_spatial_transfer(self) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """Setup data for spatial transfer experiment"""
        return self.splitter.spatial_transfer_split(
            train_cities=self.config.experiment.train_cities,
            test_cities=self.config.experiment.test_cities,
            val_fraction=self.config.experiment.val_fraction,
            years=getattr(self.config.experiment, 'years', None),
            transform=self.transform,
            limit_per_split=getattr(self.config.data, 'limit_per_split', None)
        )
    
    def setup_spatiotemporal_transfer(self) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """Setup data for spatio-temporal transfer experiment"""
        return self.splitter.spatiotemporal_transfer_split(
            cities=self.config.data.cities,
            train_years=self.config.experiment.train_years,
            test_years=self.config.experiment.test_years,
            val_fraction=self.config.experiment.val_fraction,
            transform=self.transform,
            limit_per_split=getattr(self.config.data, 'limit_per_split', None)
        )
    
    def setup_enhanced_spatiotemporal_transfer(self) -> Tuple[EnhancedTrafficDataset, EnhancedTrafficDataset, EnhancedTrafficDataset]:
        """Setup data for enhanced spatio-temporal transfer experiment"""
        return self.splitter.enhanced_spatiotemporal_transfer_split(
            train_cities=self.config.experiment.train_cities,
            train_years=self.config.experiment.train_years,
            test_city=self.config.experiment.test_city,
            test_train_year=self.config.experiment.test_train_year,
            test_target_year=self.config.experiment.test_target_year,
            adaptation_samples=getattr(self.config.experiment, 'adaptation_samples', 100),
            val_fraction=self.config.experiment.val_fraction,
            transform=self.transform,
            limit_per_split=getattr(self.config.data, 'limit_per_split', None)
        )
    
    def setup_experiment_data(self):
        """Setup data based on experiment type"""
        if self.config.experiment.type == "spatial_transfer":
            return self.setup_spatial_transfer()
        elif self.config.experiment.type == "spatiotemporal_transfer":
            return self.setup_spatiotemporal_transfer()
        elif self.config.experiment.type == "enhanced_spatiotemporal_transfer":
            return self.setup_enhanced_spatiotemporal_transfer()
        else:
            raise ValueError(f"Unknown experiment type: {self.config.experiment.type}")

# Utility functions for data analysis
def analyze_city_distribution(root_dir: str) -> Dict[str, int]:
    """Analyze distribution of data across cities"""
    
    all_files = list(Path(root_dir).rglob("**/training/*8ch.h5"))
    city_counts = {}
    
    for file_path in all_files:
        # Extract city from path
        for part in file_path.parts:
            if part in ['BARCELONA', 'MELBOURNE', 'NEWYORK', 'CHICAGO', 
                       'ANTWERP', 'VIENNA', 'BERLIN', 'BANGKOK', 'MOSCOW']:
                city_counts[part] = city_counts.get(part, 0) + 1
                break
    
    return city_counts

def analyze_year_distribution(root_dir: str) -> Dict[str, int]:
    """Analyze distribution of data across years"""
    
    all_files = list(Path(root_dir).rglob("**/training/*8ch.h5"))
    year_counts = {}
    
    for file_path in all_files:
        filename = file_path.name
        if filename.startswith('2019'):
            year_counts['2019'] = year_counts.get('2019', 0) + 1
        elif filename.startswith('2020'):
            year_counts['2020'] = year_counts.get('2020', 0) + 1
    
    return year_counts

import os
import re
import logging
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import SubsetRandomSampler

from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import T4CGeometricDataset


class SplitType(Enum):
    """Enumeration of possible data splitting strategies."""
    TIME_BASED = "time_based"
    CROSS_CITY = "cross_city"
    RANDOM = "random"  # Default original behavior


class CustomDataSplitter:
    """
    Custom data splitter that handles time-based and city-based splits for T4C dataset.
    """
    
    def __init__(
        self, 
        dataset: Union[T4CDataset, T4CGeometricDataset],
        split_type: SplitType = SplitType.RANDOM,
        test_city: Optional[str] = None,
        test_year: str = "2020",
        train_year: str = "2019",
        test_fraction: float = 0.2,
        val_fraction: float = 0.1,
        random_seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        dataloader_config: dict = None,
        geometric: bool = False,
        limit: Optional[int] = None,
    ):
        """
        Initialize the data splitter.
        
        Parameters
        ----------
        dataset : Union[T4CDataset, T4CGeometricDataset]
            The dataset to split
        split_type : SplitType
            Type of split to use (time_based, cross_city, or random)
        test_city : Optional[str]
            City to use for testing in cross_city mode (e.g., "BARCELONA")
        test_year : str
            Year to use for testing in time_based mode (default: "2020")
        train_year : str 
            Year to use for training in time_based mode (default: "2019")
        test_fraction : float
            Fraction of data to use for testing
        val_fraction : float
            Fraction of data to use for validation
        random_seed : int
            Random seed for reproducibility
        batch_size : int
            Batch size for data loaders
        num_workers : int
            Number of workers for data loaders
        dataloader_config : dict
            Additional configuration for data loaders
        geometric : bool
            Whether dataset is geometric
        limit : Optional[int]
            Maximum number of samples to use
        """
        self.dataset = dataset
        self.split_type = split_type
        self.test_city = test_city
        self.test_year = test_year
        self.train_year = train_year
        self.test_fraction = test_fraction
        self.val_fraction = val_fraction
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader_config = dataloader_config or {}
        self.geometric = geometric
        self.limit = limit
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Get file paths and indices
        self.file_paths = self._get_file_paths()
        self.train_indices, self.val_indices, self.test_indices = self._split_indices()
        
        logging.info(f"Split type: {self.split_type.value}")
        logging.info(f"Train set size: {len(self.train_indices)}")
        logging.info(f"Validation set size: {len(self.val_indices)}")
        logging.info(f"Test set size: {len(self.test_indices)}")
        
    def _get_file_paths(self) -> List[str]:
        """Get all file paths from the dataset."""
        logging.info("Checking dataset attributes for file paths...")
        if hasattr(self.dataset, 'file_list'):
            logging.info(f"Dataset has attribute 'file_list': {self.dataset.file_list}")
            if self.dataset.file_list:
                return self.dataset.file_list
        if hasattr(self.dataset, 'h5_files'):
            logging.info(f"Dataset has attribute 'h5_files': {self.dataset.h5_files}")
            if self.dataset.h5_files:
                return self.dataset.h5_files
        logging.error("Dataset does not contain valid file paths.")
        raise ValueError("Cannot determine file paths from dataset")
    
    def _split_indices(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset indices according to chosen strategy.
        
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            Indices for train, validation, and test sets
        """
        full_dataset_size = len(self.dataset)
        effective_dataset_size = min(full_dataset_size, self.limit) if self.limit else full_dataset_size
        
        if self.split_type == SplitType.TIME_BASED:
            return self._time_based_split(effective_dataset_size)
        elif self.split_type == SplitType.CROSS_CITY:
            return self._cross_city_split(effective_dataset_size)
        else:  # Random split (original behavior)
            return self._random_split(effective_dataset_size)
    
    def _time_based_split(self, effective_dataset_size: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Split based on time (year), using one year for training and another for testing.
        
        Parameters
        ----------
        effective_dataset_size : int
            Maximum number of samples to use
            
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            Indices for train, validation, and test sets
        """
        train_indices = []
        test_indices = []
        
        # Categorize files by year
        for idx, file_path in enumerate(self.file_paths):
            if idx >= effective_dataset_size:
                break
                
            file_path_str = str(file_path)
            if self.train_year in file_path_str:
                train_indices.append(idx)
            elif self.test_year in file_path_str:
                test_indices.append(idx)
        
        # Shuffle indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        # Split training indices into train and validation
        val_size = int(len(train_indices) * self.val_fraction / (1 - self.test_fraction))
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        return train_indices, val_indices, test_indices
    
    def _cross_city_split(self, effective_dataset_size: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Split based on city, using one city for testing and others for training.
        
        Parameters
        ----------
        effective_dataset_size : int
            Maximum number of samples to use
            
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            Indices for train, validation, and test sets
        """
        if not self.test_city:
            raise ValueError("test_city must be specified for cross_city split")
        
        train_indices = []
        test_indices = []
        
        # Categorize files by city
        for idx, file_path in enumerate(self.file_paths):
            if idx >= effective_dataset_size:
                break
                
            file_path_str = str(file_path)
            if self.test_city in file_path_str:
                test_indices.append(idx)
            else:
                train_indices.append(idx)
        
        # Shuffle indices
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        # Split training indices into train and validation
        val_size = int(len(train_indices) * self.val_fraction / (1 - self.test_fraction))
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        return train_indices, val_indices, test_indices
    
    def _random_split(self, effective_dataset_size: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Perform a random split of the dataset.
        
        Parameters
        ----------
        effective_dataset_size : int
            Maximum number of samples to use
            
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            Indices for train, validation, and test sets
        """
        indices = list(range(effective_dataset_size))
        np.random.shuffle(indices)
        
        test_size = int(effective_dataset_size * self.test_fraction)
        val_size = int(effective_dataset_size * self.val_fraction)
        train_size = effective_dataset_size - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return train_indices, val_indices, test_indices
    
    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """
        Create data loaders for train, validation, and test sets.
        
        Returns
        -------
        Dict[str, DataLoader]
            Dictionary with train, val, and test data loaders
        """
        if self.geometric:
            # For geometric datasets
            train_set = Subset(self.dataset, self.train_indices)
            val_set = Subset(self.dataset, self.val_indices)
            test_set = Subset(self.dataset, self.test_indices)
            
            train_loader = torch.geometric.data.DataLoader(
                train_set, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                **self.dataloader_config
            )
            val_loader = torch.geometric.data.DataLoader(
                val_set, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                **self.dataloader_config
            )
            test_loader = torch.geometric.data.DataLoader(
                test_set, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                **self.dataloader_config
            )
        else:
            # For regular datasets
            train_sampler = SubsetRandomSampler(self.train_indices)
            val_sampler = SubsetRandomSampler(self.val_indices)
            test_sampler = SubsetRandomSampler(self.test_indices)
            
            train_loader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                sampler=train_sampler, 
                **self.dataloader_config
            )
            val_loader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                sampler=val_sampler, 
                **self.dataloader_config
            )
            test_loader = DataLoader(
                self.dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers, 
                sampler=test_sampler, 
                **self.dataloader_config
            )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
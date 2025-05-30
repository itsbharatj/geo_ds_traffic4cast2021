# tests/test_data.py
"""
Tests for data loading and processing
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import h5py

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import EnhancedTrafficDataset, TrafficDataTransform
from data.splitter import TrafficDataSplitter
from utils.config import Config

class TestEnhancedTrafficDataset:
    
    def create_dummy_data(self, temp_dir, cities=["CITY1", "CITY2"], years=["2019", "2020"]):
        """Create dummy traffic data for testing"""
        
        temp_path = Path(temp_dir)
        
        for city in cities:
            for year in years:
                city_dir = temp_path / city / "training"
                city_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy H5 file
                filename = f"{year}-01-01_{city}_8ch.h5"
                filepath = city_dir / filename
                
                # Create dummy data (288 time slots, 495x436 spatial, 8 channels)
                dummy_data = np.random.randint(0, 256, size=(288, 495, 436, 8), dtype=np.uint8)
                
                with h5py.File(filepath, 'w') as f:
                    f.create_dataset('array', data=dummy_data, compression='lzf')
        
        return temp_path
    
    def test_dataset_creation(self):
        """Test basic dataset creation"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_dummy_data(temp_dir)
            
            dataset = EnhancedTrafficDataset(
                root_dir=str(data_path),
                cities=["CITY1"],
                years=["2019"],
                limit=10
            )
            
            assert len(dataset) == 10
            
            # Test getting an item
            inputs, targets = dataset[0]
            assert inputs.shape == (12, 495, 436, 8)  # 12 input timesteps
            assert targets.shape == (6, 495, 436, 8)   # 6 output timesteps
    
    def test_city_filtering(self):
        """Test filtering by cities"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_dummy_data(temp_dir, cities=["CITY1", "CITY2"])
            
            # Test with city filter
            dataset = EnhancedTrafficDataset(
                root_dir=str(data_path),
                cities=["CITY1"],
                limit=10
            )
            
            file_info = dataset.get_file_info()
            cities = [info['city'] for info in file_info]
            assert all(city == "CITY1" for city in cities)
    
    def test_year_filtering(self):
        """Test filtering by years"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = self.create_dummy_data(temp_dir, years=["2019", "2020"])
            
            # Test with year filter
            dataset = EnhancedTrafficDataset(
                root_dir=str(data_path),
                years=["2019"],
                limit=10
            )
            
            file_info = dataset.get_file_info()
            years = [info['year'] for info in file_info]
            assert all(year == "2019" for year in years)

class TestTrafficDataTransform:
    
    def test_transform_stacking(self):
        """Test channel stacking transform"""
        
        transform = TrafficDataTransform(stack_channels_on_time=True)
        
        # Create dummy data (T, H, W, C)
        data = torch.randint(0, 256, size=(12, 100, 100, 8))
        
        transformed = transform(data)
        
        # Should be (T*C, H, W)
        assert transformed.shape == (96, 100, 100)
    
    def test_transform_normalization(self):
        """Test normalization transform"""
        
        transform = TrafficDataTransform(normalize=True, stack_channels_on_time=False)
        
        # Create data with known values
        data = torch.full((12, 100, 100, 8), 255.0)
        
        transformed = transform(data)
        
        # Should be normalized to [0, 1]
        assert torch.allclose(transformed, torch.ones_like(transformed))

class TestTrafficDataSplitter:
    
    def test_spatial_split(self):
        """Test spatial transfer splitting"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data
            temp_path = Path(temp_dir)
            cities = ["CITY1", "CITY2", "CITY3"]
            
            for city in cities:
                city_dir = temp_path / city / "training"
                city_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"2019-01-01_{city}_8ch.h5"
                filepath = city_dir / filename
                
                dummy_data = np.random.randint(0, 256, size=(288, 495, 436, 8), dtype=np.uint8)
                
                with h5py.File(filepath, 'w') as f:
                    f.create_dataset('array', data=dummy_data, compression='lzf')
            
            # Test splitter
            splitter = TrafficDataSplitter(str(temp_path))
            
            train_dataset, val_dataset, test_dataset = splitter.spatial_transfer_split(
                train_cities=["CITY1", "CITY2"],
                test_cities=["CITY3"],
                limit_per_split=10
            )
            
            # Check that datasets have data
            assert len(train_dataset) > 0
            assert len(val_dataset) > 0
            assert len(test_dataset) > 0
            
            # Check that test dataset only has CITY3
            test_file_info = test_dataset.get_file_info()
            test_cities = [info['city'] for info in test_file_info]
            assert all(city == "CITY3" for city in test_cities)
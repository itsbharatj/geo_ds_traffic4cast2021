# tests/conftest.py
"""
Pytest configuration and fixtures
"""

import pytest
import tempfile
import numpy as np
import h5py
from pathlib import Path

@pytest.fixture
def dummy_traffic_data():
    """Create dummy traffic data for testing"""
    
    # Standard Traffic4Cast dimensions
    return np.random.randint(0, 256, size=(288, 495, 436, 8), dtype=np.uint8)

@pytest.fixture  
def temp_data_dir():
    """Create temporary directory with dummy traffic data"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create structure for multiple cities and years
        cities = ["BARCELONA", "MELBOURNE"]
        years = ["2019", "2020"]
        
        for city in cities:
            for year in years:
                city_dir = temp_path / city / "training"
                city_dir.mkdir(parents=True, exist_ok=True)
                
                # Create dummy H5 file
                filename = f"{year}-01-01_{city}_8ch.h5"
                filepath = city_dir / filename
                
                # Create dummy data
                dummy_data = np.random.randint(0, 256, size=(288, 495, 436, 8), dtype=np.uint8)
                
                with h5py.File(filepath, 'w') as f:
                    f.create_dataset('array', data=dummy_data, compression='lzf')
        
        yield temp_path
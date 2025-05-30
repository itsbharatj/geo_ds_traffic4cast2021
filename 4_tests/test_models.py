# tests/test_models.py
"""
Tests for model architectures
"""

import pytest
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.unet import UNet, TrafficUNet
from utils.config import Config

class TestUNet:
    
    def test_unet_creation(self):
        """Test UNet model creation"""
        
        model = UNet(in_channels=96, out_channels=48, features=[64, 128, 256])
        
        assert model.in_channels == 96
        assert model.out_channels == 48
        assert len(model.features) == 3
    
    def test_unet_forward(self):
        """Test UNet forward pass"""
        
        model = UNet(in_channels=96, out_channels=48, features=[32, 64])  # Small for testing
        
        # Test input
        x = torch.randn(2, 96, 64, 64)  # Small spatial size for testing
        
        output = model(x)
        
        assert output.shape == (2, 48, 64, 64)
    
    def test_traffic_unet(self):
        """Test TrafficUNet specific implementation"""
        
        model = TrafficUNet(
            input_timesteps=12,
            output_timesteps=6,
            data_channels=8,
            features=[32, 64]  # Small for testing
        )
        
        assert model.input_timesteps == 12
        assert model.output_timesteps == 6
        assert model.data_channels == 8
    
    def test_traffic_unet_forward_formats(self):
        """Test TrafficUNet with different input formats"""
        
        model = TrafficUNet(
            input_timesteps=12,
            output_timesteps=6,
            data_channels=8,
            features=[16, 32]  # Very small for testing
        )
        
        # Test 4D input (B, T*C, H, W)
        x_4d = torch.randn(1, 96, 32, 32)  # Small spatial size
        output_4d = model(x_4d)
        assert output_4d.shape == (1, 48, 32, 32)
        
        # Test 5D input (B, T, H, W, C)
        x_5d = torch.randn(1, 12, 32, 32, 8)
        output_5d = model(x_5d)
        assert output_5d.shape == (1, 48, 32, 32)
    
    def test_model_info(self):
        """Test model info extraction"""
        
        model = UNet(in_channels=96, out_channels=48, features=[32, 64])
        
        info = model.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
        assert info['in_channels'] == 96
        assert info['out_channels'] == 48
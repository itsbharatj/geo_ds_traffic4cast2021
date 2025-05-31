# src/models/multitask_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

from .unet import DoubleConv, Down, Up, OutConv

class MultiTaskUNet(nn.Module):
    """
    Multi-task UNet for enhanced spatio-temporal transfer learning.
    
    Extends standard UNet with:
    - Multi-task heads (traffic, city, year classification)
    - Attention mechanisms for adaptive feature fusion
    - Meta-learning components for few-shot adaptation
    """
    
    def __init__(self,
                 in_channels: int = 96,
                 out_channels: int = 48,
                 features: List[int] = None,
                 num_cities: int = 10,
                 num_years: int = 2,
                 bilinear: bool = False,
                 use_attention: bool = True,
                 use_meta_learning: bool = True):
        
        super(MultiTaskUNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_cities = num_cities
        self.num_years = num_years
        self.bilinear = bilinear
        self.use_attention = use_attention
        self.use_meta_learning = use_meta_learning
        
        # Shared encoder
        self.inc = DoubleConv(in_channels, features[0])
        
        self.down_layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_layers.append(Down(features[i], features[i + 1]))
        
        # Shared decoder for traffic prediction
        self.up_layers = nn.ModuleList()
        factor = 2 if bilinear else 1
        up_features = features[::-1]
        
        for i in range(len(up_features) - 1):
            self.up_layers.append(
                Up(up_features[i], up_features[i + 1] // factor, bilinear)
            )
        
        # Main traffic prediction head
        self.traffic_head = OutConv(features[0], out_channels)
        
        # Auxiliary task heads
        self.city_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_cities)
        )
        
        self.year_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features[-1], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_years)
        )
        
        # Attention mechanisms
        if self.use_attention:
            self.spatial_attention = SpatialAttentionModule(features[-1])
            self.temporal_attention = TemporalAttentionModule(features[-1])
            self.fusion_conv = nn.Conv2d(features[-1] * 2, features[-1], 1)
        
        # Meta-learning components
        if self.use_meta_learning:
            self.meta_adapter = MetaLearningAdapter(features[-1])
    
    def forward(self, x, metadata=None, mode="train"):
        """
        Forward pass with different modes for multi-task learning
        
        Args:
            x: Input tensor (B, C, H, W)
            metadata: Dict with city_label, year_label for multi-task learning
            mode: "train" (multi-task), "adapt" (few-shot), or "test"
        """
        
        # Shared encoder
        encoder_outputs = []
        
        # Initial convolution
        x = self.inc(x)
        encoder_outputs.append(x)
        
        # Encoder path
        for down_layer in self.down_layers:
            x = down_layer(x)
            encoder_outputs.append(x)
        
        # Bottleneck features
        bottleneck = encoder_outputs[-1]
        
        # Mode-specific processing
        if mode == "train":
            return self._forward_train(encoder_outputs, bottleneck, metadata)
        elif mode == "adapt":
            return self._forward_adapt(encoder_outputs, bottleneck)
        elif mode == "test":
            return self._forward_test(encoder_outputs, bottleneck)
        else:
            # Default: just traffic prediction
            return {'traffic': self._decode_traffic(encoder_outputs)}
    
    def _forward_train(self, encoder_outputs, bottleneck, metadata):
        """Multi-task training forward pass"""
        outputs = {}
        
        # Main traffic prediction
        outputs['traffic'] = self._decode_traffic(encoder_outputs)
        
        # Auxiliary tasks
        outputs['city'] = self.city_classifier(bottleneck)
        outputs['year'] = self.year_classifier(bottleneck)
        
        # Enhanced prediction with attention
        if self.use_attention:
            spatial_features = self.spatial_attention(bottleneck)
            temporal_features = self.temporal_attention(bottleneck)
            
            # Fuse attention features
            fused = torch.cat([spatial_features, temporal_features], dim=1)
            enhanced_bottleneck = self.fusion_conv(fused)
            
            # Enhanced traffic prediction
            enhanced_encoder_outputs = encoder_outputs[:-1] + [enhanced_bottleneck]
            outputs['enhanced_traffic'] = self._decode_traffic(enhanced_encoder_outputs)
        
        return outputs
    
    def _forward_adapt(self, encoder_outputs, bottleneck):
        """Few-shot adaptation forward pass"""
        outputs = {}
        
        # Standard traffic prediction
        outputs['traffic'] = self._decode_traffic(encoder_outputs)
        
        # Meta-learning adaptation
        if self.use_meta_learning:
            adapted_features = self.meta_adapter.adapt(bottleneck)
            adapted_outputs = encoder_outputs[:-1] + [adapted_features]
            outputs['adapted_traffic'] = self._decode_traffic(adapted_outputs)
        
        return outputs
    
    def _forward_test(self, encoder_outputs, bottleneck):
        """Test-time forward pass"""
        outputs = {}
        
        # Use adapted features if available
        if self.use_meta_learning and self.meta_adapter.has_adapted():
            adapted_features = self.meta_adapter.apply_adaptation(bottleneck)
            adapted_outputs = encoder_outputs[:-1] + [adapted_features]
            outputs['traffic'] = self._decode_traffic(adapted_outputs)
        else:
            outputs['traffic'] = self._decode_traffic(encoder_outputs)
        
        return outputs
    
    def _decode_traffic(self, encoder_outputs):
        """Standard UNet decoding for traffic prediction"""
        x = encoder_outputs[-1]  # Start with bottleneck
        skip_connections = encoder_outputs[:-1][::-1]  # Reverse skip connections
        
        for up_layer, skip in zip(self.up_layers, skip_connections):
            x = up_layer(x, skip)
        
        return self.traffic_head(x)
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MultiTaskUNet',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'features': self.features,
            'num_cities': self.num_cities,
            'num_years': self.num_years,
            'use_attention': self.use_attention,
            'use_meta_learning': self.use_meta_learning,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

class SpatialAttentionModule(nn.Module):
    """Spatial attention for city-specific patterns"""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, 1)
        self.conv2 = nn.Conv2d(channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = self.sigmoid(self.conv2(attention))
        return x * attention

class TemporalAttentionModule(nn.Module):
    """Temporal (channel) attention for year-specific patterns"""
    
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // 8)
        self.fc2 = nn.Linear(channels // 8, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MetaLearningAdapter(nn.Module):
    """Simple meta-learning adapter for few-shot adaptation"""
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.adaptation_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels)
        )
        self.adapted_params = None
    
    def adapt(self, features):
        """Learn adaptation parameters from support features"""
        self.adapted_params = self.adaptation_net(features)
        return features  # Return original features during adaptation
    
    def apply_adaptation(self, features):
        """Apply learned adaptation to new features"""
        if self.adapted_params is not None:
            b, c, h, w = features.shape
            adaptation = self.adapted_params.view(b, c, 1, 1)
            return features + 0.1 * adaptation  # Small adaptation
        return features
    
    def has_adapted(self):
        """Check if adaptation parameters are available"""
        return self.adapted_params is not None

def create_multitask_unet(config):
    """Factory function to create multi-task UNet from config"""
    
    # Get model parameters from config
    model_config = config.model
    experiment_config = config.experiment
    
    return MultiTaskUNet(
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels,
        features=model_config.features,
        num_cities=getattr(experiment_config, 'num_cities', 10),
        num_years=getattr(experiment_config, 'num_years', 2),
        bilinear=getattr(model_config, 'bilinear', False),
        use_attention=getattr(model_config, 'use_attention', True),
        use_meta_learning=getattr(model_config, 'use_meta_learning', True)
    )
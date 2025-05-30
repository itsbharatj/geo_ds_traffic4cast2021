# src/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DoubleConv(nn.Module):
    """Double convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Standard UNet implementation for Traffic4Cast
    
    Expected input: (batch_size, in_channels, height, width)
    where in_channels = input_timesteps * data_channels (e.g., 12 * 8 = 96)
    
    Output: (batch_size, out_channels, height, width)  
    where out_channels = output_timesteps * data_channels (e.g., 6 * 8 = 48)
    """
    
    def __init__(self, 
                 in_channels: int = 96,  # 12 timesteps * 8 channels
                 out_channels: int = 48,  # 6 timesteps * 8 channels
                 features: List[int] = None,
                 bilinear: bool = False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature dimensions for each level
            bilinear: Whether to use bilinear upsampling or transposed convolutions
        """
        super(UNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512, 1024]
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear
        
        # Initial convolution
        self.inc = DoubleConv(in_channels, features[0])
        
        # Encoder (downsampling path)
        self.down_layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_layers.append(Down(features[i], features[i + 1]))
        
        # Decoder (upsampling path)
        self.up_layers = nn.ModuleList()
        factor = 2 if bilinear else 1
        
        # Reverse features for upsampling
        up_features = features[::-1]
        for i in range(len(up_features) - 1):
            self.up_layers.append(
                Up(up_features[i], up_features[i + 1] // factor, bilinear)
            )
        
        # Output convolution
        self.outc = OutConv(features[0], out_channels)
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Initial convolution
        x = self.inc(x)
        encoder_outputs.append(x)
        
        # Encoder path
        for down_layer in self.down_layers:
            x = down_layer(x)
            encoder_outputs.append(x)
        
        # Decoder path (skip the last encoder output as it's the bottleneck)
        x = encoder_outputs[-1]  # Start with bottleneck
        skip_connections = encoder_outputs[:-1][::-1]  # Reverse skip connections
        
        for up_layer, skip in zip(self.up_layers, skip_connections):
            x = up_layer(x, skip)
        
        # Output layer
        x = self.outc(x)
        
        return x
    
    def get_model_info(self) -> dict:
        """Get information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'UNet',
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'features': self.features,
            'bilinear': self.bilinear,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class TrafficUNet(UNet):
    """
    UNet specifically configured for Traffic4Cast data
    
    This class provides preset configurations and helper methods
    for traffic prediction tasks.
    """
    
    def __init__(self, 
                 input_timesteps: int = 12,
                 output_timesteps: int = 6,
                 data_channels: int = 8,
                 features: List[int] = None,
                 bilinear: bool = False):
        """
        Args:
            input_timesteps: Number of input time steps
            output_timesteps: Number of output time steps
            data_channels: Number of channels per timestep (volume/speed for 4 directions)
            features: Feature dimensions for each level
            bilinear: Whether to use bilinear upsampling
        """
        
        in_channels = input_timesteps * data_channels
        out_channels = output_timesteps * data_channels
        
        if features is None:
            # Default features for traffic data
            features = [64, 128, 256, 512]
            
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels, 
            features=features,
            bilinear=bilinear
        )
        
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.data_channels = data_channels
        
    def forward(self, x):
        """
        Forward pass with input/output shape checking
        
        Args:
            x: Input tensor of shape (B, T*C, H, W) or (B, T, H, W, C)
            
        Returns:
            Output tensor of shape (B, T_out*C, H, W)
        """
        # Handle different input formats
        if x.dim() == 5:  # (B, T, H, W, C)
            B, T, H, W, C = x.shape
            x = x.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
            x = x.reshape(B, T * C, H, W)  # (B, T*C, H, W)
        elif x.dim() == 4:  # Already in correct format (B, T*C, H, W)
            pass
        else:
            raise ValueError(f"Expected input with 4 or 5 dimensions, got {x.dim()}")
            
        # Forward pass
        output = super().forward(x)
        
        return output
    
    def predict_sequence(self, x):
        """
        Predict and return output in sequence format
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor of shape (B, T_out, H, W, C)
        """
        output = self.forward(x)  # (B, T_out*C, H, W)
        
        B, TC, H, W = output.shape
        T_out = self.output_timesteps
        C = self.data_channels
        
        # Reshape to sequence format
        output = output.reshape(B, T_out, C, H, W)
        output = output.permute(0, 1, 3, 4, 2)  # (B, T_out, H, W, C)
        
        return output

def create_unet_model(config) -> UNet:
    """Factory function to create UNet model from config"""
    
    if hasattr(config.model, 'input_timesteps'):
        # Use TrafficUNet with timestep configuration
        return TrafficUNet(
            input_timesteps=config.model.input_timesteps,
            output_timesteps=config.model.output_timesteps,
            data_channels=config.model.data_channels,
            features=config.model.features,
            bilinear=getattr(config.model, 'bilinear', False)
        )
    else:
        # Use standard UNet with channel configuration
        return UNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            features=config.model.features,
            bilinear=getattr(config.model, 'bilinear', False)
        )

# Model testing utilities
def test_model_shapes(model: UNet, input_shape: tuple = (1, 96, 495, 436)):
    """Test model with given input shape"""
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(input_shape)
        y = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        print(f"Model info: {model.get_model_info()}")
        
    return y.shape

if __name__ == "__main__":
    # Test the model
    print("Testing Standard UNet:")
    unet = UNet(in_channels=96, out_channels=48)
    test_model_shapes(unet)
    
    print("\nTesting Traffic UNet:")
    traffic_unet = TrafficUNet(input_timesteps=12, output_timesteps=6)
    test_model_shapes(traffic_unet)
"""Enhanced UNet implementation for Traffic4Cast with extra channels.

Based on the original UNet implementation with modifications to handle 
the additional temporal channels (10 channels total instead of 8).
"""
#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class EnhancedUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode="upconv",
        num_input_channels=10,  # Enhanced: 8 original + 2 extra
        num_output_channels=8   # Output still has 8 channels (original format)
    ):
        """
        Enhanced U-Net for Traffic4Cast with additional temporal channels
        
        Args:
            in_channels (int): number of input channels after stacking time+channels
            n_classes (int): number of output channels after stacking time+channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
            num_input_channels (int): Number of channels in input data (10 for enhanced)
            num_output_channels (int): Number of channels in output data (8 for original format)
        """
        super(EnhancedUNet, self).__init__()
        assert up_mode in ("upconv", "upsample")
        self.padding = padding
        self.depth = depth
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x, *args, **kwargs):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = torch.nn.functional.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):  # noqa
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        diff_y_target_size_ = diff_y + target_size[0]
        diff_x_target_size_ = diff_x + target_size[1]
        return layer[:, :, diff_y:diff_y_target_size_, diff_x:diff_x_target_size_]

    def forward(self, x, bridge):  # noqa
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class EnhancedUNetTransformer:
    """Enhanced transformer for UNet with 10 input channels and 8 output channels"""

    @staticmethod
    def unet_pre_transform(
        data: np.ndarray,
        zeropad2d: Optional[Tuple[int, int, int, int]] = None,
        stack_channels_on_time: bool = False,
        batch_dim: bool = False,
        from_numpy: bool = False,
        num_channels: int = 10,
        **kwargs
    ) -> torch.Tensor:
        """Transform enhanced data from dataset to be used by UNet:

        - put time and channels into one dimension
        - padding
        """
        if from_numpy:
            data = torch.from_numpy(data).float()

        if not batch_dim:
            data = torch.unsqueeze(data, 0)

        if stack_channels_on_time:
            data = EnhancedUNetTransformer.transform_stack_channels_on_time(
                data, batch_dim=True, num_channels=num_channels)
        if zeropad2d is not None:
            zeropad2d = torch.nn.ZeroPad2d(zeropad2d)
            data = zeropad2d(data)
        if not batch_dim:
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def unet_post_transform(
        data: torch.Tensor, 
        crop: Optional[Tuple[int, int, int, int]] = None, 
        stack_channels_on_time: bool = False, 
        batch_dim: bool = False, 
        num_channels: int = 8,  # Output has 8 channels
        **kwargs
    ) -> torch.Tensor:
        """Bring data from UNet back to dataset format:

        - separates common dimension for time and channels
        - cropping
        """
        if not batch_dim:
            data = torch.unsqueeze(data, 0)

        if crop is not None:
            _, _, height, width = data.shape
            left, right, top, bottom = crop
            right = width - right
            bottom = height - bottom
            data = data[:, :, top:bottom, left:right]
        if stack_channels_on_time:
            data = EnhancedUNetTransformer.transform_unstack_channels_on_time(
                data, batch_dim=True, num_channels=num_channels)
        if not batch_dim:
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def transform_stack_channels_on_time(data: torch.Tensor, batch_dim: bool = False, num_channels: int = 10):
        """
        Enhanced version that handles variable number of channels:
        `(k, 12, 495, 436, num_channels) -> (k, 12 * num_channels, 495, 436)`
        """

        if not batch_dim:
            # `(12, 495, 436, num_channels) -> (1, 12, 495, 436, num_channels)`
            data = torch.unsqueeze(data, 0)
        num_time_steps = data.shape[1]

        # (k, 12, 495, 436, num_channels) -> (k, 12, num_channels, 495, 436)
        data = torch.moveaxis(data, 4, 2)

        # (k, 12, num_channels, 495, 436) -> (k, 12 * num_channels, 495, 436)
        data = torch.reshape(data, (data.shape[0], num_time_steps * num_channels, 495, 436))

        if not batch_dim:
            # `(1, 12 * num_channels, 495, 436) -> (12 * num_channels, 495, 436)`
            data = torch.squeeze(data, 0)
        return data

    @staticmethod
    def transform_unstack_channels_on_time(data: torch.Tensor, num_channels=8, batch_dim: bool = False):
        """
        Enhanced version for output (8 channels):
        `(k, 6 * 8, 495, 436) -> (k, 6, 495, 436, 8)`
        """
        if not batch_dim:
            # `(6 * 8, 495, 436) -> (1, 6 * 8, 495, 436)`
            data = torch.unsqueeze(data, 0)

        num_time_steps = int(data.shape[1] / num_channels)
        # (k, 6 * 8, 495, 436) -> (k, 6, 8, 495, 436)
        data = torch.reshape(data, (data.shape[0], num_time_steps, num_channels, 495, 436))

        # (k, 6, 8, 495, 436) -> (k, 6, 495, 436, 8)
        data = torch.moveaxis(data, 2, 4)

        if not batch_dim:
            # `(1, 6, 495, 436, 8) -> (6, 495, 436, 8)`
            data = torch.squeeze(data, 0)
        return data


class TrafficUNetModel(nn.Module):
    """Wrapper model for Traffic4Cast with enhanced channels"""
    
    def __init__(self, 
                 input_timesteps: int = 12,
                 output_timesteps: int = 6,
                 input_channels: int = 10,
                 output_channels: int = 8,
                 unet_depth: int = 5,
                 unet_wf: int = 6,
                 padding: bool = True,
                 batch_norm: bool = True):
        """
        Traffic prediction model using enhanced UNet
        
        Args:
            input_timesteps: Number of input time steps
            output_timesteps: Number of output time steps  
            input_channels: Number of input channels (10 for enhanced)
            output_channels: Number of output channels (8 for original format)
            unet_depth: Depth of UNet
            unet_wf: Width factor for UNet
            padding: Whether to use padding in UNet
            batch_norm: Whether to use batch normalization
        """
        super(TrafficUNetModel, self).__init__()
        
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Calculate input and output dimensions for UNet
        # Input: (batch, input_timesteps * input_channels, H, W)  
        # Output: (batch, output_timesteps * output_channels, H, W)
        unet_in_channels = input_timesteps * input_channels
        unet_out_channels = output_timesteps * output_channels
        
        self.unet = EnhancedUNet(
            in_channels=unet_in_channels,
            n_classes=unet_out_channels,
            depth=unet_depth,
            wf=unet_wf,
            padding=padding,
            batch_norm=batch_norm,
            num_input_channels=input_channels,
            num_output_channels=output_channels
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, input_timesteps, H, W, input_channels)
            
        Returns:
            Output tensor of shape (batch, output_timesteps, H, W, output_channels)
        """
        batch_size = x.shape[0]
        
        # Transform for UNet: (batch, T, H, W, C) -> (batch, T*C, H, W)
        x = EnhancedUNetTransformer.transform_stack_channels_on_time(
            x, batch_dim=True, num_channels=self.input_channels)
        
        # Pass through UNet
        x = self.unet(x)
        
        # Transform back: (batch, T*C, H, W) -> (batch, T, H, W, C)
        x = EnhancedUNetTransformer.transform_unstack_channels_on_time(
            x, num_channels=self.output_channels, batch_dim=True)
        
        return x


# Factory function to create models with different configurations
def create_traffic_model(config_type: str = "enhanced", **kwargs):
    """
    Factory function to create traffic prediction models
    
    Args:
        config_type: Type of configuration ("original", "enhanced")
        **kwargs: Additional arguments to pass to model
        
    Returns:
        Configured model
    """
    if config_type == "original":
        return TrafficUNetModel(
            input_channels=8,
            output_channels=8,
            **kwargs
        )
    elif config_type == "enhanced":
        return TrafficUNetModel(
            input_channels=10,
            output_channels=8,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown config_type: {config_type}")


# Example usage
if __name__ == "__main__":
    # Test the enhanced model
    model = create_traffic_model("enhanced")
    
    # Create dummy input: (batch=2, time=12, height=495, width=436, channels=10)
    dummy_input = torch.randn(2, 12, 495, 436, 10)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (2, 6, 495, 436, 8)")
    
    # Test original model
    original_model = create_traffic_model("original")
    dummy_input_original = torch.randn(2, 12, 495, 436, 8)
    
    output_original = original_model(dummy_input_original)
    print(f"Original model output shape: {output_original.shape}")
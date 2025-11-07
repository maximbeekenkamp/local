"""
UNet architecture for 1D temporal data.

Implements encoder-decoder with skip connections for learning
temporal mappings (earthquake accelerations → structural displacements).
"""

import torch
import torch.nn as nn
from typing import List


class ConvBlock1D(nn.Module):
    """
    Convolutional block with GroupNorm and GELU activation.

    Used as building block for encoder and decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_groups: int = 4
    ):
        """
        Initialize convolutional block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size (default 3)
            padding: Padding to preserve length (default 1)
            num_groups: Number of groups for GroupNorm (default 4)
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class UNet1D(nn.Module):
    """
    UNet architecture for 1D temporal operator learning.

    Architecture:
        - 3 encoder levels: Conv1d + GroupNorm + GELU + MaxPool1d
        - Bottleneck: Conv1d blocks at lowest resolution
        - 3 decoder levels: ConvTranspose1d + skip connections + Conv1d blocks
        - Skip connections via concatenation

    Input shape: [batch, 1, 4000]
    Output shape: [batch, 1, 4000]

    Reference: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
               Image Segmentation" (2015), adapted for 1D temporal data
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 40,
        num_levels: int = 3,
        kernel_size: int = 3,
        num_groups: int = 4
    ):
        """
        Initialize UNet1D.

        Args:
            in_channels: Number of input channels (default 1)
            out_channels: Number of output channels (default 1)
            base_channels: Base channel count, doubled at each level (default 28)
            num_levels: Number of encoder/decoder levels (default 3)
            kernel_size: Convolution kernel size (default 3)
            num_groups: Groups for GroupNorm (default 4)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.num_groups = num_groups

        # Calculate channel counts for each level
        # Level 0: base_channels (28)
        # Level 1: base_channels * 2 (56)
        # Level 2: base_channels * 4 (112)
        self.encoder_channels = [base_channels * (2 ** i) for i in range(num_levels)]
        self.decoder_channels = self.encoder_channels[::-1]  # Reverse for decoder

        # Initial convolution: 1 → base_channels
        self.initial_conv = ConvBlock1D(
            in_channels,
            base_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            num_groups=num_groups
        )

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.pooling_layers = nn.ModuleList()

        for i in range(num_levels):
            if i == 0:
                enc_in_channels = base_channels
            else:
                enc_in_channels = self.encoder_channels[i - 1]

            enc_out_channels = self.encoder_channels[i]

            # Encoder convolution block
            self.encoder_blocks.append(
                ConvBlock1D(
                    enc_in_channels,
                    enc_out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    num_groups=num_groups
                )
            )

            # Max pooling for downsampling (stride=2)
            self.pooling_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Bottleneck
        bottleneck_channels = self.encoder_channels[-1]
        self.bottleneck = ConvBlock1D(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            num_groups=num_groups
        )

        # Decoder blocks
        self.upsampling_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(num_levels):
            # Upsampling via transposed convolution
            dec_in_channels = self.decoder_channels[i]

            if i < num_levels - 1:
                dec_out_channels = self.decoder_channels[i + 1]
            else:
                dec_out_channels = base_channels

            self.upsampling_layers.append(
                nn.ConvTranspose1d(
                    dec_in_channels,
                    dec_out_channels,
                    kernel_size=2,
                    stride=2
                )
            )

            # Decoder convolution block
            # Input: upsampled features + skip connection (concatenated)
            # Skip connection comes from encoder at same spatial resolution
            # Skip has encoder_channels[num_levels - 1 - i] channels
            skip_idx = num_levels - 1 - i
            skip_channels = self.encoder_channels[skip_idx]
            decoder_input_channels = dec_out_channels + skip_channels

            self.decoder_blocks.append(
                ConvBlock1D(
                    decoder_input_channels,  # Upsampled + skip
                    dec_out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    num_groups=num_groups
                )
            )

        # Final output convolution: base_channels → out_channels
        self.output_conv = nn.Conv1d(
            base_channels,
            out_channels,
            kernel_size=1  # 1x1 convolution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet.

        Args:
            x: Input tensor of shape [batch, 1, timesteps]

        Returns:
            Output tensor of shape [batch, 1, timesteps]
        """
        # Initial convolution
        x = self.initial_conv(x)  # [B, base_channels, T]

        # Encoder path (save skip connections)
        encoder_outputs = []

        for i in range(self.num_levels):
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)  # Save for skip connections
            x = self.pooling_layers[i](x)  # Downsample

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (with skip connections)
        for i in range(self.num_levels):
            # Upsample
            x = self.upsampling_layers[i](x)

            # Get corresponding skip connection (reverse order)
            skip_idx = self.num_levels - 1 - i
            skip = encoder_outputs[skip_idx]

            # Concatenate with skip connection along channel dimension
            x = torch.cat([x, skip], dim=1)

            # Decoder convolution
            x = self.decoder_blocks[i](x)

        # Final output convolution
        x = self.output_conv(x)

        return x

    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

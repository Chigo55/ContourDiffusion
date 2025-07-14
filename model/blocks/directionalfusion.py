import torch
import torch.nn as nn

from blocks import SWISH


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        """
        Initializes the DepthwiseSeparableConv with the given input and output channels.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int | tuple[int, int]): Convolution window, int → broadcast.
            stride (int | tuple[int, int]): Window step size.
            padding (int | tuple[int, int]): Zero-padding on input.
            dilation (int | tuple[int, int]): Spacing between kernel elements.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.ds_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels
            ),
            SWISH()
        )

    def forward(self, x):
        """
        Forward pass of the DepthwiseSeparableConv.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W),

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        x = self.ds_block(x)
        return x


class DilatedResblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DilatedResblock with the given input and output channels.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(
                num_features=out_channels
            )
        )
        self.dr_block = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1
            ),
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2
            ),
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3
            ),
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2
            ),
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1
            )
        )

    def forward(self, x):
        """
        Forward pass of the DilatedResblock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        if self.in_channels == self.out_channels:
            x = self.dr_block(x) + x
        else:
            x = self.dr_block(x) + self.skip(x)
        return x


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        """
        Initializes the SqueezeExcitationBlock with the given input and output channels.

        Args:
            in_channels (int): Number of input channels.
            squeeze_channels (int): Number of squeeze channels
        """
        super().__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(
                in_features=in_channels,
                out_features=squeeze_channels,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=squeeze_channels,
                out_features=in_channels,
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.se_block(x) + x
        return x


class IntraLevelFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_bands):
        """
        Initializes the IntraLevelFusionBlock with the given input, output channels and number of subbands.

        Args:
            in_channels (int): Number of Channels per band.
            out_channels (int): Number of Output channels after fusion.
            num_bands (int): Number of directional sub-bands.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bands = num_bands
        self.total_in_channels = num_bands * in_channels
        self.squeeze_channels = in_channels // num_bands

        self.if_block = nn.Sequential(
            SqueezeExcitationBlock(
                in_channels=self.total_in_channels,
                squeeze_channels=self.squeeze_channels
            ),
            DepthwiseSeparableConv(
                in_channels=self.squeeze_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1
            ),
            DilatedResblock(
                in_channels=out_channels,
                out_channels=out_channels
            )
        )

    def forward(self, subband_list):
        """
        Args:
            subband_list (list[torch.Tensor]): A list of directional sub-band tensors for a single level. Each tensor has shape (B, C_in, H, W).

        Returns:
            torch.Tensor: A single fused feature map for the level. Shape: (B, out_channels, H, W).
        """
        x = torch.cat(tensors=subband_list, dim=1)
        x = self.if_block(x)
        return x


class DirectionalFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_levels):
        """
        Initializes the IntraLevelFusionBlock with the given input, output channels and number of laplacian pyramid level

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels
            num_levels (int): Number of laplacian pyramid level
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels

        self.fusion_blocks = nn.ModuleDict()
        for level in range(num_levels, 0, -1):
            num_bands = 2 ** level
            out_ch = hidden_channels * (2 ** (level - 1))

            self.fusion_blocks[f"if{level}"] = IntraLevelFusionBlock(
                num_bands=num_bands,
                in_channels=in_channels,
                out_channels=out_ch
            )


    def forward(self, subbands):
        """
        Forward pass of the DirectionalFusion module.

        Args:
            subbands (list[list[torch.Tensor]]): A nested list of directional sub-bands
                from `ContourletTransform`. `subbands[i]` contains the list of bands
                for the i-th resolution level (from finest to coarsest).
                e.g., subbands[0] is for the 640x640 level, subbands[1] for 320x320, etc.

        Returns:
            list[torch.Tensor]: A list of fused feature maps, one for each resolution
                                level, ordered from finest to coarsest. These can be
                                injected into a UNet encoder.
        """
        fused_features = []
        for i, (subband, fusion_block) in enumerate(iterable=zip(subbands, self.fusion_blocks.values())):
            fused_map = fusion_block(subband)
            fused_features.append(fused_map)

        return fused_features

import torch
import torch.nn as nn

from model.blocks import SWISH


class DepthwiseSeparableConv(nn.Module):
    """
    DepthwiseSeparableConv applies depthwise convolution followed by pointwise convolution
    for efficient spatial and channel-wise feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple[int, int]): Convolution window size.
        stride (int | tuple[int, int]): Window step size.
        padding (int | tuple[int, int]): Zero-padding size.
        dilation (int | tuple[int, int]): Spacing between kernel elements.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
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
        Forward pass of DepthwiseSeparableConv.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        x = self.ds_block(x)
        return x


class DilatedResblock(nn.Module):
    """
    DilatedResblock applies a series of dilated DepthwiseSeparableConv layers with residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
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
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W).
        """
        if self.in_channels == self.out_channels:
            x = self.dr_block(x) + x
        else:
            x = self.dr_block(x) + self.skip(x)
        return x


class SqueezeExcitationBlock(nn.Module):
    """
    SqueezeExcitationBlock applies channel-wise attention using the Squeeze-and-Excitation mechanism.

    Args:
        in_channels (int): Number of input channels.
        squeeze_ratio (float): Ratio to compute squeeze channels from input channels.
    """
    def __init__(self, in_channels, squeeze_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze_ratio = squeeze_ratio

        self.squeeze_channels = max(1, int(round(number=in_channels * squeeze_ratio)))

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.squeeze_channels,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.squeeze_channels,
                out_channels=in_channels,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the SqueezeExcitationBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), after applying channel-wise scaling.
        """
        x = self.se_block(x) * x
        return x


class IntraLevelFusionBlock(nn.Module):
    """
    IntraLevelFusionBlock fuses multiple directional sub-band feature maps within a single resolution level
    using Squeeze-and-Excitation, Depthwise Separable Convolution, and Dilated Residual Block.

    Args:
        in_channels (int): Number of input channels per directional sub-band.
        out_channels (int): Number of output channels after fusion.
        num_bands (int): Number of directional sub-bands to be fused.
        squeeze_ratio (float): Ratio to compute squeeze channels from total input channels.
    """
    def __init__(self, in_channels, out_channels, num_bands, squeeze_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bands = num_bands
        self.total_in_channels = in_channels * num_bands
        self.squeeze_ratio = squeeze_ratio

        self.if_block = nn.Sequential(
            SqueezeExcitationBlock(
                in_channels=self.total_in_channels,
                squeeze_ratio=self.squeeze_ratio
            ),
            DepthwiseSeparableConv(
                in_channels=self.total_in_channels,
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
        Forward pass of the IntraLevelFusionBlock.

        Args:
            subband_list (list[torch.Tensor]): List of directional sub-band tensors.
                Each tensor has shape (B, in_channels, H, W).

        Returns:
            torch.Tensor: Fused feature map of shape (B, out_channels, H, W).
        """
        x = torch.cat(tensors=subband_list, dim=1)
        x = self.if_block(x)
        return x


class DirectionalFusion(nn.Module):
    """
    DirectionalFusion fuses directional sub-band feature maps across multiple resolution levels
    using IntraLevelFusionBlock for each level in a Laplacian pyramid structure.

    Args:
        in_channels (int): Number of input channels per directional sub-band.
        hidden_channels (int): Base number of hidden channels used for fusion.
        num_levels (int): Number of Laplacian pyramid levels to process.
        squeeze_ratio (float): Ratio to compute squeeze channels from total input channels in each IntraLevelFusionBlock.
    """
    def __init__(self, in_channels, hidden_channels, num_levels, squeeze_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.squeeze_ratio = squeeze_ratio

        self.fusion_blocks = nn.ModuleDict()
        for level in range(num_levels, 0, -1):
            num_bands = 2 ** level
            out_ch = hidden_channels * (2 ** (level - 1))

            self.fusion_blocks[f"if{level}"] = IntraLevelFusionBlock(
                in_channels=in_channels,
                out_channels=out_ch,
                num_bands=num_bands,
                squeeze_ratio=squeeze_ratio
            )

    def forward(self, subbands):
        """
        Forward pass of the DirectionalFusion module.

        Args:
            subbands (list[list[torch.Tensor]]): Directional sub-bands per resolution level.
                Each sublist contains tensors of shape (B, in_channels, H, W).

        Returns:
            list[torch.Tensor]: Fused feature maps per level, ordered from finest to coarsest.
        """
        fused_features = []
        for i, (subband, fusion_block) in enumerate(iterable=zip(subbands, self.fusion_blocks.values())):
            fused_map = fusion_block(subband)
            fused_features.append(fused_map)

        return fused_features

import torch
import torch.nn as nn
from typing import List, Tuple, Union


class DepthwiseSeparableConv(nn.Module):
    """
    Applies depthwise convolution followed by pointwise convolution for efficient
    spatial and channel-wise feature extraction.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
    ) -> None:
        """
        Initializes the DepthwiseSeparableConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): Convolution window size.
            stride (Union[int, Tuple[int, int]]): Window step size.
            padding (Union[int, Tuple[int, int]]): Zero-padding size.
            dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
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
            nn.SiLU()
        )

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the depthwise separable convolution to the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape `(B, in_channels, H, W)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, out_channels, H, W)`.
        """
        return self.ds_block(input)


class DilatedResblock(nn.Module):
    """
    Applies a series of dilated DepthwiseSeparableConv layers with a residual
    connection.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        """
        Initializes the DilatedResblock module.

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

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the dilated residual block to the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape `(B, in_channels, H, W)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, out_channels, H, W)`.
        """
        if self.in_channels == self.out_channels:
            return self.dr_block(input) + input
        else:
            return self.dr_block(input) + self.skip(input)


class SqueezeExcitationBlock(nn.Module):
    """
    Applies channel-wise attention using the Squeeze-and-Excitation mechanism.
    """
    def __init__(
        self,
        in_channels: int,
        squeeze_ratio: float
    ) -> None:
        """
        Initializes the SqueezeExcitationBlock module.

        Args:
            in_channels (int): Number of input channels.
            squeeze_ratio (float): Ratio to compute squeeze channels from input
                channels.
        """
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

    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the Squeeze-and-Excitation block to the input tensor.

        Args:
            input (torch.Tensor): Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, C, H, W)`, after applying
                channel-wise scaling.
        """
        return self.se_block(input) * input


class IntraLevelFusionBlock(nn.Module):
    """
    Fuses multiple directional sub-band feature maps within a single resolution
    level using Squeeze-and-Excitation, Depthwise Separable Convolution, and a
    Dilated Residual Block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bands: int,
        squeeze_ratio: float
    ) -> None:
        """
        Initializes the IntraLevelFusionBlock module.

        Args:
            in_channels (int): Number of input channels per directional sub-band.
            out_channels (int): Number of output channels after fusion.
            num_bands (int): Number of directional sub-bands to be fused.
            squeeze_ratio (float): Ratio to compute squeeze channels from total
                input channels.
        """
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

    def forward(
        self,
        subband_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuses a list of directional sub-band tensors.

        Args:
            subband_list (List[torch.Tensor]): List of directional sub-band tensors.
                Each tensor has shape `(B, in_channels, H, W)`.

        Returns:
            torch.Tensor: Fused feature map of shape `(B, out_channels, H, W)`.
        """
        return self.if_block(torch.cat(tensors=subband_list, dim=1))


class DirectionalFusion(nn.Module):
    """
    Fuses directional sub-band feature maps across multiple resolution levels
    using an `IntraLevelFusionBlock` for each level.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_levels: int,
        squeeze_ratio: float
    ) -> None:
        """
        Initializes the DirectionalFusion module.

        Args:
            in_channels (int): Number of input channels per directional sub-band.
            hidden_channels (int): Base number of hidden channels used for fusion.
            num_levels (int): Number of Laplacian pyramid levels to process.
            squeeze_ratio (float): Ratio for Squeeze-and-Excitation blocks in each
                `IntraLevelFusionBlock`.
        """
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

    def forward(
        self,
        subbands: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Fuses directional sub-bands from multiple resolution levels.

        Args:
            subbands (List[List[torch.Tensor]]): A list of lists, where each inner
                list contains directional sub-band tensors for a specific resolution
                level. Each tensor has shape `(B, in_channels, H, W)`.

        Returns:
            List[torch.Tensor]: A list of fused feature maps, one for each level,
                ordered from finest to coarsest resolution.
        """
        fused_features = []
        for i, (subband, fusion_block) in enumerate(iterable=zip(subbands, self.fusion_blocks.values())):
            fused_map = fusion_block(subband)
            fused_features.append(fused_map)

        return fused_features

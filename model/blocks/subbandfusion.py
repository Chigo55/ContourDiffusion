import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DepthwiseSeparableConv with the given input and output channels.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, x):
        """
        Forward pass of the DepthwiseSeparableConv.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W),

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        x = self.depth_conv(x)
        x = self.point_conv(x)
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

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                dilation=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=2,
                dilation=(2, 2)
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=3,
                dilation=(3, 3)
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=2,
                dilation=(2, 2)
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                dilation=(1, 1)
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
        x = self.model(x) + x
        return x


# FIXME: Block structure definition and modification required
class SubbandFusion(nn.Module):
    """
    Initialize the SubbandFusion

    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Number of base channels.
        num_levels (int): Number of resolution levels in the DFB pyramid
        num_heads (int): Number of attention heads
    """
    def __init__(self, in_channels, base_channels, num_levels, num_heads):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.num_heads = num_heads

        self.ds_conv  = DepthwiseSeparableConv(in_channels=in_channels,  out_channels=base_channels)
        self.resblock = DilatedResblock(in_channels=base_channels, out_channels=base_channels)

        self.attn = nn.ModuleList(
            modules=[
                CrossAttentionBlock(dim=base_channels, num_heads=num_heads)
                for _ in range(num_levels)
            ]
        )

        self.lateral = nn.ModuleList(
            modules=[
                nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=1)
                for _ in range(num_levels)
            ]
        )

    def _process_band(self, x):
        x = self.ds_conv(x)
        x = self.resblock(x)
        return x

    def forward(self, subbands):
        """
        Forward pass of the SubbandFusion.

        Args:
            subbands (list[list[torch.Tensor]]): Nested list containing
                directional sub-bands produced by a Directional Filter Bank.
                Shape for each tensor: (B, C_in, H_l, W_l),
                where l ∈ {0, …, num_levels-1}.

        Returns:
            torch.Tensor: Fused multi-scale feature map of shape
                (B, base_channels, H₀, W₀).
        """
        level_feats = []
        ref_size = None

        for lvl, sb_list in enumerate(iterable=subbands):
            bands_proc = [self._process_band(sb) for sb in sb_list]

            ctx_map = torch.mean(input=torch.stack(tensors=bands_proc, dim=0), dim=0)
            bands_fused = [
                self.attn[lvl](band, ctx_map)
                for band in bands_proc
            ]

            level_map = torch.mean(input=torch.stack(tensors=bands_fused, dim=0), dim=0)
            level_feats.append(level_map)

            if lvl == 0:
                ref_size = level_map.shape[-2:]

        fused = 0
        for lvl, feat in enumerate(iterable=level_feats):
            feat = self.lateral[lvl](feat)
            if feat.shape[-2:] != ref_size:
                feat = F.interpolate(input=feat, size=ref_size, mode='nearest', align_corners=False)
            fused = fused + feat

        return fused
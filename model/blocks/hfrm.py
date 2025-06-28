import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DepthSepConvBlock with the given input and output channels.

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
        Forward pass of the DepthSepConvBlock.

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
                padding=1,
                dilation=(2, 2)
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
                dilation=(3, 3)
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
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


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        """
        Initializes the CrossAttentionBlock with the given dimension and dropout rate.

        Args:
            dim (int): Dimension of the input features.
            num_heads (int): Number of attention heads
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.q = DepthSepConvBlock(in_channels=dim, out_channels=dim)
        self.k = DepthSepConvBlock(in_channels=dim, out_channels=dim)
        self.v = DepthSepConvBlock(in_channels=dim, out_channels=dim)

    def _reshape_to_heads(self, x):
        """
        Flattens the spatial dimensions of the input tensor while keeping the batch and channel dimensions intact.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Flattened tensor of shape (B, num_heads, HW, head_dim).
        """
        B, C, H, W = x.shape
        x = x.view(B, self.num_heads, self.head_dim, H * W)
        x = x.permute(0, 1, 3, 2)
        return x

    def forward(self, x, ctx):
        """
        Forward pass of the CrossAttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            ctx (torch.Tensor): Context tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W) after applying cross-attention.
        """
        B, C, H, W = x.shape

        q = self.q(x)
        k = self.k(ctx)
        v = self.v(ctx)

        q = self._reshape_to_heads(x=q)
        k = self._reshape_to_heads(x=k)
        v = self._reshape_to_heads(x=v)

        attn = torch.matmul(input=q, other=k.transpose(-1, -2)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = torch.matmul(input=attn, other=v)

        out = attn.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, self.dim, H * W)
        return out


class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels, num_heads, dropout):
        super(HFRM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv_heads = nn.ModuleList(
            modules=[
                DepthSepConvBlock(in_channels=in_channels, out_channels=out_channels)
                for _ in range(2 ** num_levels)
            ]
        )

        self.dilated_blocks0 = nn.ModuleList(
            modules=[
                DilatedResblock(in_channels=out_channels, out_channels=out_channels)
                for _ in range(2 ** num_levels)
            ]
        )
        self.dilated_blocks1 = nn.ModuleList(
            modules=[
                DilatedResblock(in_channels=out_channels, out_channels=out_channels)
                for _ in range(2 ** num_levels)
            ]
        )
        self.dilated_blocks2 = nn.ModuleList(
            modules=[
                DilatedResblock(in_channels=out_channels, out_channels=out_channels)
                for _ in range(2 ** num_levels)
            ]
        )

        self.cross_attentions0 = self.dilated_blocks = nn.ModuleList(
            modules=[
                CrossAttentionBlock(dim=out_channels, num_heads=num_heads)
                for _ in range(2 ** num_levels)
            ]
        )
        self.cross_attentions0 = self.dilated_blocks = nn.ModuleList(
            modules=[
                CrossAttentionBlock(dim=out_channels, num_heads=num_heads)
                for _ in range(2 ** num_levels)
            ]
        )
        self.convs = self.dilated_blocks = nn.ModuleList(
            modules=[
                nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
                for _ in range(2 ** num_levels)
            ]
        )

        self.conv_tail = nn.ModuleList(
            modules=[
                DepthSepConvBlock(in_channels=out_channels, out_channels=in_channels)
                for _ in range(2 ** num_levels)
            ]
        )

    def forward(self, x):

        b, c, h, w = x.shape

        residual = x

        x = self.conv_head(x)

        x_HL, x_LH, x_HH = x[:b//3, ...], x[b//3:2*b//3, ...], x[2*b//3:, ...]

        x_HH_LH = self.cross_attention0(x_LH, x_HH)
        x_HH_HL = self.cross_attention1(x_HL, x_HH)

        x_HL = self.dilated_block_HL(x_HL)
        x_LH = self.dilated_block_LH(x_LH)

        x_HH = self.dilated_block_HH(self.conv_HH(torch.cat((x_HH_LH, x_HH_HL), dim=1)))

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))

        return out + residual
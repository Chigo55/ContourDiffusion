import math
from turtle import st
from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from typing import List, Tuple


class TimeEmbeddingBlock(nn.Module):
    """
    A block for creating sinusoidal time embeddings, which are then passed through
    a small multi-layer perceptron (MLP). This is a standard component in diffusion models
    to condition the model on the current timestep.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
    """
    A block for creating sinusoidal time embeddings, which are then passed through
    a small multi-layer perceptron (MLP). This is a standard component in diffusion models
    to condition the model on the current timestep.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ) -> None:
        """
        Initializes the TimeEmbeddingBlock.
        Initializes the TimeEmbeddingBlock.

        Args:
            in_channels (int): The dimensionality of the sinusoidal embedding to be
                generated. This is typically the same as the model's hidden dimension.
            out_channels (int): The output dimensionality of the MLP. This should match
                the channel dimension of the feature maps where the embedding will be added.
            in_channels (int): The dimensionality of the sinusoidal embedding to be
                generated. This is typically the same as the model's hidden dimension.
            out_channels (int): The output dimensionality of the MLP. This should match
                the channel dimension of the feature maps where the embedding will be added.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.linear2 = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.act = nn.SiLU()
        self.act = nn.SiLU()

    def get_timestep_embedding(
        self,
        t: torch.Tensor,
        embed_dim: int
    ) -> torch.Tensor:
    def get_timestep_embedding(
        self,
        t: torch.Tensor,
        embed_dim: int
    ) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for the given time steps.

        Args:
            t (torch.Tensor): A 1D tensor of time steps, shape `(B,)`.
            embed_dim (int): The dimensionality of the embedding space.
            t (torch.Tensor): A 1D tensor of time steps, shape `(B,)`.
            embed_dim (int): The dimensionality of the embedding space.

        Returns:
            torch.Tensor: The sinusoidal time embeddings, shape `(B, embed_dim)`.
            torch.Tensor: The sinusoidal time embeddings, shape `(B, embed_dim)`.
        """
        half_dim = embed_dim // 2
        exponent = -math.log(10000) / (half_dim - 1)
        freqs = torch.exp(input=torch.arange(end=half_dim, dtype=torch.float32, device=t.device) * exponent)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat(tensors=[torch.sin(input=args), torch.cos(input=args)], dim=1)
        return emb

    def forward(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
    def forward(
        self,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the time embedding to the input tensor.

        Args:
            t (torch.Tensor): A 1D tensor of time steps, shape `(B,)`.
            t (torch.Tensor): A 1D tensor of time steps, shape `(B,)`.

        Returns:
            torch.Tensor: The processed time embedding tensor, shape `(B, out_channels)`.
            torch.Tensor: The processed time embedding tensor, shape `(B, out_channels)`.
        """
        t_freq = self.get_timestep_embedding(t=t, embed_dim=self.in_channels)
        t_emb = self.linear1(t_freq)
        t_emb = self.act(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


class ResnetBlock(nn.Module):
    """
    A residual block that incorporates a time embedding. It consists of two
    convolutional layers with Group Normalization and SiLU activation, with the
    time embedding added after the first convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        dropout_ratio: float,
        shortcut: bool = False
    ) -> None:
    """
    A residual block that incorporates a time embedding. It consists of two
    convolutional layers with Group Normalization and SiLU activation, with the
    time embedding added after the first convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        dropout_ratio: float,
        shortcut: bool = False
    ) -> None:
        """
        Initializes the ResnetBlock.
        Initializes the ResnetBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            temb_dim (int): Dimension of the time embedding.
            dropout_ratio (float): Dropout rate.
            shortcut (bool): Whether to use a 3x3 convolution for the shortcut connection
                if channel dimensions change. If False, a 1x1 convolution is used.
            dropout_ratio (float): Dropout rate.
            shortcut (bool): Whether to use a 3x3 convolution for the shortcut connection
                if channel dimensions change. If False, a 1x1 convolution is used.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_dim = temb_dim
        self.dropout_ratio = dropout_ratio
        self.dropout_ratio = dropout_ratio
        self.shortcut = shortcut

        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.SiLU()
        self.act1 = nn.SiLU()

        self.temb_proj = TimeEmbeddingBlock(in_channels=temb_dim, out_channels=out_channels)

        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nein_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input feature map of shape `(B, in_channels, H, W)`.
            t (torch.Tensor): Time step tensor of shape `(B,)`.
            x (torch.Tensor): Input feature map of shape `(B, in_channels, H, W)`.
            t (torch.Tensor): Time step tensor of shape `(B,)`.

        Returns:
            torch.Tensor: Output feature map of shape `(B, out_channels, H, W)`.
            torch.Tensor: Output feature map of shape `(B, out_channels, H, W)`.
        """
        h = x
        h = self.gn1(h)
        h = self.act1(h)
        h = self.conv1(h)

        t_emb = self.temb_proj(t)
        h = h + t_emb[:, :, None, None]

        h = self.gn2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nein_shortcut(x)
        return x+h


class AttentionBlock(nn.Module):
    """
    A self-attention block that operates on feature maps. It reshapes the input
    for standard multi-head attention and adds the result back to the input.
    """
    def __init__(
        self,
        dim: int
    ) -> None:
    """
    A self-attention block that operates on feature maps. It reshapes the input
    for standard multi-head attention and adds the result back to the input.
    """
    def __init__(
        self,
        dim: int
    ) -> None:
        """
        Initializes the AttentionBlock.
        Initializes the AttentionBlock.

        Args:
            dim (int): The number of input and output channels.
            dim (int): The number of input and output channels.
        """
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** 0.5
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.qkv = nn.Conv2d(in_channels=dim, out_channels=dim * 3, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

    def _reshape(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def _reshape(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape the input tensor for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, 3 * C, H, W)`.
            x (torch.Tensor): Input tensor of shape `(B, 3 * C, H, W)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
                query, key, and value tensors, each of shape `(B, 1, H*W, C)`.
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
                query, key, and value tensors, each of shape `(B, 1, H*W, C)`.
        """
        B, C, H, W = x.shape
        x = x.view(B, 3, self.dim, H * W)
        x = x.permute(0, 1, 3, 2)
        q, k, v = x.chunk(chunks=3, dim=1)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention block.

        Args:
            x (torch.Tensor): Input feature map of shape `(B, C, H, W)`.
            x (torch.Tensor): Input feature map of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Output feature map after applying attention, shape `(B, C, H, W)`.
            torch.Tensor: Output feature map after applying attention, shape `(B, C, H, W)`.
        """
        B, C, H, W = x.shape
        h = x
        h = self.norm(h)
        qkv = self.qkv(h)
        q, k, v = self._reshape(x=qkv)

        attn = torch.matmul(input=q, other=k.transpose(-1, -2)) / self.scale
        attn = F.softmax(input=attn, dim=-1)
        attn = torch.matmul(input=attn, other=v)

        out = attn.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, self.dim, H, W)
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    """
    A downsampling block, which can be either a strided convolution (trainable)
    or average pooling (non-trainable).
    """
    def __init__(
        self,
        in_channels: int,
        trainable: bool = True
    ) -> None:
    """
    A downsampling block, which can be either a strided convolution (trainable)
    or average pooling (non-trainable).
    """
    def __init__(
        self,
        in_channels: int,
        trainable: bool = True
    ) -> None:
        """
        Initializes the Downsample block.
        Initializes the Downsample block.

        Args:
            in_channels (int): Number of input channels.
            trainable (bool): If True, use a strided 3x3 convolution.
                If False, use average pooling. Defaults to True.
            trainable (bool): If True, use a strided 3x3 convolution.
                If False, use average pooling. Defaults to True.
        """
        super().__init__()
        self.trainable = trainable
        if self.trainable:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the downsampling operation.

        Args:
            x (torch.Tensor): Input feature map of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Downsampled feature map of shape `(B, C, H/2, W/2)`.
        """
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the downsampling operation.

        Args:
            x (torch.Tensor): Input feature map of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Downsampled feature map of shape `(B, C, H/2, W/2)`.
        """
        if self.trainable:
            x = self.conv(x)
            return x
        else:
            x = self.avg_pool(x)
            return x


class Upsample(nn.Module):
    """
    An upsampling block, which can be either a transposed convolution (trainable)
    or bilinear interpolation (non-trainable).
    """
    def __init__(
        self,
        in_channels: int,
        trainable: bool = True
    ) -> None:
    """
    An upsampling block, which can be either a transposed convolution (trainable)
    or bilinear interpolation (non-trainable).
    """
    def __init__(
        self,
        in_channels: int,
        trainable: bool = True
    ) -> None:
        """
        Initializes the Upsample block.
        Initializes the Upsample block.

        Args:
            in_channels (int): Number of input channels.
            trainable (bool): If True, use a 3x3 convolution following interpolation.
                If False, use only bilinear interpolation. Defaults to True.
            trainable (bool): If True, use a 3x3 convolution following interpolation.
                If False, use only bilinear interpolation. Defaults to True.
        """
        super().__init__()
        self.trainable = trainable
        if self.trainable:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the upsampling operation.

        Args:
            x (torch.Tensor): Input feature map of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Upsampled feature map of shape `(B, C, H*2, W*2)`.
        """
        x = F.interpolate(input=x, scale_factor=2.0, mode="bilinear", align_corners=False)
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the upsampling operation.

        Args:
            x (torch.Tensor): Input feature map of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: Upsampled feature map of shape `(B, C, H*2, W*2)`.
        """
        x = F.interpolate(input=x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if self.trainable:
            x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    A U-Net architecture with residual blocks, self-attention, and time conditioning.
    It takes additional fused directional features as skip connections.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        temb_dim: int,
        dropout_ratio: float,
        shortcut: bool,
        trainable: bool
    ) -> None:
    """
    A U-Net architecture with residual blocks, self-attention, and time conditioning.
    It takes additional fused directional features as skip connections.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        temb_dim: int,
        dropout_ratio: float,
        shortcut: bool,
        trainable: bool
    ) -> None:
        """
        Initializes the UNet model.
        Initializes the UNet model.

        Args:
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output image channels.
            hidden_channels (int): Base number of channels for the first level of the U-Net.
            num_levels (int): Number of downsampling/upsampling levels in the U-Net.
            temb_dim (int): Dimensionality of the time embedding.
            dropout_ratio (float): Dropout rate for ResnetBlocks.
            shortcut (bool): Type of shortcut connection in ResnetBlocks.
            trainable (bool): Whether the down/upsampling layers are trainable.
            in_channels (int): Number of input image channels.
            out_channels (int): Number of output image channels.
            hidden_channels (int): Base number of channels for the first level of the U-Net.
            num_levels (int): Number of downsampling/upsampling levels in the U-Net.
            temb_dim (int): Dimensionality of the time embedding.
            dropout_ratio (float): Dropout rate for ResnetBlocks.
            shortcut (bool): Type of shortcut connection in ResnetBlocks.
            trainable (bool): Whether the down/upsampling layers are trainable.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.dropout_ratio = dropout_ratio
        self.dropout_ratio = dropout_ratio
        self.shortcut = shortcut
        self.trainable = trainable

        self.condition_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.in_conv = nn.Conv2d(in_channels=in_channels+hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.in_conv = nn.Conv2d(in_channels=in_channels+hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleDict()
        for level in range(1, num_levels + 1):
            in_ch = hidden_channels * (2 ** (level - 1))
            out_ch = hidden_channels * (2 ** level)
            module_list = [
                ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                Downsample(in_channels=out_ch, trainable=trainable)
            ]
            self.down[f'down{level}'] = nn.ModuleList(modules=module_list)

        self.mid = nn.ModuleList(
            modules=[
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                AttentionBlock(dim=hidden_channels * (2 ** num_levels)),
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                AttentionBlock(dim=hidden_channels * (2 ** num_levels)),
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False)
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False)
            ]
        )

        self.up = nn.ModuleDict()
        for level in range(num_levels, 0, -1):
            in_ch = hidden_channels * (2 ** level)
            out_ch = hidden_channels * (2 ** (level - 1))
            module_list = [
                ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_dim=temb_dim, dropout_ratio=dropout_ratio, shortcut=False),
                Upsample(in_channels=out_ch, trainable=trainable)
            ]
            self.up[f'up{level}'] = nn.ModuleList(modules=module_list)

        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        input: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        f: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            input (torch.Tensor): The input tensor (e.g., noisy image),
                shape `(B, in_channels, H, W)`.
            t (torch.Tensor): The time step tensor, shape `(B,)`.
            f (List[torch.Tensor]): A list of fused directional feature maps from the
                DirectionalFusion module, used as skip connections.

        Returns:
            torch.Tensor: The output tensor (e.g., predicted noise), shape `(B, out_channels, H, W)`.
        """
        condition = self.condition_encoder(condition)
        input = torch.cat(tensors=[input, condition], dim=1)

        h = self.in_conv(input)

        for idx, key in enumerate(iterable=self.down):
            for layer in self.down[key]:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t)
                else:
                    h = layer(h)

        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                h = layer(h, t)
            else:
                h = layer(h)
        for idx, key in enumerate(iterable=self.up):
            for layer in self.up[key]:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t)
                else:
                    h = layer(h)
                    h = h + f[idx]
                    h = h + f[idx]

        return self.out_conv(h)

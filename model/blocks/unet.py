import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SWISH(nn.Sigmoid):
    def __init__(self, *args, **kwargs):
        """
        Initialize the SWISH activation function.
        SWISH is defined as x * sigmoid(x).
        """
        super().__init__(*args, **kwargs)

    def forward(self, input):
        """
        Apply the SWISH activation function to the input tensor.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying SWISH activation.
        """
        return input * super().forward(input=input)


class TimeEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize the time embedding module.

        Args:
            in_channels (int): Number of input channels (time steps).
            out_channels (int): Number of output channels for the embedding.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.linear2 = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.act = SWISH()

    def get_timestep_embedding(self, t, embed_dim):
        """
        Generate sinusoidal embeddings for the given time steps.

        Args:
            t (torch.Tensor): Input tensor representing time steps.
            embed_dim (int): Dim of the embedding space.

        Returns:
            torch.Tensor: Sinusoidal embeddings for the time steps.
        """
        half_dim = embed_dim // 2
        exponent = -math.log(10000) / (half_dim - 1)
        freqs = torch.exp(input=torch.arange(end=half_dim, dtype=torch.float32, device=t.device) * exponent)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat(tensors=[torch.sin(input=args), torch.cos(input=args)], dim=1)
        return emb

    def forward(self, t):
        """
        Apply the time embedding to the input tensor.

        Args:
            t (torch.Tensor): Input tensor representing time steps.

        Returns:
            torch.Tensor: Output tensor after applying time embedding.
        """
        t_freq = self.get_timestep_embedding(t=t, embed_dim=self.in_channels)
        t_emb = self.linear1(t_freq)
        t_emb = self.act(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim, dropout, shortcut=False):
        """
        Initialize the residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            temb_dim (int): Dimension of the time embedding.
            dropout (float): Dropout rate.
            shortcut (bool): Whether to use a shortcut connection.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_dim = temb_dim
        self.dropout = dropout
        self.shortcut = shortcut

        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act1 = SWISH()

        self.temb_proj = TimeEmbeddingBlock(in_channels=temb_dim, out_channels=out_channels)

        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.act2 = SWISH()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nein_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step tensor of shape (B, 1).

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
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
    def __init__(self, dim):
        """
        Initialize the attention block.

        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.dim = dim
        self.scale = self.dim ** 0.5
        self.norm = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.qkv = nn.Conv2d(in_channels=dim, out_channels=dim * 3, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

    def _reshape(self, x):
        """
        Reshape the input tensor for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple: Tuple containing query, key, and value tensors.
        """
        B, C, H, W = x.shape
        x = x.view(B, 3, self.dim, H * W)
        x = x.permute(0, 1, 3, 2)
        q, k, v = x.chunk(chunks=3, dim=1)
        return q, k, v

    def forward(self, x):
        """
        Forward pass of the attention block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, C, H, W = x.shape
        h = x
        h = self.norm(h)
        qkv = self.qkv(h)
        q, k, v = self._reshape(qkv)

        attn = torch.matmul(input=q, other=k.transpose(-1, -2)) / self.scale
        attn = F.softmax(input=attn, dim=-1)
        attn = torch.matmul(input=attn, other=v)

        out = attn.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, self.dim, H, W)
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, in_channels, trainable=True):
        """
        Initialize the downsample block.

        Args:
            in_channels (int): Number of input channels.
            trainable (bool): Whether the downsample block is trainable. If False, it uses average pooling.
        """
        super().__init__()
        self.trainable = trainable
        if self.trainable:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.trainable:
            x = self.conv(x)
            return x
        else:
            x = self.avg_pool(x)
            return x


class Upsample(nn.Module):
    def __init__(self, in_channels, trainable=True):
        """
        Initialize the upsample block.

        Args:
            in_channels (int): Number of input channels.
            trainable (bool): Whether the upsample block is trainable. If False, it uses nearest neighbor interpolation.
        """
        super().__init__()
        self.trainable = trainable
        if self.trainable:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.trainable:
            x = F.interpolate(input=x, scale_factor=2.0, mode="nearest")
            x = self.conv(x)
            return x
        else:
            x = F.interpolate(input=x, scale_factor=2.0, mode="nearest")
            return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_levels, temb_dim, dropout, shortcut, trainable):
        """
        Initialize the UNet.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_channels (int): Number of hidden channels.
            num_levels (int): Number of levels in the UNet architecture.
            temb_dim (int): Dimension of the time embedding.
            dropout (float): Dropout rate.
            shortcut (bool): Whether to use shortcut connections in the residual blocks.
            trainable (bool): Whether the downsample and upsample blocks are trainable.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.dropout = dropout
        self.shortcut = shortcut
        self.trainable = trainable

        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleDict()
        for level in range(1, num_levels + 1):
            in_ch = hidden_channels * (2 ** (level - 1))
            out_ch = hidden_channels * (2 ** level)
            module_list = [
                ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_dim=temb_dim, dropout=dropout, shortcut=False),
                Downsample(in_channels=out_ch, trainable=trainable)
            ]

            if level % 2 == 1:
                module_list.insert(1, AttentionBlock(dim=out_ch))

            self.down[f'down{level}'] = nn.ModuleList(modules=module_list)

        self.mid = nn.ModuleList(
            modules=[
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout=dropout, shortcut=False),
                AttentionBlock(dim=hidden_channels * (2 ** num_levels)),
                ResnetBlock(in_channels=hidden_channels * (2 ** num_levels), out_channels=hidden_channels * (2 ** num_levels), temb_dim=temb_dim, dropout=dropout, shortcut=False)
            ]
        )

        self.up = nn.ModuleDict()
        for level in range(num_levels, 0, -1):
            in_ch = hidden_channels * (2 ** level)
            out_ch = hidden_channels * (2 ** (level - 1))
            module_list = [
                ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_dim=temb_dim, dropout=dropout, shortcut=False),
                Upsample(in_channels=out_ch, trainable=trainable)
            ]

            if level % 2 == 1:
                module_list.insert(1, AttentionBlock(dim=out_ch))

            self.up[f'up{level}'] = nn.ModuleList(modules=module_list)

        self.out_conv = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        """
        Forward pass of the UNet.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            t (torch.Tensor): Time step tensor of shape (B, 1).

        Returns:
            torch.Tensor: Output tensor after applying the UNet.
        """
        h = self.in_conv(x)

        for level in range(1, self.num_levels + 1):
            for layer in self.down[f'down{level}']:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t)
                else:
                    h = layer(h)

        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                h = layer(h, t)
            else:
                h = layer(h)

        for level in range(self.num_levels , 0, -1):
            for layer in self.up[f'up{level}']:
                if isinstance(layer, ResnetBlock):
                    h = layer(h, t)
                else:
                    h = layer(h)

        return self.out_conv(h)

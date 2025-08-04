import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# FIXME: LaplacianPyramid is not implemented in the original code.
class LaplacianPyramid(nn.Module):
    """
    Laplacian Pyramid module that applies a Laplacian filter to the input tensor.
    This module is used to extract features at different scales.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the Laplacian Pyramid module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Apply the Laplacian filter to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Laplacian filter.
        """
        return self.conv(x)


# FIXME: DirectionalFilterBank is not implemented in the original code.
class DirectionalFilterBank(nn.Module):
    """
    Directional Filter Bank module that applies a set of directional filters to the input tensor.
    This module is used to extract directional features from the input.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the Directional Filter Bank module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Apply the directional filters to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the directional filters.
        """
        return self.conv(x)


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


class TimeEmbedding(nn.Module):
    """
    Time embedding module that projects time steps into a higher-dimensional space.
    This module uses sinusoidal functions to create embeddings for time steps.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the time embedding module.

        Args:
            in_channels (int): Number of input channels (time steps).
            out_channels (int): Number of output channels for the embedding.
        """
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.linear2 = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.act = SWISH()

    def get_timestep_embedding(self, t, embed_dim):
        """        Generate sinusoidal embeddings for the given time steps.
        Args:
            t (torch.Tensor): Input tensor representing time steps.
            embed_dim (int): Dimension of the embedding space.
        Returns:
            torch.Tensor: Sinusoidal embeddings for the time steps.
        """
        half_dim = embed_dim // 2
        exponent = -math.log(x=10000) / (half_dim - 1)
        freqs = torch.exp(input=torch.arange(end=half_dim, dtype=torch.float32) * exponent)
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
        # FIXME:
        t_freq = self.get_timestep_embedding(t=t, embed_dim=in_channels)
        t_emb = self.linear1(t_freq)
        t_emb = self.act(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


class ResnetBlock(nn.Module):
    """
    A basic ResNet block with two convolutional layers.
    This block is used to learn residual mappings.
    """
    def __init__(self, in_channels, out_channels, shortcut=False, dropout=0.2, temb_channels=512):
        """
        Initialize the ResNet block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            shortcut (bool): Whether to use a shortcut connection. Default is False.
            dropout (float): Dropout rate. Default is 0.2.
            temb_channels (int): Number of channels in the time embedding. Default is 512.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut = shortcut

        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.act1 = SWISH()

        self.temb_proj = TimeEmbedding(in_channels=temb_channels, out_channels=out_channels)

        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.act2 = SWISH()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.gn1(h)
        h = self.act1(h)
        h = self.conv1(h)

        h = h + self.temb_proj(self.act1(temb))[:, :, None, None]

        h = self.gn2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttentionBlock(nn.Module):
    """
    Attention block that applies self-attention to the input tensor.
    This block is used to capture long-range dependencies in the data.
    """
    def __init__(self, in_channels):
        """
        Initialize the attention block.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.qkv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 3, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm(h)
        qkv = self.qkv(h).reshape(shape=(h.shape[0], 3, self.in_channels, h.shape[2] * h.shape[3]))
        q, k, v = qkv.chunk(chunks=3, dim=1)

        attention = (q.transpose(2, 3) @ k) * (self.in_channels ** -0.5)
        attention = F.softmax(attention, dim=-1)

        out = (attention @ v.transpose(2, 3)).transpose(2, 3).reshape(shape=h.shape)
        out = self.proj_out(out)

        return x + out


class Downsample(nn.Module):
    """
    Downsample block that reduces the spatial dimensions of the input tensor.
    This block uses a convolutional layer with stride 2 to downsample the input.
    """
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
            return self.conv(x)
        else:
            return self.avg_pool(x)

class Upsample(nn.Module):
    """
    Upsample block that increases the spatial dimensions of the input tensor.
    This block uses a transposed convolutional layer to upsample the input.
    """
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
            return self.conv(x)
        else:
            return torch.nn.functional.interpolate(input=x, scale_factor=2.0, mode="nearest")


class DiffusionUNet(nn.Module):
    """
    A simple UNet architecture for diffusion models.
    This class defines the structure of the UNet, including downsampling and upsampling blocks.
    """
    def __init__(self, in_channels:int=1, out_channels:int=1, base_channels:int=64, resolution:int=5, num_blocks:int=2, multiply:tuple=(1, 2, 3, 4, 5), shortcut:bool=False, trainable:bool=False, dropout:float=0.2):
        """
        Initialize the DiffusionUNet.

        Args:
            in_channels (int): Number of input channels. Default is 3 (RGB).
            out_channels (int): Number of output channels. Default is 3 (RGB).
            base_channels (int): Base number of channels for the first layer. Default is 64.
        """
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1)

        in_ch = 0
        in_multiply = (1,)+multiply
        self.down = nn.ModuleList()
        for i_level in range(resolution):
            resnet = nn.ModuleList()
            attention = nn.ModuleList()
            in_ch = base_channels * in_multiply[i_level]
            out_ch = base_channels * multiply[i_level]
            for i_block in range(num_blocks):
                resnet.append(module=ResnetBlock(in_channels=in_ch, out_channels=out_ch, temb_channels=base_channels * 4, dropout=dropout, shortcut=shortcut))
                in_ch = out_ch
                if i_level == 2:
                    attention.append(module=AttentionBlock(in_channels=in_ch))

            down = nn.Module()
            down.resnet = resnet
            down.attention = attention
            if i_level != self.resolution-1:
                down.downsample = Downsample(in_channels=in_ch, trainable=trainable)
            self.down.append(module=down)

        self.mid = nn.Module()
        self.mid.resnet1 = ResnetBlock(
            in_channels=in_ch,
            out_channels=in_ch,
            temb_channels=base_channels * 4,
            dropout=dropout
        )
        self.mid.attention = AttentionBlock(in_channels=in_ch)
        self.mid.resnet2 = ResnetBlock(
            in_channels=in_ch,
            out_channels=in_ch,
            temb_channels=base_channels * 4,
            dropout=dropout
        )

        self.up = nn.ModuleList()
        for i_level in reversed(range(resolution)):
            resnet = nn.ModuleList()
            attention = nn.ModuleList()
            out_ch = base_channels*multiply[i_level]
            sk_ch = base_channels*multiply[i_level]
            for i_block in range(num_blocks+1):
                if i_block == num_blocks:
                    sk_ch = base_channels*in_multiply[i_level]
                resnet.append(module=ResnetBlock(in_channels=in_ch+sk_ch, out_channels=out_ch, temb_channels=base_channels * 4, dropout=dropout, shortcut=shortcut))
                in_ch = out_ch
                if i_level == 2:
                    attention.append(module=AttentionBlock(in_channels=in_ch))

            up = nn.Module()
            up.resnet = resnet
            up.attention = attention
            if i_level != 0:
                up.upsample = Upsample(in_channels=in_ch, trainable=trainable)
            self.up.insert(index=0, module=up)

        self.out_conv = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.in_conv(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        return self.out_conv(h)
import torch
import torch.nn as nn
from typing import List, Tuple

from model.blocks.contourlet import ContourletTransform
from model.blocks.directionalfusion import DirectionalFusion
from model.blocks.unet import UNet

class ContourletDiffusion(nn.Module):
    """
    The main model that orchestrates the Contourlet Diffusion process.

    This model integrates three main components:
    1.  `ContourletTransform`: Decomposes the input image into a multi-scale,
        multi-directional representation (pyramid and sub-bands).
    2.  `DirectionalFusion`: Fuses the directional sub-band features to create
        rich feature maps for skip connections.
    3.  `UNet`: A time-conditioned U-Net that takes the noisy image and the fused
        directional features to predict the denoised output.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_channels=64,
        num_levels=4,
        temb_dim=64,
        dropout_ratio=0.1,
        filter_size=5,
        sigma=1.0,
        omega_x=0.25,
        omega_y=0.25,
        squeeze_ratio=0.3,
        shortcut=True,
        trainable=False
    ) -> None:
        """
        Initializes the ContourletDiffusion model.

        Args:
            in_channels (int): Number of input channels for the image (e.g., 3 for RGB).
            out_channels (int): Number of output channels for the image.
            hidden_channels (int): Base number of channels for the U-Net and fusion blocks.
            num_levels (int): Number of decomposition levels for the Contourlet transform.
            temb_dim (int): Dimensionality of the time embedding for the U-Net.
            dropout_ratio (float): Dropout rate used in the U-Net's ResnetBlocks.
            filter_size (int): Kernel size for the Contourlet transform filters.
            sigma (float): Standard deviation for the Gaussian kernel in the Contourlet transform.
            omega_x (float): Modulation frequency in the input-direction for the Contourlet transform.
            omega_y (float): Modulation frequency in the y-direction for the Contourlet transform.
            squeeze_ratio (float): Squeeze ratio for the Squeeze-and-Excitation blocks in the fusion module.
            shortcut (bool): Whether to use shortcut connections in the U-Net's ResnetBlocks.
            trainable (bool): Whether the down/upsampling layers in the U-Net are trainable.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.squeeze_ratio = squeeze_ratio
        self.temb_dim = temb_dim
        self.dropout_ratio = dropout_ratio
        self.shortcut = shortcut
        self.trainable = trainable

        self.contourlet = ContourletTransform(
            in_channels=in_channels,
            num_levels=num_levels,
            filter_size=filter_size,
            sigma=sigma,
            omega_x=omega_x,
            omega_y=omega_y
        )

        self.directioanlfusion = DirectionalFusion(
            in_channels=in_channels,
            num_levels=num_levels,
            hidden_channels=hidden_channels,
            squeeze_ratio=squeeze_ratio
        )

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            temb_dim=temb_dim,
            dropout_ratio=dropout_ratio,
            shortcut=shortcut,
            trainable=trainable
        )

    def forward(
        self,
        input: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]:
        """
        Defines the forward pass of the ContourletDiffusion model.

        Args:
            input (torch.Tensor): The input tensor (e.g., noisy image), shape `(B, C, H, W)`.
            t (torch.Tensor): The time step tensor, shape `(B,)`.

        Returns:
            Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]: A tuple containing:
                - pyramid (List[torch.Tensor]): The Laplacian pyramid from the Contourlet transform.
                - subbands (List[List[torch.Tensor]]): The directional sub-bands from the Contourlet transform.
                - fusion (List[torch.Tensor]): The fused directional features used as skip connections.
                - pred (torch.Tensor): The final predicted output from the U-Net.
        """
        pyramid, subbands = self.contourlet(input)
        fusion = self.directioanlfusion(subbands)
        pred = self.unet(input, condition, t, fusion)
        return pyramid, subbands, fusion, pred
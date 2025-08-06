import torch
import torch.nn as nn

from model.blocks.contourlet import ContourletTransform
from model.blocks.directionalfusion import DirectionalFusion
from model.blocks.unet import UNet

class ContourletDiffusion(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        hidden_channels=64,
        num_levels=3,
        temb_dim=64,
        dropout=0.1,
        filter_size=5,
        sigma=1.0,
        omega_x=0.25,
        omega_y=0.25,
        squeeze_ratio=0.3,
        shortcut=True,
        trainable=False
        ):
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
        self.dropout = dropout
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
            dropout=dropout,
            shortcut=shortcut,
            trainable=trainable
        )

    def forward(self, x, t):
        pyramid, subbands = self.contourlet(x)
        fusion = self.directioanlfusion(subbands)
        pred = self.unet(x, t, pyramid, fusion)
        return pyramid, subbands, fusion, pred
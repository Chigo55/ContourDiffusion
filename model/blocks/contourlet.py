import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    def __init__(self, num_levels, filter_size, sigma, channels):
        """
        Initialize the Laplacian Pyramid  module.

        Args:
            num_levels (int): Number of levels in the Laplacian Pyramid.
            filter_size (int): Size of the Gaussian filter.
            sigma (float): Standard deviation of the Gaussian distribution.
            channels (int): Number of channels for the filter.
        """
        super().__init__()
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.channels = channels
        self.register_buffer( name='filter', tensor=self._gaussian_filter(filter_size=filter_size, sigma=sigma, channels=channels))

    def _gaussian_filter(self, filter_size, sigma, channels):
        """
        Generate a Gaussian filter.

        Args:
            filter_size (int): Size of the filter.
            sigma (float): Standard deviation of the Gaussian distribution.
            channels (int): Number of channels for the filter.

        Returns:
            torch.Tensor: Gaussian filter.

        Raises:
            AssertionError: If filter_size is not odd.
        """
        assert filter_size % 2 == 1, "Size must be odd"
        ax = torch.arange(end=filter_size, dtype=torch.float32) - (filter_size - 1) / 2
        xx, yy = torch.meshgrid(ax, ax)
        filter = torch.exp(input=-(xx**2 + yy**2) / (2 * sigma**2))
        filter = filter / torch.sum(input=filter)

        filter = filter.view(1, 1, filter_size, filter_size)
        filter = filter.repeat(channels, 1, 1, 1)
        return filter

    def forward(self, x):
        """
        Apply the Laplacian Pyramid to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            List[Tensor]: List of tensors representing the Laplacian pyramid.
        """
        c = x
        pyramid, laplacian = [], []

        for i in range(self.num_levels+1):
            p = c
            blurred = F.conv2d(input=c, weight=self.filter, padding=self.filter_size // 2, groups=self.channels)
            down = F.avg_pool2d(input=blurred, kernel_size=2, stride=2)
            up = F.interpolate(input=down, size=c.shape[-2:], mode='bilinear', align_corners=False)
            l = c - up
            c = down
            pyramid.append(p)
            laplacian.append(l)
        return pyramid, laplacian


class DirectionalFilterBank(nn.Module):
    def __init__(self, num_levels, filter_size, sigma, omega_x, omega_y, channels):
        """
        Initialize the Directional Filter Bank  module.

        Args:
            num_levels (int): Number of levels in the Directional Filter Bank.
            filter_size (int): Size of the filter.
            sigma (float): Standard deviation of the Gaussian distribution.
            omega_x (float): Frequency in the x-direction.
            omega_y (float): Frequency in the y-direction.
            channels (int): Number of channels for the filter.
        """
        super().__init__()
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.channels = channels
        self.register_buffer(name='filter', tensor=self._fan_filter(filter_size=filter_size, sigma=sigma, omega_x=omega_x, omega_y=omega_y, channels=channels))

    def _lowpass_filter(self, filter_size, sigma, channels):
        """
        Apply a low-pass filter to the input tensor.

        Args:
            filter_size (int): Size of the filter.
            sigma (float): Standard deviation of the Gaussian distribution.
            channels (int): Number of channels for the filter.

        Returns:
            torch.Tensor: Gaussian filter.

        Raises:
            AssertionError: If filter_size is not odd.
        """
        assert filter_size % 2 == 1, "Size must be odd"
        ax = torch.arange(end=filter_size, dtype=torch.float32) - (filter_size - 1) / 2
        xx, yy = torch.meshgrid(ax, ax)
        filter = torch.exp(input=-(xx**2 + yy**2) / (2 * sigma**2))
        filter = filter / torch.sum(input=filter)

        filter = filter.view(1, 1, filter_size, filter_size)
        filter = filter.repeat(channels, 1, 1, 1)
        return filter

    def _highpass_filter(self, lp_filter):
        """ Apply a high-pass filter to the input tensor.

        Args:
            lp_filter (torch.Tensor): Low-pass filter tensor.

        Returns:
            torch.Tensor: High-pass filtered tensor.
        """
        hp_filter = -lp_filter.clone()
        H, W = lp_filter.shape[-2:]
        center_h, center_w = H // 2, W // 2
        hp_filter[0, 0, center_h, center_w] += 1.0
        return hp_filter

    def _fan_filter(self, filter_size, sigma, omega_x, omega_y, channels):
        """
        Generate a fan filter for directional filtering.

        Args:
            filter_size (int): Size of the filter.
            sigma (float): Standard deviation of the Gaussian distribution.
            omega_x (float): Frequency in the x-direction.
            omega_y (float): Frequency in the y-direction.
            channels (int): Number of channels for the filter.

        Returns:
            torch.Tensor: Fan filter tensor.
        """
        lp_filter = self._lowpass_filter(filter_size=filter_size, sigma=sigma, channels=channels)
        hp_filter = self._highpass_filter(lp_filter=lp_filter)
        H, W = hp_filter.shape[-2:]
        coords_y = torch.arange(end=H, dtype=torch.float32) - H // 2
        coords_x = torch.arange(end=W, dtype=torch.float32) - W // 2
        y, x = torch.meshgrid(coords_y, coords_x, indexing='ij')

        modulation = torch.cos(input=2 * math.pi * (omega_x * x + omega_y * y))
        modulated = hp_filter * modulation
        return modulated

    def _apply_shear(self, x):
        """
        Apply shear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sheared tensor.
        """
        B, C, H, W = x.shape
        shear_matrix_pos = torch.tensor(data=[[1, 1, 0], [0, 1, 0]], dtype=torch.float32)
        shear_matrix_neg = torch.tensor(data=[[1, -1, 0], [0, 1, 0]], dtype=torch.float32)
        shear_matrix_pos = shear_matrix_pos.unsqueeze(0).repeat(B, 1, 1)
        shear_matrix_neg = shear_matrix_neg.unsqueeze(0).repeat(B, 1, 1)

        grid_pos = F.affine_grid(theta=shear_matrix_pos, size=x.size(), align_corners=False)
        grid_neg = F.affine_grid(theta=shear_matrix_neg, size=x.size(), align_corners=False)

        sheared_pos=  F.grid_sample(input=x, grid=grid_pos, align_corners=False)
        sheared_neg=  F.grid_sample(input=x, grid=grid_neg, align_corners=False)
        return sheared_pos, sheared_neg

    def _dfb(self, x):
        """
        Apply the directional filters to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the directional filters.
        """
        sheared_pos, sheared_neg = self._apply_shear(x=x)
        out1 = F.conv2d(input=sheared_pos, weight=self.filter, padding=self.filter_size // 2, groups=self.channels)
        out2 = F.conv2d(input=sheared_neg, weight=self.filter, padding=self.filter_size // 2, groups=self.channels)
        return out1, out2

    def _binary_dfb(self, x, level):
        """
        Apply the Directional Filter Bank to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[Tensor]: List of 2^num_levels directional subbands
        """
        if level == 0:
            return [x]
        d1, d2 = self._dfb(x=x)
        return self._binary_dfb(x=d1, level=level - 1) + self._binary_dfb(x=d2, level=level - 1)

    def forward(self, x):
        """
        Apply the Directional Filter Bank to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            List[Tensor]: List of 2^num_levels directional subbands
        """
        return self._binary_dfb(x=x, level=self.num_levels)


class ContourletTransform(nn.Module):
    def __init__(self, num_levels, filter_size, sigma, omega_x, omega_y, channels):
        """
        Initialize the Contourlet Transform module.

        Args:
            num_levels (int): Number of scales for the Laplacian Pyramid.
            filter_size (int): Size of the filter.
            sigma (float): Standard deviation of the Gaussian distribution.
            omega_x (float): Frequency in the x-direction.
            omega_y (float): Frequency in the y-direction.
            channels (int): Number of channels for the filter.
        """
        super().__init__()
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.channels = channels

        self.lp = LaplacianPyramid(num_levels=num_levels, filter_size=filter_size, sigma=sigma, channels=channels)
        self.dfb = DirectionalFilterBank(num_levels=num_levels, filter_size=filter_size, sigma=sigma, omega_x=omega_x, omega_y=omega_y, channels=channels)

    def forward(self, x):
        """
        Apply the Contourlet Transform to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            List[Tensor]: List containing the Laplacian pyramid and directional subbands.
        """
        pyramid, laplacian = self.lp(x=x)
        subbands = []
        for i in range(self.num_levels):
            subbands.append(self.dfb(x=laplacian[i]))
        return

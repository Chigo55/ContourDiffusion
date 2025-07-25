import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaplacianPyramid(nn.Module):
    """
    LaplacianPyramid constructs a multi-resolution Laplacian pyramid from an image tensor.

    Args:
        num_levels (int): Number of pyramid levels.
        filter_size (int): Size of the Gaussian blur filter (must be odd).
        sigma (float): Standard deviation for the Gaussian kernel.
        channels (int): Number of channels in the input tensor.
    """
    def __init__(self, num_levels, filter_size, sigma, channels):
        super().__init__()
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.channels = channels
        self.register_buffer( name='filter', tensor=self._gaussian_filter(filter_size=filter_size, sigma=sigma, channels=channels))

    def _gaussian_filter(self, filter_size, sigma, channels):
        """
        Generate a multi-channel Gaussian blur filter.

        Args:
            filter_size (int): Size of the Gaussian filter (must be odd).
            sigma (float): Standard deviation of the Gaussian distribution.
            channels (int): Number of output channels for the filter.

        Returns:
            torch.Tensor: A tensor of shape (channels, 1, filter_size, filter_size) containing
                          the separable Gaussian kernels.

        Raises:
            AssertionError: If filter_size is not odd.
        """
        assert filter_size % 2 == 1, "filter_size must be odd"
        ax = torch.arange(end=filter_size, dtype=torch.float32) - (filter_size - 1) / 2
        xx, yy = torch.meshgrid(ax, ax)
        filter = torch.exp(input=-(xx**2 + yy**2) / (2 * sigma**2))
        filter = filter / torch.sum(input=filter)

        filter = filter.view(1, 1, filter_size, filter_size)
        filter = filter.repeat(channels, 1, 1, 1)
        return filter

    def forward(self, x):
        """
        Build the Laplacian pyramid for the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Two lists:
                - pyramid: feature maps at each level.
                - laplacian: corresponding detail maps (input minus upsampled blur).
        """
        c = x
        pyramid, laplacian = [], []

        for i in range(self.num_levels):
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
    """
    DirectionalFilterBank applies directional frequency filtering at multiple scales
    using fan-shaped filters, shear transforms, and recursive binary filtering.

    Args:
        num_levels (int): Number of decomposition levels (depth of binary filtering).
        filter_size (int): Size of the directional filter (must be odd).
        sigma (float): Standard deviation for Gaussian low-pass kernel.
        omega_x (float): Modulation frequency in the x-direction.
        omega_y (float): Modulation frequency in the y-direction.
        channels (int): Number of input/output channels.
    """
    def __init__(self, num_levels, filter_size, sigma, omega_x, omega_y, channels):
        super().__init__()
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.channels = channels
        self.register_buffer(
            name='filter',
            tensor=self._fan_filter(
                filter_size=filter_size,
                sigma=sigma, omega_x=omega_x,
                omega_y=omega_y,
                channels=channels
            )
        )

    def _lowpass_filter(self, filter_size, sigma, channels):
        """
        Generate multi-channel Gaussian low-pass filter.

        Args:
            filter_size (int): Filter size (must be odd).
            sigma (float): Gaussian standard deviation.
            channels (int): Number of channels.

        Returns:
            torch.Tensor: Low-pass filter tensor of shape (channels, 1, filter_size, filter_size).

        Raises:
            AssertionError: If filter_size is not odd.
        """
        assert filter_size % 2 == 1, "filter_size must be odd"
        ax = torch.arange(end=filter_size, dtype=torch.float32) - (filter_size - 1) / 2
        xx, yy = torch.meshgrid(ax, ax)
        filter = torch.exp(input=-(xx**2 + yy**2) / (2 * sigma**2))
        filter = filter / torch.sum(input=filter)

        filter = filter.view(1, 1, filter_size, filter_size)
        filter = filter.repeat(channels, 1, 1, 1)
        return filter

    def _highpass_filter(self, lp_filter):
        """
        Generate high-pass filter from low-pass kernel.

        Args:
            lp_filter (torch.Tensor): Gaussian low-pass filter tensor.

        Returns:
            torch.Tensor: High-pass filter tensor of same shape.
        """
        hp_filter = -lp_filter.clone()
        H, W = lp_filter.shape[-2:]
        center_h, center_w = H // 2, W // 2
        hp_filter[0, 0, center_h, center_w] += 1.0
        return hp_filter

    def _fan_filter(self, filter_size, sigma, omega_x, omega_y, channels):
        """
        Create fan-shaped directional filter by modulating high-pass kernel.

        Args:
            filter_size (int): Filter size (odd).
            sigma (float): Gaussian standard deviation.
            omega_x (float): X-direction frequency.
            omega_y (float): Y-direction frequency.
            channels (int): Number of channels.

        Returns:
            torch.Tensor: Fan filter tensor of shape (channels, 1, filter_size, filter_size).
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
        Apply positive and negative shear transforms to input tensor.

        Args:
            x (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sheared tensors (positive, negative).
        """
        B, C, H, W = x.shape
        device = x.device
        shear_matrix_pos = torch.tensor(data=[[1, 1, 0], [0, 1, 0]], device=device, dtype=torch.float32)
        shear_matrix_neg = torch.tensor(data=[[1, -1, 0], [0, 1, 0]], device=device, dtype=torch.float32)
        shear_matrix_pos = shear_matrix_pos.unsqueeze(dim=0).repeat(B, 1, 1)
        shear_matrix_neg = shear_matrix_neg.unsqueeze(dim=0).repeat(B, 1, 1)

        grid_pos = F.affine_grid(theta=shear_matrix_pos, size=x.size(), align_corners=False)
        grid_neg = F.affine_grid(theta=shear_matrix_neg, size=x.size(), align_corners=False)

        sheared_pos=  F.grid_sample(input=x, grid=grid_pos, align_corners=False)
        sheared_neg=  F.grid_sample(input=x, grid=grid_neg, align_corners=False)
        return sheared_pos, sheared_neg

    def _dfb(self, x):
        """
        Perform one-stage directional filtering with fan filters.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Directional subband outputs.
        """
        sheared_pos, sheared_neg = self._apply_shear(x=x)
        out1 = F.conv2d(input=sheared_pos, weight=self.filter, padding=self.filter_size // 2, groups=self.channels)
        out2 = F.conv2d(input=sheared_neg, weight=self.filter, padding=self.filter_size // 2, groups=self.channels)
        return out1, out2

    def _binary_dfb(self, x, level):
        """
        Recursively apply directional filter bank to decompose into subbands.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            level (int): Remaining decomposition levels.

        Returns:
            List[torch.Tensor]: List of 2**level directional subbands.
        """
        if level == 0:
            return [x]
        d1, d2 = self._dfb(x=x)
        return self._binary_dfb(x=d1, level=level - 1) + self._binary_dfb(x=d2, level=level - 1)

    def forward(self, x):
        """
        Apply recursive directional filtering to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: 2**num_levels directional subband tensors.
        """
        return self._binary_dfb(x=x, level=self.num_levels)


class ContourletTransform(nn.Module):
    """
    ContourletTransform performs a Laplacian pyramid decomposition followed by
    directional sub-band extraction on the resulting detail images.

    Args:
        num_levels (int): Number of scales in the Laplacian pyramid.
        filter_size (int): Size of the directional filters (must be odd).
        sigma (float): Standard deviation for Gaussian kernel.
        omega_x (float): Modulation frequency in the x-direction.
        omega_y (float): Modulation frequency in the y-direction.
        channels (int): Number of channels in the input tensor.
    """
    def __init__(self, num_levels, filter_size, sigma, omega_x, omega_y, channels):
        super().__init__()
        self.num_levels = num_levels
        self.filter_size = filter_size
        self.sigma = sigma
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.channels = channels

        self.lp = LaplacianPyramid(num_levels=num_levels, filter_size=filter_size, sigma=sigma, channels=channels)
        self.dfb = nn.ModuleDict()

        for level in range(1, num_levels + 1):
            self.dfb[f"dfb{level}"] = DirectionalFilterBank(
                num_levels=level,
                filter_size=filter_size,
                sigma=sigma,
                omega_x=omega_x,
                omega_y=omega_y,
                channels=channels
            )

    def forward(self, x):
        """
        Apply the Contourlet Transform to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
                - pyramid: List of Laplacian pyramid maps (length num_levels+1).
                - subbands: List of lists, each containing directional sub-bands
                  corresponding to each detail map.
        """
        pyramid, laplacian = self.lp(x=x)
        subbands = []

        for i, l in enumerate(iterable=laplacian, start=1):
            subbands.append(self.dfb[f"dfb{i}"](l))
        return pyramid, subbands[::-1]

import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class MeanAbsoluteError(nn.L1Loss):
    """
    A wrapper around the `torch.nn.L1Loss` module.

    This loss computes the mean absolute error (L1 loss) between each element in
    the input and target tensors.
    """
    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initializes the MeanAbsoluteError loss.

        Args:
            *args: Variable length argument list for `nn.L1Loss`.
            **kwargs: Arbitrary keyword arguments for `nn.L1Loss`.
        """
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the L1 loss.

        Args:
            input (torch.Tensor): The predicted tensor of shape `(...)`.
            target (torch.Tensor): The ground truth tensor of the same shape as input.

        Returns:
            torch.Tensor: The scalar L1 loss value.
        """
        return super().forward(input=input, target=target)


class TotalVariance(nn.Module):
    """
    Computes the total variation (TV) loss, which is a measure of the
    noise in an image. It encourages spatial smoothness in the generated image.
    """
    def __init__(self) -> None:
        """Initializes the TotalVariance loss module."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the total variation of the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar total variation loss.
        """
        B, C, H, W = x.shape

        count_H = (H - 1) * W
        count_W = H * (W - 1)

        tv_H = torch.pow(input=(x[:, :, 1:, :] - x[:, :, :H - 1, :]), exponent=2).sum()
        tv_W = torch.pow(input=(x[:, :, :, 1:] - x[:, :, :, :W - 1]), exponent=2).sum()
        return 2 * (tv_H / count_H + tv_W / count_W) / B


class NoiseLoss(nn.Module):
    """
    Combines L1 loss and Total Variance (TV) loss. This is often used
    in image denoising tasks to penalize pixel-wise differences and noise.
    """
    def __init__(
        self,
        l1_weight: float = 0.8,
        tv_weight: float = 0.2
    ) -> None:
        """
        Initializes the NoiseLoss module.

        Args:
            l1_weight (float): The weight for the L1 loss component. Defaults to 0.8.
            tv_weight (float): The weight for the Total Variance loss component. Defaults to 0.2.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.tv_weight = tv_weight
        self.l1_loss = MeanAbsoluteError()
        self.tv_loss = TotalVariance()

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the combined noise loss as a weighted sum of L1 and TV losses.

        The loss is a weighted sum of the L1 loss between predictions and targets,
        and the Total Variance loss of the predictions.

        Args:
            preds (torch.Tensor): The predicted images tensor of shape `(B, C, H, W)`.
            targets (torch.Tensor): The ground truth images tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar combined noise loss.
        """
        l1_loss_val = self.l1_loss(input=preds, target=targets)
        tv_loss_val = self.tv_loss(preds)
        return self.l1_weight * l1_loss_val + self.tv_weight * tv_loss_val


class MeanSquaredError(nn.MSELoss):
    """
    A wrapper around the `torch.nn.MSELoss` module.

    This loss computes the mean squared error (L2 loss) between each element in
    the input and target tensors.
    """
    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Initializes the MeanSquaredError loss.

        Args:
            *args: Variable length argument list for `nn.MSELoss`.
            **kwargs: Arbitrary keyword arguments for `nn.MSELoss`.
        """
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the L2 loss.

        Args:
            input (torch.Tensor): The predicted tensor of shape `(...)`.
            target (torch.Tensor): The ground truth tensor of the same shape as input.

        Returns:
            torch.Tensor: The scalar L2 loss value.
        """
        return super().forward(input=input, target=target)


class PhaseSpectrum(nn.Module):
    """
    Computes the L1 loss between the phase spectra of two sets of images.

    The phase spectrum is obtained by taking the angle of the 2D Fast Fourier Transform.
    """
    def __init__(self) -> None:
        """Initializes the PhaseSpectrum loss module."""
        super().__init__()

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the L1 loss between the phase spectra of predictions and targets.

        Args:
            preds (torch.Tensor): The predicted images tensor of shape `(B, C, H, W)`.
            targets (torch.Tensor): The ground truth images tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar phase spectrum loss.
        """
        preds_fft = torch.fft.fft2(preds, dim=(-2, -1))
        targets_fft = torch.fft.fft2(targets, dim=(-2, -1))
        preds_phase = torch.angle(input=preds_fft)
        targets_phase = torch.angle(input=targets_fft)

        return F.l1_loss(input=preds_phase, target=targets_phase)


class FrequencyLoss(nn.Module):
    """
    Combines a magnitude-based loss (L2) and a phase-based loss (L1)
    in the frequency domain.
    """
    def __init__(
        self,
        l2_weight: float = 0.8,
        ps_weight: float = 0.2
    ) -> None:
        """
        Initializes the FrequencyLoss module.

        Args:
            l2_weight (float): The weight for the magnitude loss component. Defaults to 0.8.
            ps_weight (float): The weight for the phase loss component. Defaults to 1.0.
        """
        super().__init__()
        self.l2_weight = l2_weight
        self.ps_weight = ps_weight
        self.l2_loss = MeanSquaredError()
        self.ps_loss = PhaseSpectrum()

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the combined frequency domain loss.

        Args:
            preds (torch.Tensor): The predicted images tensor of shape `(B, C, H, W)`.
            targets (torch.Tensor): The ground truth images tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar combined frequency loss.
        """
        l2_loss_val = self.l2_loss(input=preds, target=targets)
        ps_loss_val = self.ps_loss(input=preds, target=targets)
        return self.l2_weight * l2_loss_val + self.ps_weight * ps_loss_val


class StructuralSimilarity(nn.Module):
    """
    Computes the Structural Similarity (SSIM) loss between two images using `pyiqa`.
    """
    def __init__(
        self,
        device: str = "cuda"
    ) -> None:
        """
        Initializes the StructuralSimilarity loss module.

        Args:
            device (str): The device to run the metric on, e.g., "cuda" or "cpu".
                Defaults to "cuda".
        """
        super().__init__()
        self.ssim = pyiqa.create_metric(
            metric_name='ssim',
            device=device,
            as_loss=True
        )

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the SSIM loss.

        Args:
            preds (torch.Tensor): The predicted images tensor of shape `(B, C, H, W)`.
            targets (torch.Tensor): The ground truth images tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar SSIM loss.
        """
        return 1 - self.ssim(preds, targets)


class PerceptualSimilarity(nn.Module):
    """
    Computes the Learned Perceptual Image Patch Similarity (LPIPS)
    loss between two images using a pre-trained VGG network.
    """
    def __init__(
        self,
        device: str = "cuda"
    ) -> None:
        """
        Initializes the PerceptualSimilarity loss module.

        Args:
            device (str): The device to run the metric on, e.g., "cuda" or "cpu".
                Defaults to "cuda".
        """
        super().__init__()
        self.lpips = pyiqa.create_metric(
            metric_name='lpips-vgg',
            device=device,
            as_loss=True
        )

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the LPIPS loss.

        Args:
            preds (torch.Tensor): The predicted images tensor of shape `(B, C, H, W)`.
            targets (torch.Tensor): The ground truth images tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar LPIPS loss.
        """
        return self.lpips(preds, targets)


class StructuralLoss(nn.Module):
    """
    Computes a weighted combination of Structural Similarity (SSIM)
    and Perceptual Similarity (LPIPS) losses.
    """
    def __init__(
        self,
        ssim_weight: float = 0.8,
        lpips_weight: float = 0.2,
        device: str = "cuda"
    ) -> None:
        """
        Initializes the StructuralLoss module.

        Args:
            ssim_weight (float): The weight for the SSIM loss component. Defaults to 0.8.
            lpips_weight (float): The weight for the LPIPS loss component. Defaults to 0.2.
            device (str): The device to run the metrics on, e.g., "cuda" or "cpu".
                Defaults to "cuda".
        """
        super().__init__()
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.ss_loss = StructuralSimilarity(device=device)
        self.ps_loss = PerceptualSimilarity(device=device)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the combined structural loss.

        Args:
            preds (torch.Tensor): The predicted images tensor of shape `(B, C, H, W)`.
            targets (torch.Tensor): The ground truth images tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor: The scalar combined structural loss.
        """
        ssim_loss = self.ss_loss(preds, targets)
        lpips_loss = self.ps_loss(preds, targets)
        return self.ssim_weight * ssim_loss + self.lpips_weight * lpips_loss
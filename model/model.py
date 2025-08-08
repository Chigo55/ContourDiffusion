import torch
import torch.nn as nn
import lightning as L
from typing import Any, Dict, Tuple, List

from torch.optim.sgd import SGD
from torch.optim.asgd import ASGD
from torch.optim.rmsprop import RMSprop
from torch.optim.rprop import Rprop
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.adamax import Adamax
from torch.optim.adagrad import Adagrad
from torch.optim.adadelta import Adadelta
from torch.optim.lbfgs import LBFGS
from torch.optim import Optimizer

from model.loss import NoiseLoss, FrequencyLoss, StructuralLoss
from model.blocks import ContourletDiffusion
from utils.metrics import ImageQualityMetrics



class ContourletDiffusionLightning(L.LightningModule):
    """
    A PyTorch Lightning module for the ContourletDiffusion model.

    This module wraps the core `ContourletDiffusion` model and handles the training,
    validation, testing, and prediction logic. It defines the loss functions,
    configures the optimizers, and manages logging of metrics and images.

    Note: The `hparams` dictionary is used extensively. For better type safety and
    clarity, consider using a structured configuration object (e.g., with `dataclasses`
    or `pydantic`) in the future.
    """
    def __init__(
        self,
        hparams: Dict
    ) -> None:
        """
        Initializes the ContourletDiffusionLightning module.

        Args:
            hparams (Dict): A dictionary containing all hyperparameters.
        """
        super().__init__()
        self.hparam = hparams
        self.save_hyperparameters(hparams)

        self.model = ContourletDiffusion(
            in_channels=self.hparam['in_channels'],
            out_channels=self.hparam['out_channels'],
            hidden_channels=self.hparam['hidden_channels'],
            num_levels=self.hparam['num_levels'],
            temb_dim=self.hparam['temb_dim'],
            dropout_ratio=self.hparam['dropout_ratio'],
            filter_size=self.hparam['filter_size'],
            sigma=self.hparam['sigma'],
            omega_x=self.hparam['omega_x'],
            omega_y=self.hparam['omega_y'],
            squeeze_ratio=self.hparam['squeeze_ratio'],
            shortcut=self.hparam['shortcut'],
            trainable=self.hparam['trainable']
        )

        self.noise_loss = NoiseLoss()
        self.freq_loss = FrequencyLoss()
        self.struc_loss = StructuralLoss()

        self.metric = ImageQualityMetrics(device=self.hparam['device'])

    def forward(
        self,
        input: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]:
        """
        Performs a forward pass through the ContourletDiffusion model.

        Args:
            input (torch.Tensor): The input tensor (e.g., low-light image), shape `(B, C, H, W)`.
            t (torch.Tensor): The time step tensor, shape `(B,)`.

        Returns:
            Tuple[...]: The output from the ContourletDiffusion model, containing the
                pyramid, sub-bands, fusion features, and the final prediction.
        """
        return self.model(input, condition, t)

    def _add_noise(self, high_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = torch.linspace(start=1e-4, end=0.02, steps=self.hparam["timesteps"], device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(input=alphas, dim=0)

        sqrt_alphas_cumprod = torch.sqrt(input=alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(input=1.0 - alphas_cumprod)

        t = torch.randint(low=0, high=self.hparam["timesteps"], size=(high_img.shape[0],), device=self.device)

        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t, None, None, None]
        epsilon = torch.randn_like(input=high_img)
        noisy_image = sqrt_alphas_cumprod_t * high_img + sqrt_one_minus_alphas_cumprod_t * epsilon

        return noisy_image, epsilon, t

    def _calculate_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates and aggregates the different loss components.

        Args:
            pred (torch.Tensor): The model's prediction.
            target (torch.Tensor): The ground truth tensor.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing noise_loss, freq_loss,
                struc_loss, and the total combined loss.
        """
        loss_noise = self.noise_loss(pred, target)
        loss_freq = self.freq_loss(pred, target)
        loss_struc = self.struc_loss(pred, target)

        loss_tot = (
            loss_noise +
            loss_freq +
            loss_struc
        )
        return loss_noise, loss_freq, loss_struc, loss_tot

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the low-light
                input images and high-quality target images.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The total loss for this training step.
        """
        low_img, high_img = batch

        noisy_img, epsilon, t = self._add_noise(high_img=high_img)

        _, _, _, pred = self.forward(input=noisy_img, condition=low_img, t=t)

        loss_noise, loss_freq, loss_struc, loss_tot = self._calculate_loss(pred=pred, target=high_img)
        if batch_idx % 250 == 0:
            self.logger.experiment.add_images(
                "train/1_target",
                high_img,
                self.global_step
            )
            self.logger.experiment.add_images(
                "train/2_pred",
                pred,
                self.global_step
            )

        self.log_dict(dictionary={
            "train/1_noise_loss": loss_noise,
            "train/2_freq_loss": loss_freq,
            "train/3_struc_loss": loss_struc,
            "train/4_total_loss": loss_tot,
        }, prog_bar=True)
        return loss_tot

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Performs a single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing the low-light
                input images and high-quality target images.
            batch_idx (int): The index of the current batch.
        """
        low_img, high_img = batch

        noisy_img, epsilon, t = self._add_noise(high_img=high_img)

        pyramid, subbands, fusion, pred = self.forward(input=noisy_img, condition=low_img, t=t)

        loss_noise, loss_freq, loss_struc, loss_tot = self._calculate_loss(pred=pred, target=high_img)

        self.log_dict(dictionary={
            "valid/1_noise_loss": loss_noise,
            "valid/2_freq_loss": loss_freq,
            "valid/3_struc_loss": loss_struc,
            "valid/4_total_loss": loss_tot,
        }, prog_bar=True)

        if batch_idx % 250 == 0:
            self.logger.experiment.add_images(
                "valid/1_target",
                high_img,
                self.global_step
            )
            self.logger.experiment.add_images(
                "valid/2_pred",
                pred,
                self.global_step
            )
            # Note: Logging pyramid, subbands, and fusion might require visualization utilities
            # like `torchvision.utils.make_grid` as they are lists of tensors.
            # self.logger.experiment.add_image(
            #     "valid/3_pyramid",
            #     pyramid,
            #     self.global_step
            # )
            # self.logger.experiment.add_image(
            #     "valid/4_subbands",
            #     subbands,
            #     self.global_step
            # )
            # self.logger.experiment.add_image(
            #     "valid/5_fusion",
            #     fusion,
            #     self.global_step
            # )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, float]:
        """
        Performs a single test step.

        Note: This method requires a full sampling loop to generate an image from the
        diffusion model, which is not implemented here. The current logic appears to be
        from a different, non-diffusion model and will not work as is.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Low-light and high-quality image pair.
            batch_idx (int): The index of the current batch.
            dataloader_idx (int): The index of the current dataloader.

        Returns:
            Dict[str, float]: A dictionary of image quality metrics.
        """
        low_img, high_img = batch

        # TODO: Implement the full reverse diffusion (sampling) process here.
        # This typically involves a loop from T to 0, calling the model at each step.
        # The code below is a placeholder and will raise an error.
        raise NotImplementedError("The sampling loop for the diffusion model is not implemented in `test_step`.")

        # Placeholder for the final enhanced image from the sampling loop
        # enh_img = self.sample(low_img)

        metrics = self.metric.full(preds=enh_img, targets=high_img)

        self.log_dict(dictionary={
            "bench/1_PSNR": metrics["PSNR"],
            "bench/2_SSIM": metrics["SSIM"],
            "bench/3_LPIPS": metrics["LPIPS"],
            "bench/4_NIQE": metrics["NIQE"],
            "bench/5_BRISQUE": metrics["BRISQUE"],
        }, prog_bar=True)
        return metrics

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Performs a single prediction step.

        Note: This method also requires the full sampling loop.
        """
        low_img, _ = batch
        # TODO: Implement the full reverse diffusion (sampling) process here.
        raise NotImplementedError("The sampling loop for the diffusion model is not implemented in `predict_step`.")
        # return self.sample(low_img)

    def configure_optimizers(self) -> Optimizer:
        """
        Configures the optimizer for the model based on the hyperparameters.

        Returns:
            Optimizer: The configured optimizer.
        """
        optim_name = self.hparam['optim'].lower()
        lr = self.hparam.get('lr', 1e-4)
        weight_decay = self.hparam.get('decay', 1e-5)
        eps = self.hparam.get('eps', 1e-8)

        if optim_name == "sgd":
            return SGD(
                params=self.parameters(),
                lr=lr,
                momentum=self.hparam.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_name == "asgd":
            return ASGD(
                params=self.parameters(),
                lr=lr,
                lambd=self.hparam.get('lambd', 1e-8),
                alpha=self.hparam.get('alpha', 0.75),
                t0=self.hparam.get('t0', 1e6),
                weight_decay=weight_decay
            )
        elif optim_name == "rmsprop":
            return RMSprop(
                params=self.parameters(),
                lr=lr,
                alpha=self.hparam.get('alpha', 0.99),
                eps=eps,
                momentum=self.hparam.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_name == "rprop":
            return Rprop(
                params=self.parameters(),
                lr=lr,
                etas=self.hparam.get('etas', (0.5, 1.2)),
                step_sizes=self.hparam.get('step_sizes', (1e-9, 50))
            )
        elif optim_name == "adam":
            return Adam(
                params=self.parameters(),
                lr=lr,
                betas=self.hparam.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adamw":
            return AdamW(
                params=self.parameters(),
                lr=lr,
                betas=self.hparam.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adamax":
            return Adamax(
                params=self.parameters(),
                lr=lr,
                betas=self.hparam.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adagrad":
            return Adagrad(
                params=self.parameters(),
                lr=lr,
                lr_decay=self.hparam.get('lr_decay', 1e-9),
                weight_decay=weight_decay,
                eps=eps
            )
        elif optim_name == "adadelta":
            return Adadelta(
                params=self.parameters(),
                lr=lr,
                rho=self.hparam.get('rho', 0.9),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "lbfgs":
            return LBFGS(
                params=self.parameters(),
                lr=lr,
                max_iter=self.hparam.get('max_iter', 20),
                max_eval=self.hparam.get('max_eval', None),
                tolerance_grad=self.hparam.get('tolerance_grad', 1e-8),
                tolerance_change=self.hparam.get('tolerance_change', 1e-9),
                history_size=self.hparam.get('history_size', 100),
                line_search_fn=self.hparam.get('line_search_fn', None)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

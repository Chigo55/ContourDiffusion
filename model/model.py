import torch
import torch.nn as nn
import lightning as L

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

from model.loss import NoiseLoss, FrequencyLoss, StructuralLoss
from model.blocks import ContourletDiffusion
from utils.metrics import ImageQualityMetrics



class ContourletDiffusionLightning(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = ContourletDiffusion(
            in_channels=hparams['in_channels'],
            out_channels=hparams['out_channels'],
            hidden_channels=hparams['hidden_channels'],
            num_levels=hparams['num_levels'],
            temb_dim=hparams['temb_dim'],
            dropout=hparams['dropout'],
            filter_size=hparams['filter_size'],
            sigma=hparams['sigma'],
            omega_x=hparams['omega_x'],
            omega_y=hparams['omega_y'],
            squeeze_ratio=hparams['squeeze_ratio'],
            shortcut=hparams['shortcut'],
            trainable=hparams['trainable']
        )

        self.noise_loss = NoiseLoss()
        self.freq_loss = FrequencyLoss()
        self.struc_loss = StructuralLoss()

        self.metric = ImageQualityMetrics(device=hparams['device'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        pyramid, subbands, fusion, pred = self(x)

        loss_noise = self.noise_loss(pred, x)
        loss_freq = self.freq_loss(pred)
        loss_struc = self.struc_loss(pred)

        loss_tot = (
            loss_noise +
            loss_freq +
            loss_struc
        )

        self.log_dict(dictionary={
            "train/1_spa": loss_noise,
            "train/2_col": loss_freq,
            "train/3_exp": loss_struc,
            "train/4_tot": loss_tot,
        }, prog_bar=True)
        return loss_tot

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        pyramid, subbands, fusion, pred = self(x)

        loss_noise = self.noise_loss(pred, x)
        loss_freq = self.freq_loss(pred)
        loss_struc = self.struc_loss(pred)

        loss_tot = (
            loss_noise +
            loss_freq +
            loss_struc
        )

        self.log_dict(dictionary={
            "valid/1_spa": loss_noise,
            "valid/2_col": loss_freq,
            "valid/3_exp": loss_struc,
            "valid/4_tot": loss_tot,
        }, prog_bar=True)

        if batch_idx % 250 == 0:
            self.logger.experiment.add_image(
                "valid/1_target",
                x,
                self.global_step
            )
            self.logger.experiment.add_image(
                "valid/2_pred",
                pred,
                self.global_step
            )
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
        return loss_tot

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)

        metrics = self.metric.full(preds=enh_img, targets=x)

        self.log_dict(dictionary={
            "bench/1_PSNR": metrics["PSNR"],
            "bench/2_SSIM": metrics["SSIM"],
            "bench/3_LPIPS": metrics["LPIPS"],
            "bench/4_NIQE": metrics["NIQE"],
            "bench/5_BRISQUE": metrics["BRISQUE"],
        }, prog_bar=True)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch.to(self.device)
        Y, Cr, Cb, x_i, x_d, o_i, n_i, n_Y, enh_img = self(x)
        return enh_img

    def configure_optimizers(self):
        optim_name = self.hparams['optim'].lower()
        lr = self.hparams.get('lr', 1e-4)
        weight_decay = self.hparams.get('decay', 1e-5)
        eps = self.hparams.get('eps', 1e-8)

        if optim_name == "sgd":
            return SGD(
                params=self.parameters(),
                lr=lr,
                momentum=self.hparams.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_name == "asgd":
            return ASGD(
                params=self.parameters(),
                lr=lr,
                lambd=self.hparams.get('lambd', 1e-8),
                alpha=self.hparams.get('alpha', 0.75),
                t0=self.hparams.get('t0', 1e6),
                weight_decay=weight_decay
            )
        elif optim_name == "rmsprop":
            return RMSprop(
                params=self.parameters(),
                lr=lr,
                alpha=self.hparams.get('alpha', 0.99),
                eps=eps,
                momentum=self.hparams.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optim_name == "rprop":
            return Rprop(
                params=self.parameters(),
                lr=lr,
                etas=self.hparams.get('etas', (0.5, 1.2)),
                step_sizes=self.hparams.get('step_sizes', (1e-9, 50))
            )
        elif optim_name == "adam":
            return Adam(
                params=self.parameters(),
                lr=lr,
                betas=self.hparams.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adamw":
            return AdamW(
                params=self.parameters(),
                lr=lr,
                betas=self.hparams.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adamax":
            return Adamax(
                params=self.parameters(),
                lr=lr,
                betas=self.hparams.get('betas', (0.9, 0.999)),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "adagrad":
            return Adagrad(
                params=self.parameters(),
                lr=lr,
                lr_decay=self.hparams.get('lr_decay', 1e-9),
                weight_decay=weight_decay,
                eps=eps
            )
        elif optim_name == "adadelta":
            return Adadelta(
                params=self.parameters(),
                lr=lr,
                rho=self.hparams.get('rho', 0.9),
                eps=eps,
                weight_decay=weight_decay
            )
        elif optim_name == "lbfgs":
            return LBFGS(
                params=self.parameters(),
                lr=lr,
                max_iter=self.hparams.get('max_iter', 20),
                max_eval=self.hparams.get('max_eval', None),
                tolerance_grad=self.hparams.get('tolerance_grad', 1e-8),
                tolerance_change=self.hparams.get('tolerance_change', 1e-9),
                history_size=self.hparams.get('history_size', 100),
                line_search_fn=self.hparams.get('line_search_fn', None)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

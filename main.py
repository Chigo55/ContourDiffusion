import random
import torch
from engine import LightningEngine
from model.model import *

torch.set_float32_matmul_precision(precision='high')

def get_hparams():
    hparams = {
        # 모델 구조
        "in_channels": 3,
        "out_channels": 3,
        "hidden_channels": 64,
        "num_levels": 4,
        "temb_dim": 64,
        "dropout_ratio": 0.1,
        "filter_size": 5,
        "sigma": 1.0,
        "omega_x": 0.25,
        "omega_y": 0.25,
        "squeeze_ratio": 0.3,
        "shortcut": True,
        "trainable": False,

        # 데이터 모듈
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",
        "image_size":256,
        "batch_size": 1,
        "num_workers": 10,

        # 엔진
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": 1,
        "precision": 32,
        "log_every_n_steps": 5,
        "gradient_clip_val": 0.1,
        "save_dir": "./runs/HomomorphicUnet/optims/st",
        "experiment_name": "test",

        # 모델 모듈
        "device": "cuda",
        "timesteps": 1000,

        # 최적화 및 학습 설정
        "optim": "sgd",
        "lr": 1e-7,
        "decay": 1e-8,
        "epochs": 100,
        "patience": 30,
        "batch_size": 16,
        "seed": 42,

        # 데이터 경로
        "train_data_path": "data/1_train",
        "valid_data_path": "data/2_valid",
        "bench_data_path": "data/3_bench",
        "infer_data_path": "data/4_infer",

        # 로깅 설정
        "log_dir": "./runs/HomomorphicUnet/optims",
        "experiment_name": "test",
        "inference": "inference",
    }
    return hparams


def main():
    hparams = get_hparams()
    opts = ["sgd", "asgd", "rmsprop", "rprop",
            "adam", "adamw", "adamax", "adadelta"]

    for opt in opts:
        print(f"\n[STARTING] Optimizer: {opt}")
        hparams["optim"] = opt
        hparams["experiment_name"] = opt

        engin = LightningEngine(
            model=ContourletDiffusionLightning,
            hparams=hparams
        )

        engin.train()
        # engin.valid()
        # engin.bench()
        # engin.infer()


if __name__ == "__main__":
    main()

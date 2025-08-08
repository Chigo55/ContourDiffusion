from abc import ABC, abstractmethod
from typing import Optional, Type
from lightning import LightningModule, Trainer

from data.dataloader import LowLightDataModule


class _BaseRunner(ABC):
    """
    An abstract base class for runners that execute different stages of a PyTorch Lightning
    workflow, such as training, validation, or inference. It handles the common setup
    for model and datamodule initialization.
    """
    def __init__(
        self,
        model: Type[LightningModule],
        trainer: Trainer,
        hparams: dict,
        checkpoint_path: Optional[str] = None,
        default_checkpoint_name: Optional[str] = None,
    ) -> None:
        """
        Initializes the _BaseRunner.

        Args:
            model (Type[LightningModule]): The LightningModule class to be instantiated.
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            hparams (dict): A dictionary of hyperparameters.
            checkpoint_path (Optional[str]): Path to a model checkpoint to load.
            default_checkpoint_name (Optional[str]): The default checkpoint name to use if
                `checkpoint_path` is not provided.
        """
        self.trainer = trainer
        self.hparams = hparams

        if checkpoint_path:
            self.model = model.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                map_location="cpu",
            )
            self.checkpoint_path = checkpoint_path
        else:
            self.model = model(hparams=hparams)
            self.checkpoint_path = default_checkpoint_name

        self.datamodule = self._build_datamodule()

    def _build_datamodule(self) -> LowLightDataModule:
        """
        Builds and configures the data module for the experiment.

        Returns:
            LowLightDataModule: An instance of the data module.
        """
        datamodule = LowLightDataModule(
            train_dir=self.hparams["train_data_path"],
            valid_dir=self.hparams["valid_data_path"],
            infer_dir=self.hparams["infer_data_path"],
            bench_dir=self.hparams["bench_data_path"],
            image_size=self.hparams["image_size"],
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
        )

        return datamodule

    @abstractmethod
    def run(self) -> None:
        """
        The main execution method for the runner. Subclasses must implement this to
        define the specific logic for their stage (e.g., `trainer.fit()`).
        """
        raise NotImplementedError


class LightningTrainer(_BaseRunner):
    """A runner for executing the training process (`trainer.fit()`)."""
    def run(self) -> None:
        """
        Starts the model training process using the configured trainer, model,
        and datamodule.
        """
        print("[INFO] Start Training...")
        self.trainer.fit(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Training Completed.")


class LightningValidater(_BaseRunner):
    """A runner for executing the validation process (`trainer.validate()`)."""
    def run(self) -> None:
        """
        Starts the model validation process using the configured trainer, model,
        and datamodule.
        """
        print("[INFO] Start Validating...")
        self.trainer.validate(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Validation Completed.")


class LightningBenchmarker(_BaseRunner):
    """A runner for executing the benchmarking process (`trainer.test()`)."""
    def run(self) -> None:
        """
        Starts the model benchmarking (testing) process using the configured trainer,
        model, and datamodule.
        """
        print("[INFO] Start Benchmarking...")
        self.trainer.test(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Benchmark Completed.")


class LightningInferencer(_BaseRunner):
    """A runner for executing the inference process (`trainer.predict()`)."""
    def run(self) -> None:
        """
        Starts the model inference process using the configured trainer, model,
        and datamodule.
        """
        print("[INFO] Start Inferencing...")
        self.trainer.predict(
            model=self.model,
            datamodule=self.datamodule,
            ckpt_path=self.checkpoint_path,
        )
        print("[INFO] Inference Completed.")

from typing import Optional, Type, List

from lightning import Trainer, LightningModule, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from lightning.pytorch.loggers import TensorBoardLogger

from engine.runner import _BaseRunner, LightningTrainer, LightningValidater, LightningBenchmarker, LightningInferencer


class LightningEngine:
    """
    A high-level engine to orchestrate the training, validation, benchmarking, and inference
    processes using PyTorch Lightning. It sets up the trainer, logger, and callbacks based on
    hyperparameters and manages different execution modes through dedicated runner classes.
    """
    def __init__(
        self,
        model: Type[LightningModule],
        hparams: dict,
        checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Initializes the LightningEngine.

        Args:
            model (Type[LightningModule]): The LightningModule class to be instantiated.
            hparams (dict): A dictionary containing all hyperparameters for the model,
                trainer, and data.
            checkpoint_path (Optional[str], optional): Path to a checkpoint to resume
                training or for validation/inference. Defaults to None.
        """
        self.model = model
        self.hparams = hparams
        self.checkpoint_path = checkpoint_path

        seed_everything(seed=self.hparams["seed"], workers=True)

        self.logger = self._build_logger()
        self.callbacks = self._build_callbacks()

        self.trainer = self._set_build_trainer()

    def _set_build_trainer(self) -> Trainer:
        """
        Configures and initializes the PyTorch Lightning Trainer.

        Returns:
            Trainer: An instance of the `lightning.Trainer` configured with the
                provided hyperparameters.
        """
        return Trainer(
            max_epochs=self.hparams["epochs"],
            accelerator=self.hparams["accelerator"],
            devices=self.hparams["devices"],
            precision=self.hparams["precision"],
            log_every_n_steps=self.hparams["log_every_n_steps"],
            gradient_clip_val=self.hparams["gradient_clip_val"],
            logger=self.logger,
            callbacks=self.callbacks,
        )

    def _build_logger(self) -> TensorBoardLogger:
        """
        Initializes the TensorBoard logger.

        Returns:
            TensorBoardLogger: An instance of the logger for tracking experiments.
        """
        return TensorBoardLogger(
            save_dir=self.hparams["log_dir"],
            name=self.hparams["experiment_name"]
        )

    def _build_callbacks(self) -> List[Callback]:
        """
        Initializes the list of PyTorch Lightning callbacks.

        This includes callbacks for model checkpointing, early stopping, and
        learning rate monitoring.

        Returns:
            List[Callback]: A list of configured callback instances.
        """
        return [
            ModelCheckpoint(
                monitor="valid/5_tot",
                save_top_k=1,
                mode="min",
                filename="best-{epoch:02d}",
            ),
            ModelCheckpoint(
                every_n_epochs=1,
                save_top_k=-1,
                filename="epoch-{epoch:02d}",
            ),
            EarlyStopping(
                monitor="valid/5_tot",
                patience=self.hparams["patience"],
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ]

    def _create_and_run_runner(
        self,
        runner_class: Type[_BaseRunner]
    ) -> None:
        """
        Creates an instance of a runner class and executes its `run` method.

        This is a helper method to reduce code duplication in the `train`, `valid`,
        `bench`, and `infer` methods.

        Args:
            runner_class (Type[_BaseRunner]): The runner class to instantiate (e.g.,
                LightningTrainer, LightningValidater).
        """
        runner = runner_class(
            model=self.model,
            trainer=self.trainer,
            hparams=self.hparams,
            checkpoint_path=self.checkpoint_path
        )
        runner.run()

    def train(self) -> None:
        """
        Starts the training process by creating and running a LightningTrainer.
        """
        self._create_and_run_runner(runner_class=LightningTrainer)

    def valid(self) -> None:
        """
        Starts the validation process by creating and running a LightningValidater.
        """
        self._create_and_run_runner(runner_class=LightningValidater)

    def bench(self) -> None:
        """
        Starts the benchmarking process by creating and running a LightningBenchmarker.
        """
        self._create_and_run_runner(runner_class=LightningBenchmarker)

    def infer(self) -> None:
        """
        Starts the inference process by creating and running a LightningInferencer.
        """
        self._create_and_run_runner(runner_class=LightningInferencer)

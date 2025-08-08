import lightning as L

from pathlib import Path
from typing import List, Literal, Optional, Union, overload
from torch.utils.data import DataLoader, ConcatDataset

from data.utils import LowLightDataset


class LowLightDataModule(L.LightningDataModule):
    """
    A LightningDataModule for loading low-light image enhancement datasets.

    This module handles the setup of training, validation, benchmarking, and inference
    datasets and dataloaders. It expects a directory structure where each subdirectory
    under the main path (e.g., train_dir) contains 'low' and 'high' subfolders with
    corresponding low-light and high-quality images.
    """

    def __init__(
        self,
        train_dir: Union[str, Path],
        valid_dir: Union[str, Path],
        bench_dir: Union[str, Path],
        infer_dir: Union[str, Path],
        image_size: int,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        """
        Initializes the LowLightDataModule.

        Args:
            train_dir (Union[str, Path]): Path to the training data directory.
            valid_dir (Union[str, Path]): Path to the validation data directory.
            bench_dir (Union[str, Path]): Path to the benchmark (test) data directory.
            infer_dir (Union[str, Path]): Path to the inference data directory.
            image_size (int): The size to which images will be resized.
            batch_size (int, optional): The number of samples per batch. Defaults to 32.
            num_workers (int, optional): The number of subprocesses to use for data loading.
                Defaults to 4.
        """
        super().__init__()
        self.train_dir = Path(train_dir)
        self.valid_dir = Path(valid_dir)
        self.bench_dir = Path(bench_dir)
        self.infer_dir = Path(infer_dir)

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(
        self,
        stage: Optional[str] = None
    ) -> None:
        """
        Sets up the datasets for the given stage.

        This method is called by Lightning automatically. It iterates through the
        subdirectories of the provided paths and creates a `LowLightDataset` for each.

        Args:
            stage (Optional[str], optional): The stage to set up ('fit', 'validate',
                'test', or 'predict'). Defaults to None.
        """
        self.train_datasets = self._set_dataset(path=self.train_dir)
        self.valid_datasets = self._set_dataset(path=self.valid_dir)
        self.bench_datasets = self._set_dataset(path=self.bench_dir)
        self.infer_datasets = self._set_dataset(path=self.infer_dir)

    def _set_dataset(
        self,
        path: Path
    ) -> List[LowLightDataset]:
        """
        Creates a list of datasets from subdirectories in a given path.

        Args:
            path (Path): The root directory containing dataset subfolders.

        Returns:
            List[LowLightDataset]: A list of initialized `LowLightDataset` objects.
        """
        datasets = []
        for folder in path.iterdir():
            if folder.is_dir():
                datasets.append(
                    LowLightDataset(
                        path=folder,
                        image_size=self.image_size
                    )
                )
        return datasets

    @overload
    def _set_dataloader(
        self,
        datasets: List[LowLightDataset],
        concat: Literal[True],
        shuffle: bool = False
    ) -> DataLoader:
        ...

    @overload
    def _set_dataloader(
        self,
        datasets: List[LowLightDataset],
        concat: Literal[False] = False,
        shuffle: bool = False
    ) -> List[DataLoader]:
        ...

    def _set_dataloader(
        self,
        datasets: List[LowLightDataset],
        concat: bool = False,
        shuffle: bool = False
    ) -> Union[DataLoader, List[DataLoader]]:
        """
        Creates a DataLoader or a list of DataLoaders from a list of datasets.

        Args:
            datasets (List[LowLightDataset]): The list of datasets to wrap.
            concat (bool, optional): Whether to concatenate the datasets into a single
                DataLoader. Defaults to False.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

        Returns:
            Union[DataLoader, List[DataLoader]]: A single DataLoader if `concat` is True,
                otherwise a list of DataLoaders.
        """
        if concat:
            dataloader = DataLoader(
                dataset=ConcatDataset(datasets=datasets),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True
            )
            return dataloader
        else:
            dataloaders = []
            for dataset in datasets:
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    persistent_workers=True,
                    pin_memory=True
                )
                dataloaders.append(loader)
            return dataloaders

    def train_dataloader(self) -> DataLoader:
        """
        Creates the training DataLoader.

        Returns:
            DataLoader: The DataLoader for the training set, concatenated and shuffled.
        """
        return self._set_dataloader(datasets=self.train_datasets, concat=True, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Creates the validation DataLoader.

        Returns:
            DataLoader: The DataLoader for the validation set, concatenated.
        """
        return self._set_dataloader(datasets=self.valid_datasets, concat=True)

    def test_dataloader(self) -> List[DataLoader]:
        """
        Creates the test (benchmark) DataLoaders.

        Returns:
            List[DataLoader]: A list of DataLoaders for the benchmark sets.
        """
        return self._set_dataloader(datasets=self.bench_datasets)

    def predict_dataloader(self) -> List[DataLoader]:
        """
        Creates the prediction DataLoaders.

        Returns:
            List[DataLoader]: A list of DataLoaders for the inference sets.
        """
        return self._set_dataloader(datasets=self.infer_datasets)

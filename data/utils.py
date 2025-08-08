from pathlib import Path
from typing import Tuple, Union, cast

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    """
    A custom PyTorch Dataset for loading pairs of low-light and corresponding
    high-quality images.

    The dataset expects a directory structure where the `path` contains two subdirectories:
    'low' and 'high'. The filenames in 'low' and 'high' are expected to match.
    """

    def __init__(self, path: Union[str, Path], image_size: int) -> None:
        """
        Initializes the LowLightDataset.

        Args:
            path (Union[str, Path]): The root directory of the dataset, which should
                contain 'low' and 'high' subdirectories.
            image_size (int): The target size (height and width) to which images
                will be resized.
        """
        super().__init__()
        self.path = Path(path)
        self.image_size = image_size
        self.transform = transforms.Compose(
            transforms=[
                transforms.Resize(size=self.image_size),
                transforms.ToTensor(),
            ]
        )

        self.low_path = self.path / "low"
        self.high_path = self.path / "high"

        self.low_datas = sorted(list(self.low_path.rglob(pattern="*.*")))
        self.high_datas = sorted(list(self.high_path.rglob(pattern="*.*")))

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of low-light images.
        """
        return len(self.low_datas)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed
                low-light image tensor and the corresponding high-quality image tensor.
        """
        low_data = self.low_datas[index]
        high_data = self.high_path / low_data.name

        low_image = Image.open(fp=low_data).convert(mode="RGB")
        high_image = Image.open(fp=high_data).convert(mode="RGB")

        low_data_tensor = cast(torch.Tensor, self.transform(img=low_image))
        high_data_tensor = cast(torch.Tensor, self.transform(img=high_image))

        return low_data_tensor, high_data_tensor
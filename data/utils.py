from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class LowLightDataset(Dataset):
    def __init__(self, path, image_size):
        super().__init__()
        self.path = Path(path)
        self.image_size = image_size

        self.transform = self._build_transform()

    def _build_transform(self):
        base = [
            transforms.Resize(size=(self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]

        return transforms.Compose(transforms=base)

    def __call__(self, image):
        return self.transform(img=image)

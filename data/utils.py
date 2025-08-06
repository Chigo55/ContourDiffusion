from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class LowLightDataset(Dataset):
    def __init__(self, path, image_size):
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

        self.low_datas = sorted(list(self.low_path.rglob(pattern='*.*')))
        self.high_datas = sorted(list(self.high_path.rglob(pattern='*.*')))

    def __len__(self):
        return len(self.low_datas)

    def __getitem__(self, index):
        low_data = self.low_datas[index]
        high_data = self.high_path / low_data.name

        low_image = Image.open(fp=low_data).convert(mode="RGB")
        high_image = Image.open(fp=high_data).convert(mode="RGB")

        low_data_tensor = self.transform(img=low_image)
        high_data_tensor = self.transform(img=high_image)

        return low_data_tensor, high_data_tensor
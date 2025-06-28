# %%
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os
import torch


# %%
from torch.utils.data import DataLoader


# %%
from ..data.dataloader import CustomDataset
from ..data.utils import DataTransform
from ..model.blocks.unet import UNet


# %%
transform=DataTransform(image_size=640)

dataset = CustomDataset(
    path="/data/projects/jih/study/data/1_train/1_LOLdataset",
    transform=transform
)


# %%
dataloader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    num_workers=int(os.cpu_count() * 0.9),
    persistent_workers=True,
    pin_memory=True
)


# %%
data = next(iter(dataloader))


# %%
def show_batch(images, ncols=4):
    nimgs = images.shape[0]
    nrows = (nimgs + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i in range(nimgs):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(F.to_pil_image(images[i]))
        plt.axis('off')
        plt.title(f"Image {i}")
    plt.tight_layout()
    plt.show()

show_batch(data)


# %%
rn = UNet(in_channels=3, out_channels=3, hidden_channels=64, num_levels=3, temb_dim=64, dropout=0.1, shortcut=True, trainable=False)
t = torch.randint(low=0, high=10000, size=(16, 1))
rn_d = rn(data)

show_batch(rn_d)
# %%

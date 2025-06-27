# %%
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os


# %%
from torch.utils.data import DataLoader


# %%
from data.dataloader import CustomDataset
from data.utils import DataTransform
from model.blocks.contourlet import LaplacianPyramid, DirectionalFilterBank, ContourletTransform


# %%
transform=DataTransform(image_size=640)

dataset = CustomDataset(
    path="data/1_train/1_LOLdataset",
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
lp  = LaplacianPyramid(
    num_levels=4,
    filter_size=5,
    sigma=1.0,
    channels=3
)

py, lc = lp(data)


# %%
for i, (p, l) in enumerate(zip(py, lc)):
    show_batch(images=p, ncols=4)
    show_batch(images=l, ncols=4)


# %%
dfb = DirectionalFilterBank(
    num_levels=4,
    filter_size=5,
    sigma=1.0,
    omega_x=0.25,
    omega_y=0.25,
    channels=3
)

for l in lc:
    dfb_data = dfb(l)
    for i, d in enumerate(dfb_data):
        show_batch(images=d, ncols=8)


# %%
contourlet = ContourletTransform(
    num_levels=3,
    filter_size=5,
    sigma=1.0,
    omega_x=0.25,
    omega_y=0.25,
    channels=3
)

pyramid, subbands = contourlet(data)


# %%
for i, subband in enumerate(subbands):
    for i, s in enumerate(subband):
        show_batch(images=s, ncols=8)




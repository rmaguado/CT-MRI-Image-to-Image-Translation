import json
#from libtiff import TIFF
from glob import glob
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from models.vit_mae import MaskedAutoencoderViT
from trainer import Trainer

"""
def load_images():
    img_dirs = glob("./tempdata/tiff_images/*.tif")
    images = [
        TIFF.open(img_dir, mode='r').read_image() for img_dir in img_dirs
    ]
    images_standard = [
        (img / max( np.max(img), abs(np.min(img)) ) + 1 ) / 2 for img in images
    ]
    images = np.array(images_standard, dtype=np.float32)
    
    images = np.expand_dims(images, axis=1)
    images = torch.tensor(images)
    return images

images = load_images()
"""

with open("config.json") as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

class randData(Dataset):
    def __init__(self):
        self.images = [torch.rand(1, 512, 512) for _ in range(100)]
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]

testloader = DataLoader(
    randData(),
    **config["dataloader"]
)

model = MaskedAutoencoderViT(**config["model"])
optim = AdamW(model.parameters(), lr=lr)

trainer = Trainer(
    model, 
    optim, 
    config, 
    **config["trainer"]
)

trainer.train(testloader, testloader)

import json
from libtiff import TIFF
from glob import glob
import numpy as np
import torch
from torch.optim import AdamW
from transformers import ViTMAEConfig
from tqdm import tqdm

import matplotlib.pyplot as plt

from models.vit_mae import MaskedAutoencoderViT

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

with open("config.json") as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

model = MaskedAutoencoderViT(model_config).float()
optim = AdamW(model.parameters(), lr=lr)

device = torch.device("cpu")
model.to(device)
images.to(device)

total_epochs = 100
model.train()
loop = tqdm(range(total_epochs), leave=True, ascii=" >=")

for epoch_number in loop:
    model.zero_grad()
    outputs = model.forward(images)
    loss = outputs.loss
    loss.backward()
    loop.set_postfix({
        "loss":f"{loss.item():.2f}"
    })
    optim.step()


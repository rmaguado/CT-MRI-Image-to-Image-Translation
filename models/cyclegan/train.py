"""
Train a CycleGAN model.
"""

import json

from models.cyclegan.cyclegan_model import CycleGANModel
from models.cyclegan.cyclegan_trainer import CycleGANTrainer, BlankOptim
from models.cyclegan.cyclegan_dataloader import CycleGANDataloader

DATA_ROOT_PATH = "/nfs/home/clruben/workspace/nst/data/"
CONFIG_PATH = "/nfs/home/clruben/workspace/nst/models/cyclegan/config.json"

with open(CONFIG_PATH, encoding="utf-8") as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

train_loader = CycleGANDataloader(
    DATA_ROOT_PATH,
    'train',
    mini_batch_sample_size=2
)

model = CycleGANModel(**config["model"])
optim = BlankOptim()

trainer = CycleGANTrainer(
    model,
    optim,
    config,
    **config["trainer"]
)

trainer.train(train_loader)

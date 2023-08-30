import json

import torch
import lightning.pytorch as pl

from models.cyclegan_light.cyclegan import CycleGAN
from dataloaders import CycleganDataloader

DATA_PATH = "/nfs/home/clruben/workspace/nst/data/batch1"
CONFIG_PATH = "/nfs/home/clruben/workspace/nst/models/cyclegan_light/config.json"

torch.set_float32_matmul_precision('medium')

with open(CONFIG_PATH, encoding="utf-8") as file:
    config = json.load(file)

train_loader = CycleganDataloader(
    DATA_PATH,
    'train',
    batch_size=config["batch_size"]
)

model = CycleGAN(**config["model"])

logger = pl.loggers.TensorBoardLogger(
    **config["logger"]
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    **config["checkpoint"]
)

trainer = pl.Trainer(
    enable_progress_bar=False,
    logger=logger,
    callbacks=[checkpoint_callback],
    **config["trainer"]
)
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    #ckpt_path="/nfs/home/clruben/workspace/nst/models/mae_gan/checkpoints/last-v3.ckpt"
)

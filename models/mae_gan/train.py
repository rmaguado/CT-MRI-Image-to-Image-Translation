import json

import torch
import lightning.pytorch as pl

from dataloaders import Dataloader
from models.mae_gan.mae_gan import MaeGanLM

DATA_PATH = "/nfs/home/clruben/workspace/nst/data/batch1"
CONFIG_PATH = "/nfs/home/clruben/workspace/nst/models/mae_gan/config.json"

torch.set_float32_matmul_precision('medium')

with open(CONFIG_PATH, encoding="utf-8") as file:
    config = json.load(file)

train_loader = Dataloader(
    DATA_PATH,
    'train',
    batch_size=config["batch_size"]
)

model = MaeGanLM(config)

logger = pl.loggers.TensorBoardLogger(
    **config["logger"]
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    **config["checkpoint"]
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[checkpoint_callback],
    **config["trainer"]
)
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    #ckpt_path="/nfs/home/clruben/workspace/nst/models/mae_gan/checkpoints/last-v3.ckpt"
)

import json

import torch
import lightning.pytorch as pl

from dataloaders import Dataloader
from models.maegan.network import MaeGanLM

DATA_PATH = "/nfs/home/clruben/workspace/nst/data/"
CONFIG_PATH = "/nfs/home/clruben/workspace/nst/models/maegan/config.json"

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
    enable_progress_bar=False,
    logger=logger,
    callbacks=[checkpoint_callback],
    **config["trainer"]
)
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    ckpt_path="/nfs/home/clruben/workspace/nst/models/maegan/checkpoints/last.ckpt"
)

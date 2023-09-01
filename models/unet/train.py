import json

import torch
import lightning.pytorch as pl

from dataloaders import UNetDataloader
from models.unet.network import UNetLM

torch.set_float32_matmul_precision('medium')

DATA_PATH = "/home/radiomica/ruben/CT-MRI-Image-to-Image-Translation/data/reg"
CONFIG_PATH = "/home/radiomica/ruben/CT-MRI-Image-to-Image-Translation/models/unet/config.json"

with open(CONFIG_PATH, encoding="utf-8") as file:
    config = json.load(file)

train_loader = UNetDataloader(
    data_root_path,
    'train',
    enable_data_augmentation=True
)

model = UNetLM(config)

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
    train_dataloaders=train_loader
    #ckpt_path="/nfs/home/clruben/workspace/nst/models/mae_gan/checkpoints/model_checkpoint.ckpt"
)

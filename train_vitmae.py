import json
import numpy as np
from torch.optim import AdamW
from dataloader import Dataloader
from tqdm import tqdm

from models.VITMAE.vit_mae import MaskedAutoencoderViT
from trainer import Trainer

with open("config.json") as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

rootpath = "/nfs/home/clruben/workspace/nst/data/preprocessed"

class MaskingAE_Dataloader:
    def __init__(self, source, dataset):
        """Pools together MRI and CT scans for masked modeling.
        """
        self.loaders = [
            Dataloader(source, dataset, "CT"),
            Dataloader(source, dataset, "MRI")
        ]
        self.equal_num_examples = min(
            len(self.ct_loader),
            len(self.mri_loader)
        ) * 2
        self.mode = True
        self.counter = 0
        self.masking_ratio = 0.75
    def __len__(self):
        return self.equal_num_examples
    def __iter__(self):
        return self
    def __next__(self):
        self.mode = not self.mode
        if self.counter == self.equal_num_examples:
            self.counter = 0
        next_item = next(self.loaders[self.mode])
        mode = ["CT", "MRI"][self.mode]
        return next_item, mode, self.masking_ratio

train_loader = MaskingAE_Dataloader(
    rootpath,
    'train'
)

test_loader = MaskingAE_Dataloader(
    rootpath,
    'test'
)

model = MaskedAutoencoderViT(**config["model"])
optim = AdamW(model.parameters(), lr=lr)

trainer = Trainer(
    model, 
    optim, 
    config, 
    **config["trainer"]
)

trainer.train(train_loader, test_loader)
#model.mode = "nst"
#trainer.train(testloader, testloader)

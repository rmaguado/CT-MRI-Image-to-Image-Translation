import json
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from models.vit_mae import MaskedAutoencoderViT
from trainer import Trainer

with open("config.json") as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

class testData(Dataset):
    def __init__(self):
        self.images = np.load(
            "/nfs/home/clruben/workspace/nst/tempdata/preprocessed/test_batch.npz"
        )["arr_0"]

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx]

testloader = DataLoader(
    testData(),
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
#model.mode = "nst"
#trainer.train(testloader, testloader)
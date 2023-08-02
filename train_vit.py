import json
from torch.optim import AdamW
from dataloader import Dataloader

from models.ViT_MAE.vit_translation import ViT_Translation
from trainer import Trainer

rootpath = "/nfs/home/clruben/workspace/nst/data/"
config_path = "/nfs/home/clruben/workspace/nst/models/ViT_MAE/config.json"

with open(config_path) as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

class MAE_Dataloader:
    def __init__(self, source, dataset):
        """Pools together MRI and CT scans for masked modeling.
        """
        self.loaders = [
            Dataloader(source, dataset, "CT"),
            Dataloader(source, dataset, "MR")
        ]
        self.equal_num_examples = min(
            [len(x) for x in self.loaders]
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
        mode = ["CT", "MR"][self.mode]
        return next_item, mode

train_loader = MAE_Dataloader(
    rootpath,
    'test' ############ remember to change to train
)

test_loader = MAE_Dataloader(
    rootpath,
    'test'
)

model = ViT_Translation(**config["model"])
optim = AdamW(model.parameters(), lr=lr)

trainer = Trainer(
    model, 
    optim, 
    config, 
    **config["trainer"]
)

trainer.train(train_loader, test_loader)
#model.mode = "translation"
#trainer.train(testloader, testloader)

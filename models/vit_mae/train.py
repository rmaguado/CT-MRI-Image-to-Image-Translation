import json
from torch.optim import AdamW

from dataloaders import Dataloader
from models.vit_mae.vit_translation import ViT_Translation
from trainer import Trainer

data_root_path = "/nfs/home/clruben/workspace/nst/data/"
config_path = "/nfs/home/clruben/workspace/nst/models/vit_mae/config.json"

with open(config_path) as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

train_loader = Dataloader(
    data_root_path,
    'train'
)
test_loader = Dataloader(
    data_root_path,
    'test',
    size_limit=500
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

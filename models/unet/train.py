import json
from torch.optim import AdamW

from models.unet.unet_dataloader import UNetDataloader
from models.unet.unet_model import UNet
from trainer import Trainer

data_root_path = "/nfs/home/clruben/workspace/nst/data/reg"
config_path = "/nfs/home/clruben/workspace/nst/models/unet/config.json"

with open(config_path) as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

train_loader = UNetDataloader(
    data_root_path,
    'train',
    enable_data_augmentation=True
)
test_loader = UNetDataloader(
    data_root_path,
    'test',
    enable_data_augmentation=False
)

model = UNet(**config["model"])
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

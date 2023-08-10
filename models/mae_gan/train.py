import json
from torch.optim import AdamW

from dataloaders import Dataloader
from models.mae_gan.mae_gan import MaeGan
from models.mae_gan.mae_gan_trainer import MaeGanTrainer

data_root_path = "/nfs/home/clruben/workspace/nst/data/"
config_path = "/nfs/home/clruben/workspace/nst/models/mae_gan/config.json"

with open(config_path) as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

train_loader = Dataloader(
    data_root_path,
    'train'
)

model = MaeGan(**config["model"])
optim = AdamW(model.parameters(), lr=lr)

trainer = MaeGanTrainer(
    model, 
    optim, 
    config,
    **config["trainer"]
)

trainer.train(train_loader)
#model.set_mode("translation")
#trainer.restart_warmup()
#trainer.clear_best_models()
#trainer.train(train_loader)

import json
from torch.optim import AdamW

from dataloaders import Dataloader
from models.mae_gan.mae_gan import MaeGan
from models.mae_gan.mae_gan_trainer import MaeGanTrainer

from torch.optim.lr_scheduler import CosineAnnealingLR

data_root_path = "/nfs/home/clruben/workspace/nst/data/"
config_path = "/nfs/home/clruben/workspace/nst/models/mae_gan/config.json"

with open(config_path) as file:
    config = json.load(file)
lr = config["trainer"]["learning_rate"]

train_loader = Dataloader(
    data_root_path,
    'train',
    size_limit=50
)

model = MaeGan(**config["model"])
optim = AdamW(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optim, **config["scheduler"])

trainer = MaeGanTrainer(
    model, 
    optim, 
    config,
    scheduler=scheduler,
    **config["trainer"]
)

trainer.train(train_loader)
trainer.model.set_mode("translation")
trainer.reset_warmup()
trainer.clear_save_models()

trainer.scheduler = CosineAnnealingLR(optim, **config["scheduler"])

trainer.train(train_loader)

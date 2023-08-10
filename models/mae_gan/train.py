import json
from torch.optim import AdamW

from dataloaders import Dataloader
from models.mae_gan.mae_gan import MaeGan
from trainer import Trainer

class MaeGanTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def train_tensorboard(self, outputs):
        loss = outputs.reconstruction_loss
        self.writer.add_scalar(
            "Loss/train",
            loss.item(),
            self.batch_counter
        )
        self.writer.add_scalar(
            "Learning Rate",
            self.optim.param_groups[0]['lr'],
            self.batch_counter
        )

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
#model.mode = "translation"
#trainer.train(testloader, testloader)

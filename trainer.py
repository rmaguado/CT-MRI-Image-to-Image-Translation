from datetime import datetime
from tqdm import tqdm
import logging
import shutil
import json
import os

from torch.utils.tensorboard import SummaryWriter
import torch

class Trainer:
    def __init__(
            self,
            model,
            optim,
            config : dict,
            model_name : str,
            model_kwargs : list = ["x"],
            scheduler = None,
            train_epochs : int = 5,
            device : str = "cpu",
            log_dir : str = "",
            loading_model : bool = True,
            load_model_dir : str = "",
            enable_tensorboard : bool = False,
            tensorboard_logdir : str = "",
            tensorboard_log_frequency : int = 1,
            enable_warmup : bool = False,
            warmup_steps : int = 1000,
            learning_rate : float = 0.0005,
            save_dir : str = "",
            warmup_factor : int = 10,
            enable_delete_worse_models : bool = False,
            max_models_saved : int = 3
        ):
        self.start_timestamp = datetime.now().strftime("%m-%d-%Y-%H_%M_%S")
        logging.basicConfig(
            filename=os.path.join(log_dir, f'{self.start_timestamp}.txt'),
            level=logging.INFO
        )

        self.model = model
        self.optim = optim
        self.config = config
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.scheduler = scheduler
        self.train_epochs = train_epochs
        if device != "cpu" and not torch.cuda.is_available():
            logging.error("Device %s not available.", device)
            raise ValueError(f"Device {device} not available.")
        self.device = torch.device(device)
        self.loading_model = loading_model
        self.load_model_dir = load_model_dir
        self.enable_tensorboard = enable_tensorboard
        self.tensorboard_logdir = tensorboard_logdir
        self.tensorboard_log_frequency = tensorboard_log_frequency
        self.enable_warmup = enable_warmup
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.warmup_factor = warmup_factor
        self.enable_delete_worse_models = enable_delete_worse_models
        self.max_models_saved = max_models_saved

        self.model.to(self.device)
        logging.info("Running on %s", self.device)
        
        self.batch_counter = 0
        if self.enable_tensorboard:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.tensorboard_logdir,self.start_timestamp)
            )
        
        if self.enable_delete_worse_models:
            self.best_model_logs = []

    def load_model(self, load_model_dir : str):
        checkpoint = torch.load(
            os.path.join(load_model_dir, "parameters.torch")
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        with open(os.path.join(load_model_dir, "logs.json"), "r", encoding="utf-8") as file:
            logs = json.load(file)
        self.batch_counter = logs["batch_counter"]
        logging.info("Loaded model from checkpoint.")

    def save_model(self, epoch_number : int, eval_loss : float):
        save_timestamp = datetime.now().strftime('%m-%d-%Y-%H_%M_%S')
        save_dirname = self.model_name + "-" + save_timestamp

        path = os.path.join(self.save_dir, save_dirname)
        os.mkdir(path)

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else dict()
            }, 
            os.path.join(path, "parameters.torch")
        )

        logs = {
            'epoch_number': epoch_number,
            'tensorboard_counter' : self.batch_counter,
            'eval_loss': eval_loss,
            'lr': self.optim.param_groups[0]['lr'],
            'start_time' : self.start_timestamp
        }

        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f)
        with open(os.path.join(path, "logs.json"), "w", encoding="utf-8") as f:
            json.dump(logs, f)

        logging.info("Saved model checkpoint.")
        return path

    def delete_worse_models(self):
        losses = [x["eval_loss"] for x in self.best_model_logs]
        worst_idx = losses.index(max(losses))
        worst_save_dir = self.best_model_logs[worst_idx]["save_dir"]
        shutil.rmtree(worst_save_dir)
        del self.best_model_logs[worst_idx]
        
    def step_lr_warmup(self):
        lr_factor = self.learning_rate / self.warmup_factor
        new_lr = lr_factor + (self.learning_rate - lr_factor) * self.batch_counter / self.warmup_steps
        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr

    def eval_tensorboard(self, outputs):
        loss = outputs.loss
        self.writer.add_scalar("Loss/eval", loss.item(), self.batch_counter)
        
    def eval_loop(self, eval_dataloader, epoch_number):
        self.model.eval()
        loop = tqdm(
            range(len(eval_dataloader)), 
            leave=True, 
            ascii=" >="
        )
        loop.set_description(f'Test Epoch {epoch_number}')
        total_eval_loss = 0
        for _ in loop:
            batch_data = next(eval_dataloader)
            if len(self.model_kwargs) == 1:
                outputs = self.model(
                    batch_data.to(self.device)
                )
            else:
                outputs = self.model(
                    **{kw:batch_data[i].to(self.device) for i,kw in enumerate(self.model_kwargs)}
                )
            loss = outputs.loss
            loop.set_postfix(
                {"loss":loss.item()}
            )
            total_eval_loss += loss.item()
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        if self.enable_tensorboard:
            self.eval_tensorboard(outputs)
        return avg_eval_loss
    
    def train_tensorboard(self, outputs):
        loss = outputs.loss
        self.writer.add_scalar("Loss/train", loss.item(), self.batch_counter)
        self.writer.add_scalar("Learning Rate", self.optim.param_groups[0]['lr'], self.batch_counter)
    
    def train_loop(self, train_dataloader, epoch_number):
        self.model.train()
        loop = tqdm(
            range(len(train_dataloader)), 
            leave=True, 
            ascii=" >="
        )
        loop.set_description(f'Train Epoch {epoch_number}')
        for _ in loop:
            batch_data = next(train_dataloader)
            self.model.zero_grad()
            if len(self.model_kwargs) == 1:
                outputs = self.model(
                    batch_data.to(self.device)
                )
            else:
                outputs = self.model(
                    **{kw:batch_data[i].to(self.device) for i,kw in enumerate(self.model_kwargs)}
                )
            loss = outputs.loss
            loss.backward()
            loop.set_postfix({
                "loss":f"{loss.item():.2f}"
            })
            self.optim.step()

            if self.enable_warmup and self.batch_counter < self.warmup_steps:
                self.step_lr_warmup()
            elif self.scheduler is not None:
                self.scheduler.step()

            if self.enable_tensorboard and self.batch_counter % self.tensorboard_log_frequency == 0:
                self.train_tensorboard(outputs)

            self.batch_counter += 1

    def train(self, train_dataloader, eval_dataloader):
        for epoch_number in range(1,self.train_epochs+1):
            logging.info("Starting epoch %s", epoch_number)
            self.train_loop(train_dataloader, epoch_number)
            logging.info("Finished training.")
            eval_loss = self.eval_loop(eval_dataloader, epoch_number)
            logging.info("Finished evaluation. Average loss: %s:.6f", eval_loss)
            if self.enable_delete_worse_models:
                if len(self.best_model_logs) < self.max_models_saved or \
                eval_loss < max([save["eval_loss"] for save in self.best_model_logs]):

                    save_dir = self.save_model(epoch_number, eval_loss)

                    self.best_model_logs.append({
                        "eval_loss": eval_loss,
                        "save_dir" : save_dir
                    })
                if len(self.best_model_logs) > self.max_models_saved:
                    self.delete_worse_models()
        logging.info("Finished training. Exiting.")

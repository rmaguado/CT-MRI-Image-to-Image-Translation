from datetime import datetime
from tqdm import tqdm
import logging
import shutil
import json
import numpy as np
import os
from typing import Optional

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Trains a model.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            optim: torch.optim.Optimizer,
            config: dict,
            model_name: str,
            model_kwargs: Optional[list],
            scheduler: Optional[lr_scheduler._LRScheduler] = None,
            train_epochs: Optional[int] = 5,
            device: Optional[str] = "cpu",
            log_dir: Optional[str] = "",
            loading_model: Optional[bool] = True,
            load_model_dir: Optional[str] = "",
            enable_tensorboard: Optional[bool] = False,
            tensorboard_logdir: Optional[str] = "",
            tensorboard_log_frequency: Optional[int] = 1,
            enable_tqdm: Optional[bool] = True,
            enable_warmup: Optional[bool] = False,
            warmup_steps: Optional[int] = 1000,
            warmup_factor: Optional[float] = 10.0,
            learning_rate: Optional[float] = 5e-4,
            save_dir: Optional[str] = "",
            enable_batch_checkpointing: Optional[bool] = True,
            save_frequency: Optional[int] = 5000,
            enable_delete_worse_models: Optional[bool] = False,
            max_models_saved: Optional[int] = 3
    ):
        self.start_timestamp: str = self.get_timestamp()
        logging.basicConfig(
            filename=os.path.join(log_dir, f'{self.start_timestamp}.txt'),
            level=logging.INFO
        )

        self.model: torch.nn.Module = model
        self.optim: torch.optim.Optimizer = optim
        self.config: dict = config
        self.model_name: str = model_name
        self.model_kwargs: list = model_kwargs
        self.scheduler: Optional[lr_scheduler._LRScheduler] = scheduler
        self.train_epochs: int = train_epochs
        if device != "cpu" and not torch.cuda.is_available():
            logging.error("Device %s not available.", device)
            raise ValueError(f"Device {device} not available.")
        self.device: torch.device = torch.device(device)
        self.loading_model: bool = loading_model
        self.load_model_dir: str = load_model_dir
        self.enable_tensorboard: bool = enable_tensorboard
        self.tensorboard_logdir: str = tensorboard_logdir
        self.tensorboard_log_frequency: int = tensorboard_log_frequency
        self.enable_tqdm: bool = enable_tqdm
        self.enable_warmup: bool = enable_warmup
        self.warmup_steps: int = warmup_steps
        self.learning_rate: float = learning_rate
        self.save_dir: str = save_dir
        self.enable_batch_checkpointing: bool = enable_batch_checkpointing
        self.save_frequency: int = save_frequency
        self.warmup_factor: float = warmup_factor
        self.enable_delete_worse_models: bool = enable_delete_worse_models
        self.max_models_saved: int = max_models_saved
        
        self.warmup_batch_counter: int = 0

        self.model.to(self.device)
        logging.info("%s Running on %s", self.get_timestamp(), self.device)

        self.batch_counter: int = 0
        if self.enable_tensorboard:
            self.writer: SummaryWriter = SummaryWriter(
                log_dir=os.path.join(
                    self.tensorboard_logdir,
                    self.model_name + self.start_timestamp
                )
            )

        if self.loading_model:
            self.load_model(self.load_model_dir)
        
        for param_group in self.optim.param_groups:
            param_group['lr'] = self.learning_rate

        if self.enable_delete_worse_models:
            self.best_model_logs: list = []

    def to_device(self, data):
        if torch.is_tensor(data):
            return data.to(self.device)
        if isinstance(data, np.ndarray):
            return torch.tensor(data.copy()).to(self.device)
        return data

    def get_timestamp(self):
        return datetime.now().strftime('%m-%d-%Y-%H_%M_%S')

    def get_iterator(self, dataloader, epoch, mode):
        if self.enable_tqdm:
            loop = tqdm(
                range(len(dataloader)),
                leave=True, 
                ascii=" >=",
                bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}',
                dynamic_ncols=True
            )
            loop.set_description(f'{mode} Epoch {epoch}')
            return loop
        return range(len(dataloader))

    def load_model(self, load_model_dir: str):
        logging.info(
            "%s Loading model from %s",
            self.get_timestamp(),
            load_model_dir
        )
        checkpoint: dict = torch.load(
            os.path.join(load_model_dir, "parameters.torch")
        )
        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )
        self.optim.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
        if self.scheduler is not None:
            if checkpoint["scheduler_state_dict"] is not dict():
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict']
                )
            else:
                logging.warning("No scheduler state dict found in checkpoint.")

        with open(
            os.path.join(load_model_dir, "logs.json"),
            "r", encoding="utf-8"
        ) as file:
            logs: dict = json.load(file)
        self.batch_counter: int = logs["batch_counter"]
        logging.info("%s Loaded model from checkpoint.", self.get_timestamp())

    def save_model(self, epoch_number: int, loss: float):
        save_timestamp: str = self.get_timestamp()
        save_dirname: str = self.model_name + "-" + save_timestamp

        path: str = os.path.join(self.save_dir, save_dirname)
        os.mkdir(path)

        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
                if self.scheduler is not None else dict()
            },
            os.path.join(path, "parameters.torch")
        )

        logs: dict = {
            'epoch_number': epoch_number,
            'batch_counter': self.batch_counter,
            'loss': loss,
            'lr': self.optim.param_groups[0]['lr'],
            'start_time': self.start_timestamp
        }

        with open(
            os.path.join(path, "config.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(self.config, f, indent=4)
        with open(
            os.path.join(path, "logs.json"),
            "w", encoding="utf-8"
        ) as f:
            json.dump(logs, f, indent=4)

        logging.info("%s Saved model checkpoint.", self.get_timestamp())
        return path
    
    def create_checkpoint(
            self,
            epoch_number: int,
            outputs = None,
            loss: float = None
        ):
        if outputs is not None:
            loss = outputs.loss.item()
        if self.enable_delete_worse_models:
            if len(self.best_model_logs) < self.max_models_saved or \
                    loss < max(save["loss"] for save in self.best_model_logs):

                save_dir: str = self.save_model(epoch_number, loss)

                self.best_model_logs.append({
                    "loss": loss,
                    "save_dir": save_dir
                })
            if len(self.best_model_logs) > self.max_models_saved:
                self.delete_worse_models()
            return
        self.save_model(epoch_number, loss)

    def delete_worse_models(self):
        saved_model_losses = [x["loss"] for x in self.best_model_logs]
        worst_model_idx: int = saved_model_losses.index(
            max(saved_model_losses)
        )
        worst_save_dir: str = self.best_model_logs[worst_model_idx]["save_dir"]
        shutil.rmtree(worst_save_dir)
        del self.best_model_logs[worst_model_idx]

    def step_lr_warmup(self):
        lr_factor: float = self.learning_rate / self.warmup_factor
        new_lr: float = lr_factor + (
            self.learning_rate - lr_factor
        ) * self.batch_counter / self.warmup_steps

        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr

    def eval_tensorboard(self, outputs):
        loss = outputs.loss
        self.writer.add_scalar("Loss/eval", loss.item(), self.batch_counter)

    def eval_loop(self, eval_dataloader, epoch_number):
        self.model.eval()
        loop = self.get_iterator(eval_dataloader, epoch_number, "Eval ")
        eval_iter = iter(eval_dataloader)
        total_eval_loss: float = 0.0
        for _ in loop:
            with torch.no_grad():
                batch_data = next(eval_iter)
                if len(self.model_kwargs) == 1:
                    outputs = self.model(
                        self.to_device(batch_data)
                    )
                else:
                    outputs = self.model(
                        **{kw: self.to_device(batch_data[i])
                            for i, kw in enumerate(self.model_kwargs)}
                    )
                loss = outputs.loss
                if self.enable_tqdm:
                    loop.set_postfix(
                        {"loss": loss.item()}
                    )
                total_eval_loss += loss.item()
        avg_eval_loss: float = total_eval_loss / len(eval_dataloader)
        if self.enable_tensorboard:
            self.eval_tensorboard(outputs)
        return avg_eval_loss

    def train_tensorboard(self, outputs):
        loss = outputs.loss
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

    def train_loop(self, train_dataloader, epoch_number):
        self.model.train()
        loop = self.get_iterator(train_dataloader, epoch_number, "Train")
        train_iter = iter(train_dataloader)
        for _ in loop:
            batch_data = next(train_iter)
            self.model.zero_grad()
            if len(self.model_kwargs) == 1:
                outputs = self.model(
                    self.to_device(batch_data)
                )
            else:
                outputs = self.model(
                    **{kw: self.to_device(batch_data[i])
                        for i, kw in enumerate(self.model_kwargs)}
                )
            loss = outputs.loss
            loss.backward()
            if self.enable_tqdm:
                loop.set_postfix({
                    "loss":f"{loss.item():.2f}"
                })
            self.optim.step()

            if self.enable_batch_checkpointing and \
                    self.batch_counter % self.save_frequency == 0 and \
                    self.batch_counter != 0:
                self.create_checkpoint(epoch_number, outputs=outputs)

            if self.enable_warmup and \
                    self.warmup_batch_counter < self.warmup_steps:
                self.step_lr_warmup()
                self.warmup_batch_counter += 1
            elif self.scheduler is not None:
                self.scheduler.step()

            if self.enable_tensorboard and \
                    self.batch_counter % self.tensorboard_log_frequency == 0:
                self.train_tensorboard(outputs)

            self.batch_counter += 1

    def train(self, train_dataloader, eval_dataloader=None):
        for epoch_number in range(1, self.train_epochs+1):
            logging.info(
                "%s Starting epoch %s", self.get_timestamp(), epoch_number
            )
            self.train_loop(train_dataloader, epoch_number)
            logging.info("%s Finished training.", self.get_timestamp())
            if eval_dataloader is not None:
                logging.info(
                    "%s Starting evaluation.", self.get_timestamp()
                )
                eval_loss = self.eval_loop(eval_dataloader, epoch_number)
                loss_str = f"{eval_loss:.6f}"
                logging.info(
                    "%s Finished evaluation. Average loss: %s",
                    self.get_timestamp(), loss_str
                )
                if not self.enable_batch_checkpointing:
                    self.create_checkpoint(epoch_number, loss=eval_loss)
        logging.info("%s Finished training. Exiting.", self.get_timestamp())

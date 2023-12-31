"""
Modifies trainer to work with cyclegan.
"""

from trainer import Trainer


class BlankOptim:
    """Optimizer that does nothing.
    """
    def __init__(self):
        self.param_groups = [
            {
                "lr": 0
            }
        ]

    def state_dict(self):
        """Returns an empty state dict.
        """
        return {}

    def load_state_dict(self, state_dict):
        """Clears the state dict input in trainer.
        """
        return state_dict

    def step(self):
        """Clears optimizer step.
        """
        return


class CycleGANTrainer(Trainer):
    """Modified Trainer for a GAN model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_loop(self, train_dataloader, epoch_number):
        self.model.train()
        loop = self.get_iterator(train_dataloader, epoch_number, "Train")
        train_iter = iter(train_dataloader)
        for _ in loop:
            batch_data = next(train_iter)

            self.model.set_input(batch_data)
            self.model.optimize_parameters()
            losses = self.model.get_current_losses()

            summary_loss = losses["cycle_A"] + losses["cycle_A"]

            if self.enable_batch_checkpointing and \
                    self.batch_counter % self.save_frequency == 0 and \
                    self.batch_counter != 0:
                self.create_checkpoint(epoch_number, loss=summary_loss)

            if self.enable_warmup and self.batch_counter < self.warmup_steps:
                self.step_lr_warmup()

            if self.enable_tensorboard and \
                    self.batch_counter % self.tensorboard_log_frequency == 0:
                for key, loss in losses.items():
                    self.writer.add_scalar(
                        f"CycleGAN/{key}",
                        loss,
                        self.batch_counter
                    )
                self.writer.add_scalar(
                    "Loss/train",
                    summary_loss,
                    self.batch_counter
                )

            self.batch_counter += 1

    def eval_loop(self, *args, **kwargs):
        return 0.

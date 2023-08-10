from trainer import Trainer


class MaeGanTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def train_tensorboard(self, outputs):
        images = self.model.unpatchify(outputs.pred)[0][0]
        self.writer.add_images(
            "Images/Reconstruction",
            images,
            self.batch_counter,
            dataformats="WH"
        )
        
        self.writer.add_scalar(
            "Loss/Reconstruction",
            outputs.reconstruction_loss.item(),
            self.batch_counter
        )
        self.writer.add_scalar(
            "Loss/Adversarial",
            outputs.adversarial_loss.item(),
            self.batch_counter
        )
        self.writer.add_scalar(
            "Loss/Discriminator_Real",
            outputs.discriminator_real_loss.item(),
            self.batch_counter
        )
        self.writer.add_scalar(
            "Loss/Discriminator_Fake",
            outputs.discriminator_fake_loss.item(),
            self.batch_counter
        )
        self.writer.add_scalar(
            "Learning Rate",
            self.optim.param_groups[0]['lr'],
            self.batch_counter
        )
    def create_checkpoint(
            self,
            epoch_number: int,
            outputs = None,
            loss: float = None
        ):
        if outputs is not None:
            loss = outputs.reconstruction_loss.item()
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

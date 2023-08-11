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
        
        # if reconstruction_loss is not None:
        if outputs.reconstruction_loss is not None:
            
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
        
        else:
            
            self.writer.add_scalar(
                "Loss/Reconstruction",
                outputs.loss.item(),
                self.batch_counter
            )
        
        self.writer.add_scalar(
            "Learning Rate",
            self.optim.param_groups[0]['lr'],
            self.batch_counter
        )

import torch
import wandb
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from utils import NoisingScheduler, CosineWarmup
from discriminator import ValueModule
from dataloader import MembraneDataset, MembraneDataModule, get_datasets

config = OmegaConf.load("/work/sg666/MeMDLM/src/diffusion/config.yaml")
wandb.login(key='2b76a2fa2c1cdfddc5f443602c17b011fefb0a8f')

class SolubilityClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ValueModule(config)
        self.loss_fn = nn.BCELoss()

    def forward(self, batch):
        return self.model(batch['input_ids'], batch['attention_mask'])

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch['labels'])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        val_loss = self.loss_fn(logits, batch['labels'])
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.value.training.lr)
        lr_scheduler = CosineWarmup(
            optimizer,
            warmup_steps=self.config.lr_schduler.num_warmup_steps,
            total_steps=self.config.trainer.max_steps
        )
        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": 'step',
            'frequency': 1,
            'monitor': 'val/loss',
            'name': 'trainer/lr'
        }
        return [optimizer], [scheduler_dict]

def main():
    # data
    datasets = get_datasets(config)
    data_module = MembraneDataModule(
        train_dataset=datasets['train'],
        val_dataset=datasets['val'],
        test_dataset=datasets['test'],
        batch_size=config.training.batch_size
    )

    # wandb logging
    wandb.init()
    wandb_logger = WandbLogger(**config.wandb)

    # lightning checkpoints
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=config.value.training.ckpt_path,
        filename="best_model"
    )

    # lightning trainer
    trainer = pl.Trainer(
        max_epochs=config.value.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # train or evalute the model
    model = SolubilityClassifier(config)
    if config.value.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif config.value.mode == "test":
        trainer.test(datamodule=data_module)
    else:
        raise ValueError(f"{config.value.mode} is invalid. Must be 'train' or 'test'")


if __name__ == "__main__":
    main()
import torch
import wandb
import gc
import torch.nn as nn
import lightning.pytorch as pl

from omegaconf import OmegaConf
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import NoisingScheduler, CosineWarmup, _print
from discriminator import ValueModule
from dataloader import MembraneDataset, MembraneDataModule, get_datasets

config = OmegaConf.load("/workspace/sg666/MeMDLM/configs/config.yaml")
#wandb.login(key='2b76a2fa2c1cdfddc5f443602c17b011fefb0a8f')

class SolubilityClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = ValueModule(config)
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, batch):
        return self.model(batch['embeddings'], batch['attention_mask'])

    def on_train_epoch_start(self):
        self.model.train()

    def training_step(self, batch, batch_idx):
        train_loss = self._compute_loss(batch)
        self.log("train_loss", train_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return train_loss

    def on_train_epoch_end(self):
        ckpt_path = f"{self.config.value.training.ckpt_path}epoch{self.current_epoch}.pth"
        torch.save(self.state_dict(), ckpt_path)
        _print(f"epoch {self.current_epoch} at {ckpt_path}")

    def on_validation_epoch_start(self):
        self.model.eval()

    def validation_step(self, batch, batch_idx):
        val_loss = self._compute_loss(batch)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
    
    def on_test_epoch_start(self):
        self.model.eval()

    def test_step(self, batch):
        test_loss = self._compute_loss(batch)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=False, logger=True)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        path = self.config.value.training
        optimizer = torch.optim.AdamW(self.parameters(), lr=path.lr)
        lr_scheduler = CosineWarmup(
            optimizer,
            warmup_steps=path.dataset_size * path.warmup_ratio,
            total_steps=(path.dataset_size // path.batch_size) * path.epochs
        )
        scheduler_dict = {
            "scheduler": lr_scheduler,
            "interval": 'step',
            'frequency': 1,
            'monitor': 'val_loss',
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler_dict]

    def _compute_loss(self, batch):
        """Helper method to compute loss"""
        embeds, attn_masks, labels = batch['embeddings'], batch['attention_mask'], batch['labels']
        
        preds = self.model(embeds, attn_masks)
        loss = self.loss_fn(preds, labels)

        # only calculate loss over non-pad tokens
        loss_mask = (labels != -1)
        loss *= loss_mask
        loss = loss.sum() / loss_mask.sum()
        return loss

def main():
    # data
    datasets = get_datasets(config)
    data_module = MembraneDataModule(
        config=config,
        train_dataset=datasets['train'],
        val_dataset=datasets['val'],
        test_dataset=datasets['test'],
    )

    # wandb logging
    wandb.init()
    wandb_logger = WandbLogger(**config.wandb)

    # lightning checkpoints
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=config.value.training.ckpt_path,
        filename="best_model",
    )

    # lightning trainer
    trainer = pl.Trainer(
        max_epochs=config.value.training.epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=config.value.training.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10
    )

    # train or evalute the model
    model = SolubilityClassifier(config)
    if config.value.training.mode == "train":
        trainer.fit(model, datamodule=data_module)
        wandb.finish()
    elif config.value.training.mode == "test":
        trainer.test(datamodule=data_module)
        wandb.finish()
    else:
        raise ValueError(f"{config.value.training.mode} is invalid. Must be 'train' or 'test'")


if __name__ == "__main__":
    main()
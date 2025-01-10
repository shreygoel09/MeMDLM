import os
import gc
import torch
import wandb
import torch.nn as nn
import lightning.pytorch as pl

from omegaconf import OmegaConf
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
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
        self._validate_config()

        self.model = ValueModule(config)
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, batch):
        return self.model(batch['embeddings'], batch['attention_mask'])
    
    def on_train_epoch_start(self):
        self.model.train()

    def training_step(self, batch, batch_idx):
        train_loss, _ = self._compute_loss(batch)
        self.log(
            name="train_loss",
            value=train_loss.item(),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True)
        return train_loss

    def on_train_epoch_end(self):
        ckpt_path = f"{self.config.value.training.ckpt_path}epoch{self.current_epoch}.ckpt"
        torch.save(self.state_dict(), ckpt_path)
        _print(f"epoch {self.current_epoch} at {ckpt_path}")

    def on_validation_epoch_start(self):
        self.model.eval()

    def validation_step(self, batch, batch_idx):
        val_loss, _ = self._compute_loss(batch)
        self.log(name="val_loss",
                 value=val_loss.item(),
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 sync_dist=True)
        return val_loss
    
    def on_test_epoch_start(self):
        self.model.eval()

    @torch.no_grad()
    def test_step(self, batch):
        test_loss, preds = self._compute_loss(batch)
        self.log(name="test_loss",
                 value=test_loss.item(),
                 on_step=False,
                 on_epoch=True,
                 logger=True,
                 sync_dist=True)
        
        auroc, accuracy = self._get_metrics(batch, preds)
        _print(f"Test loss: {test_loss}")
        _print(f"Test AUROC: {auroc}")
        _print(f"Test accuracy: {accuracy}")

        return test_loss
    
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
        """Helper method to handle loss calculation"""
        embeds, attn_masks, labels = batch['embeddings'], batch['attention_mask'], batch['labels']
        
        preds = self.model(embeds, attn_masks)
        loss = self.loss_fn(preds, labels)

        # only calculate loss over non-pad tokens
        loss_mask = (labels != self.config.value.batching.label_pad_value) # diff pad value due to binary classes
        loss *= loss_mask
        loss = loss.sum() / loss_mask.sum()
        return loss, preds

    def _validate_config(self):
        """"Helper method to ensure the training parameters are valid"""
        ckpt_path = self.config.value.training.ckpt_path
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
            _print(f"created ckpt dir at {ckpt_path}")
        assert os.path.isdir(self.config.value.training.ckpt_path), "invalid ckpt path"

        assert self.config.value.training.mode in ["train", "test"], f"invalid mode"
        assert os.path.isdir(self.config.value.training.pretrained_model), "invalid MeMDLM model path"
        assert (self.config.value.model.d_model % self.config.value.model.num_heads)==0, "d_model % num_heads != 0"
        assert (self.config.value.model.dropout >= 0 and \
                self.config.value.model.dropout <= 1), "dropout must be btwn 0 and 1"
    
    def _get_metrics(self, batch, preds):
        """Helper method to compute metrics"""
        auroc = BinaryAUROC(preds, batch['labels'])
        accuracy = BinaryAccuracy(preds, batch['labels'])
        return auroc, accuracy


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
    wandb.init(project=config.wandb.project, name=config.wandb.name)
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
        log_every_n_steps=config.value.training.log_n_steps
    )

    # train or evalute the model
    model = SolubilityClassifier(config)
    if config.value.training.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif config.value.training.mode == "test":
        #ckpt_path = os.path.join(config.value.training.ckpt_path, "best_model.ckpt")
        #_print(ckpt_path)
        #model.load_state_dict(torch.load(ckpt_path))

        temp_path = "/workspace/sg666/MeMDLM/checkpoints/classifier/test/epoch1.ckpt"
        checkpoint = torch.load(temp_path,
                                map_location='cuda' if torch.cuda.is_available() else 'cpu',
                                weights_only=True)
        print(checkpoint)
        model.load_state_dict(checkpoint)

        trainer.test(model, datamodule=data_module, ckpt_path=temp_path)
    else:
        raise ValueError(f"{config.value.training.mode} is invalid. Must be 'train' or 'test'")
    
    wandb.finish()


if __name__ == "__main__":
    main()
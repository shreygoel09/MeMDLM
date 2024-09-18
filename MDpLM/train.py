import pytorch_lightning as L
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import config
import torch
from pl_data_loader import get_protein_dataloader
from transformers import AutoTokenizer
from diffusion import Diffusion
import wandb
import sys

# Initialize the fine-tuned MLM tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.Training.MLM_MODEL_PATH)

# Get dataloaders
train_loader, val_loader, _ = get_protein_dataloader(config)

# Get GPU
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Initialize diffusion model
diffusion_model = Diffusion(config, tokenizer=tokenizer)
sys.stdout.flush()

# Define checkpoints to save best model by minimum validation loss
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode='min',
    dirpath="/workspace/a03-sgoel/MDpLM/",
    filename="best_model_epoch{epoch:02d}"
)

# Initialize trainer
trainer = L.Trainer(
    max_epochs=config.Training.NUM_EPOCHS,
    precision=config.Training.PRECISION,
    devices=1,
    accelerator='gpu',
    strategy=DDPStrategy(find_unused_parameters=False),
    accumulate_grad_batches=config.Training.ACCUMULATE_GRAD_BATCHES,
    default_root_dir=config.Training.SAVE_DIR,
    callbacks=[checkpoint_callback]
)

print(trainer)
print("Training model...")
sys.stdout.flush()

# Train the model
trainer.fit(diffusion_model, train_loader, val_loader)

wandb.finish()

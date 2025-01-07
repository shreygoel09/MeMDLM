import abc
import torch
import pandas as pd
import torch.nn as nn
import lightning.pytorch as pl

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from functools import partial
from omegaconf import OmegaConf

from utils import NoisingScheduler


def collate_fn(batch, tokenizer, noise_scheduler):
    sequences = [item['Sequence'] for item in batch]
    tokens = tokenizer(sequences, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
    input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

    noised_tokens = noise_scheduler(input_ids, attention_mask)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'noised_tokens': noised_tokens
    }

def get_datasets(config):
    train_dataset = MembraneDataset(config, config.data.train.membrane_esm_train_path, config.model.mdlm_model_path)
    val_dataset = MembraneDataset(config, config.data.valid.membrane_esm_train_path, config.model.mdlm_model_path)
    test_dataset = MembraneDataset(config, config.data.test.membrane_esm_train_path, config.model.mdlm_model_path)
    
    return  {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

class MembraneDataset(Dataset):
    def __init__(self, config, data_path, mdlm_model_path):
        self.config = config
        self.data = pd.read_csv(data_path)
        self.mdlm_model = AutoModel.from_pretrained(mdlm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(mdlm_model_path)
        self.noise = NoisingScheduler(self.config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["sequence"]
        embeddings, attention_mask = self._get_embeddings(sequence)
        return embeddings, attention_mask

    def _get_embeddings(self, noised_tokens, attention_mask):
        """Following the LaMBO-2 implementation, we obtain embeddings
        from the denoising model to train the discriminator network.

        Args:
            sequence (String): protein sequence to be optimized
        Returns:
            embeds (torch.Tensor) [bsz=1 x seq_len x embedding_dim]: embeddings from MeMDLM
        """
        with torch.no_grad():
            outputs = self.mdlm_model(input_ids=noised_tokens, attention_mask=attention_mask)
            embeds = outputs.last_hidden_state.squeeze(0)#[None, :, :]
        return {
            "embeddings": embeds,
            "attention_mask": attention_mask
        }


class MembraneDataModule(pl.LightningDataModule):
    def __init__(self, config, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.batch_size = config.value.training.batch_size
        self.noise_scheduler = NoisingScheduler(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.finetuned_esm_mdlm_automodel_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer, noise_scheduler=self.noise_scheduler),
                          num_workers=8,
                          pin_memory=True)
    
    def train_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer, noise_scheduler=self.noise_scheduler),
                          num_workers=8,
                          pin_memory=True)
    
    def train_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=partial(self.collate_fn, tokenizer=self.tokenizer, noise_scheduler=self.noise_scheduler),
                          num_workers=8,
                          pin_memory=True)
    
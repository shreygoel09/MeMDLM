import torch
import pandas as pd
import lightning.pytorch as pl

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from MeMDLM.src.guidance.utils import NoisingScheduler


class MembraneDataset(Dataset):
    def __init__(self, config, data_path, mdlm_model_path):
        self.config = config
        self.data = pd.read_csv(data_path)
        self.mdlm_model = AutoModel.from_pretrained(mdlm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(mdlm_model_path)
        self.noise = NoisingScheduler(self.config, self.tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]["Sequence"]

        tokens = self.tokenizer(
            sequence.upper(),
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.config.value.batching.max_seq_len,
        )
        input_ids, attention_masks = tokens['input_ids'], tokens['attention_mask']

        # use MDLM noising scheduler and embeddings for classifier training
        noised_tokens = self.noise(input_ids, attention_masks)
        embeddings = self._get_embeddings(noised_tokens, attention_masks)

        # create and manually pad per-residue labels
        labels = self._get_labels(sequence)

        return {
            "embeddings": embeddings,
            "attention_mask": attention_masks,
            "labels": labels
        }

    def _get_embeddings(self, noised_tokens, attention_mask):
        """Following the LaMBO-2 implementation, we obtain embeddings
        from the denoising model to train the discriminator network.
        """
        with torch.no_grad():
            outputs = self.mdlm_model(input_ids=noised_tokens, attention_mask=attention_mask)
            embeds = outputs.last_hidden_state.squeeze(0)
        return embeds

    def _get_labels(self, sequence):
        max_len = self.config.value.batching.max_seq_len

        # Create per-residue labels
        labels = torch.tensor([1 if residue.islower() else 0 for residue in sequence], dtype=torch.float)
        
        if len(labels) < max_len: # Padding if sequence shorter than tokenizer truncation length
            padded_labels = torch.cat([labels, torch.full(size=(max_len - len(labels),),
                                                          fill_value=self.config.value.batching.label_pad_value)])
        else: # Truncation otherwise
            padded_labels = labels[:max_len]
        return padded_labels



def collate_fn(batch):
    embeds = torch.stack([item['embeddings'] for item in batch])
    masks = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'embeddings': embeds,
        'attention_mask': masks,
        'labels': labels
    }


class MembraneDataModule(pl.LightningDataModule):
    def __init__(self, config, train_dataset, val_dataset, test_dataset, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.collate_fn = collate_fn
        self.batch_size = config.value.training.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)
        self.noise_scheduler = NoisingScheduler(config, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collate_fn,
                          num_workers=8,
                          pin_memory=True)
    

def get_datasets(config):
    """Helper method to grab datasets to quickly init data module in main.py"""
    train_dataset = MembraneDataset(config, config.data.train.membrane_esm_train_path, config.value.training.pretrained_model)
    val_dataset = MembraneDataset(config, config.data.valid.membrane_esm_valid_path, config.value.training.pretrained_model)
    test_dataset = MembraneDataset(config, config.data.test.membrane_esm_test_path, config.value.training.pretrained_model)
    
    return  {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }
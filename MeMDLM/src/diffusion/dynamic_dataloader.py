import random
import torch
import pandas as pd

import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler, SequentialSampler
from datasets import Dataset, load_from_disk
from functools import partial

def collate_fn(batch):
    input_ids = torch.tensor(batch[0]['input_ids'])
    attention_mask = torch.tensor(batch[0]['attention_mask'])

    # if input_ids.shape[1] > 900:
    #     input_ids = input_ids[:, :900]
    #     attention_mask = attention_mask[:, :900]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, tokenizer, batch_size: int = 1, collate_fn=collate_fn):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(self.collate_fn),
            num_workers=8,
            pin_memory=True,
            shuffle=False,
            persistent_workers=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(self.collate_fn),
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=partial(self.collate_fn),
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )


# class DynamicBatchingDataset(Dataset):
#     def __init__(self, csv_path, tokenizer):
#         print('Initializing dataset...')
#         self.sequences = pd.read_csv(csv_path)['Sequences'].tolist()
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         return {"Sequence": self.sequences[idx]} 


# def collate_fn(batch, tokenizer):
#     # Flatten the batch out to collect sequences
#     print(type(batch))
#     print(len(batch))
#     sequences = [seq for dct in batch for seq in dct['labels']]
#     tokens = tokenizer(
#         sequences,
#         return_tensors='pt',
#         truncation=True,
#         padding='max_length',
#         max_length=1024
#     )
#     return {
#         'input_ids': tokens['input_ids'],
#         'attention_mask': tokens['attention_mask'],
#     }


# class DynamicBatchSampler(BatchSampler):
#     def __init__(self, dataset, shuffle, tokenizer, max_tokens=1024):
#         self.tokenizer = tokenizer
#         self.dataset = dataset
#         self.shuffle = shuffle
#         self.max_tokens = max_tokens

#         self.sampler = SequentialSampler(dataset)
#         self.batches = self.create_batches()

#         # Must initialize BatchSampler with a predefined sampler
#         super().__init__(self.sampler, batch_size=1, drop_last=False)

#     def create_batches(self):
#         batches = []
#         curr_batch = []
#         curr_tokens = 0

#         idxs = list(range(len(self.dataset)))
#         random.shuffle(idxs)

#         for idx in idxs:
#             seqs = self.dataset[idx]['labels']
#             seq_len = [
#                 len(self.tokenizer(seq, truncation=True, max_length=self.max_tokens)['input_ids'])
#                 for seq in seqs
#             ]

#             if curr_tokens + seq_len > self.max_tokens:
#                 if curr_batch:
#                     batches.append(curr_batch)
#                 curr_batch = []
#                 curr_tokens = 0
            
#             curr_batch.append(idx)
#             curr_tokens += seq_len
    
#         if curr_batch:
#             batches.append(curr_batch)
        
#         return batches

#     def __iter__(self):
#         for batch in self.batches:
#             yield batch
    
#     def __len__(self):
#         return len(self.batches)


# class CustomDataModule(pl.LightningDataModule):
#     def __init__(
#             self,
#             train_dataset,
#             val_dataset,
#             test_dataset,
#             tokenizer,
#             batch_size=1,
#             max_tokens=1024
#     ):
#         super().__init__()
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.test_dataset = test_dataset
#         self.tokenizer = tokenizer
#         self.bsz = batch_size
#         self.max_tokens = max_tokens
    
#     def _make_dataloader(self, dataset, shuffle):
#         return DataLoader(
#             dataset,
#             batch_sampler=DynamicBatchSampler(dataset, shuffle, self.tokenizer),
#             collate_fn=partial(collate_fn, tokenizer=self.tokenizer),
#             num_workers=12,
#             pin_memory=True
#         )  
        
#     def train_dataloader(self):
#         return self._make_dataloader(self.train_dataset, shuffle=True)

#     def val_dataloader(self):
#         return self._make_dataloader(self.val_dataset, shuffle=False)

#     def test_dataloader(self):
#         return self._make_dataloader(self.test_dataset, shuffle=False)


import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
MAX_LENGTH = 1024

def mask_and_pad_seq(item):
    sequences = item['Sequence'].upper()

    if item["Label"] == 0: mask = [1 if i.isupper() else 0 for i in item["Sequence"]]
    else: mask = [0 if i.isupper() else 1 for i in item["Sequence"]]
    mask = [1] + mask
    if len(mask) > MAX_LENGTH: mask = mask[:MAX_LENGTH]
    elif len(mask) < MAX_LENGTH: mask += [1] * (MAX_LENGTH - len(mask))
    print(mask)

    tokens = tokenizer(sequences, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH)

    masked_tokens = tokens.input_ids.where(torch.as_tensor(mask)==1, torch.full_like(tokens.input_ids, tokenizer.mask_token_id))
    # print(masked_tokens.size())
    return tokenizer.decode(masked_tokens.squeeze())

if __name__ == "__main__":
    filename = sys.argv[1]
    
    dataset = load_dataset("csv", data_files=filename)["train"]
    
    output = mask_and_pad_seq(dataset[1])
    print(output)
    
import pl_data_loader as dataloader
from datasets import load_dataset
from transformers import AutoTokenizer


if __name__ == "__main__":
    tokenizer = tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    train_dataset = load_dataset('csv', data_files="data/membrane/train.csv")
    val_dataset = load_dataset('csv', data_files="data/membrane/val.csv")
    test_dataset = load_dataset('csv', data_files="data/membrane/test.csv")

    data = dataloader.CustomDataModule(train_dataset["train"], val_dataset["train"], test_dataset["train"], batch_size=4, collate_fn=dataloader.membrane_collate_fn, tokenizer=tokenizer)
    loader = data.train_dataloader()
    print(next(iter(loader)))
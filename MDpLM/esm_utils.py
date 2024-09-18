import torch
import config
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

def load_esm2_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    masked_model = AutoModelForMaskedLM.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)
    return tokenizer, masked_model, embedding_model

def get_latents(model, tokenizer, sequence):
    inputs = tokenizer(sequence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits
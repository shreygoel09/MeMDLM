import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

def load_esm2_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    masked_model = AutoModelForMaskedLM.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)
    return tokenizer, masked_model, embedding_model

def get_latents(model, tokenizer, sequence, device):
    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.squeeze(0)
    return outputs
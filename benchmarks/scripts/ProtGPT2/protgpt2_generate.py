from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import torch
import sys
import pandas as pd

# Function to calculate perplexity of each generated sequence
def calculate_perplexity(sequence, model, tokenizer, device):
    sequence = "<|endoftext|>"  + sequence + "<|endoftext|>"
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, _ = outputs[:2]
    return math.exp(loss)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(device)
    sys.stdout.flush()
    path = "/workspace/sg666/MeMDLM/MeMDLM/benchmarks"

    # Load fine-tuned model and tokenizer
    model_path = path + "/scripts/ProtGPT2/finetuned_models/checkpoint-1985/"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate sequences
    protgpt2 = pipeline('text-generation', model=model_path, device=device)
    sequences = protgpt2("", max_length=500, do_sample=True, top_k=950, repetition_penalty=1.5, num_return_sequences=100, eos_token_id=0)

    # Store generated sequences and their associated perplexities
    generated_sequences = []
    perplexities = []

    # Calculate PPL for sequences
    for item in sequences:
        raw_sequence = item['generated_text']
        ppl = calculate_perplexity(raw_sequence, model, tokenizer, device)
        generated_sequences.append(raw_sequence)
        perplexities.append(ppl)

    # Clean the generated sequences
    cleaned_sequences = [seq.replace('\n', '').replace('<|endoftext|>', '') for seq in generated_sequences]

    # Create df with cleaned sequences and perplexities
    df = pd.DataFrame({"Sequence": cleaned_sequences, "Perplexity": perplexities})
    df.sort_values(by='Perplexity', inplace=True)

    # Save results
    df.to_csv(path + "/results/de_novo/protgpt/de_novo_protgpt.csv", index=False)

    # View the average de novo generation perplexity
    avg_generation_ppl = df.loc[:, 'Perplexity'].mean()
    print(f'Average de novo generation perplexity: {avg_generation_ppl}')


if __name__ == "__main__":
    main()
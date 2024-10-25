import torch
import torch.nn.functional as F
import math
import random
import sys
import pandas as pd
from generate_utils import mask_for_de_novo, calculate_cosine_sim, calculate_hamming_dist
from diffusion import Diffusion
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline


@torch.no_grad()
def generate_sequence(sequence_length: int, tokenizer, mdlm: Diffusion):
    global masked_sequence
    masked_sequence = mask_for_de_novo(sequence_length)
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    generated_sequence = tokenizer.decode(logits.squeeze())

    return generated_sequence


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    path = "/workspace/sg666/MDpLM"

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    mdlm_model = Diffusion.load_from_checkpoint(config.eval.checkpoint_path, config=config, tokenizer=tokenizer)
    
    mdlm_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mdlm_model.to(device)

    print("loaded models...")

    # Get 100 random sequence lengths to generate
    sequence_lengths = [random.randint(50, 500) for _ in range(700)] 

    generation_results = []
    for seq_length in tqdm(sequence_lengths, desc=f"Generating sequences: "):
        generated_sequence = generate_sequence(seq_length, tokenizer, mdlm_model)
        generated_sequence = generated_sequence[5:-5].replace(" ", "") # Remove bos/eos tokens & spaces between residues
        
        perplexity = mdlm_model.compute_masked_perplexity([generated_sequence], masked_sequence)
        perplexity = round(perplexity, 4)

        generation_results.append([generated_sequence, perplexity])

        print(f"perplexity: {perplexity} | length: {seq_length} | generated sequence: {generated_sequence}")
        sys.stdout.flush()

        df = pd.DataFrame(generation_results, columns=['Generated Sequence', 'Perplexity'])
        df.to_csv(path + f'/benchmarks/mdlm_de-novo_generation_results.csv', index=False)
        

    
if __name__ == "__main__":
    main()
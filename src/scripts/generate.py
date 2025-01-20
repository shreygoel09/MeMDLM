import sys

import hydra
import random
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

from generate_utils import mask_for_de_novo, mask_for_scaffold
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.guidance import SolubilityGuider

data_path = "/workspace/sg666/MeMDLM/MeMDLM/data/membrane/test.csv"


@torch.no_grad()
def generate_sequence(sequence_length: int, tokenizer, mdlm: Diffusion):
    global masked_sequence
    masked_sequence = mask_for_de_novo(sequence_length)
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    ids = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    generated_sequence = tokenizer.decode(ids.squeeze())[5:-5].replace(" ", "") # bos/eos tokens & spaces between residues

    return generated_sequence

cfg_pth = '/workspace/sg666/MeMDLM/MeMDLM/configs'
@hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
def main(config, optimize: bool=True):
    path = "/workspace/sg666/MeMDLM"

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')

    mdlm = Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config,
        tokenizer=tokenizer
    ).eval().to(device)

    guidance = SolubilityGuider(config, device, mdlm) # Pass Diffusion model to prevent loading twice
    
    with open(data_path, "r") as f:
        sequences = f.readlines()[1:]

    print("loaded models...")

    # Get 100 random sequence lengths to generate
    #sequence_lengths = [random.randint(50, 500) for _ in range(700)] 

    generation_results = []
    for sequence in tqdm(sequences, desc=f"Generating sequences: "):
        og_sequence = mask_for_scaffold(sequence, "uppercase") # SOLUBLE IS LOWERCASE!!!!!!!
        tokens = tokenizer(og_sequence, return_tensors='pt')
        
        # calculate og solubility
        nlower = 0
        for s in sequence: 
            if s.islower(): nlower+=1
        
        raw_solub_frac = nlower / len(sequence)
        og_tokens = tokenizer(sequence, return_tensors='pt')
        og_solubility_preds = guidance.embed_and_predict_solubility(og_tokens["input_ids"], og_tokens["attention_mask"])["solubility_predictions"]
        og_solubility = guidance.compute_frac_soluble(og_tokens["input_ids"], og_solubility_preds)

        sys.stdout.flush()
        if optimize:
            optimized_sequence, sequence_str = guidance.sample_guidance(x_0 = tokens, guidance=True)
            no_guide_seq, no_guide_str = guidance.sample_guidance(x_0 = tokens, guidance=False)
        
        # og_ppl = mdlm.compute_masked_perplexity([og_sequence], masked_sequence)
        # print("size: ", optimized_sequence.size())
        # optimized_ppl = mdlm.compute_masked_perplexity([sequence_str], og_sequence)
        # optimized_ppl = round(optimized_ppl, 4)
        
        print("sequence: ", sequence)
        print("optimized: ", sequence_str.replace(" ", ""))
        
        tokens = tokenizer(sequence_str, return_tensors="pt")
        solubility_preds = guidance.embed_and_predict_solubility(tokens["input_ids"], tokens["attention_mask"])["solubility_predictions"]
        optimized_solubility = guidance.compute_frac_soluble(tokens["input_ids"], solubility_preds)
        
        no_tokens = tokenizer(no_guide_str, return_tensors="pt")
        no_solubility_preds = guidance.embed_and_predict_solubility(no_tokens["input_ids"], no_tokens["attention_mask"])["solubility_predictions"]
        no_optimized_solubility = guidance.compute_frac_soluble(no_tokens["input_ids"], no_solubility_preds)

        generation_results.append([og_sequence, optimized_sequence, 123])

        # print(f"original ppl: {og_ppl} | original solubility: {og_solubility} | length: {seq_length} | generated sequence: {og_sequence}")
        print(f"og solubility: {og_solubility} | raw solubility: {raw_solub_frac} | no guide solubility: {no_optimized_solubility} | optimized solubility: {optimized_solubility} | solubility diff: {optimized_solubility - og_solubility}")
        sys.stdout.flush()

    # df = pd.DataFrame(generation_results, columns=['OG Sequence', 'OG PPL', 'Optimized Sequence', 'Optimized PPL'])
    # df.to_csv(path + f'/benchmarks/de_novo_generation_results.csv', index=False)

    
if __name__ == "__main__":
    main()
import sys

import hydra
import random
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer

from generate_utils import mask_for_de_novo
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.guidance import SolubilityGuider



@torch.no_grad()
def generate_sequence(sequence_length: int, tokenizer, mdlm: Diffusion):
    global masked_sequence
    masked_sequence = mask_for_de_novo(sequence_length)
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    generated_sequence = tokenizer.decode(logits.squeeze())[5:-5].replace(" ", "") # bos/eos tokens & spaces between residues

    return generated_sequence, logits

cfg_pth = '/workspace/sg666/MeMDLM/MeMDLM/configs'
@hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
def main(config, optimize: bool=True):
    path = "/workspace/sg666/MeMDLM"

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")

    device = torch.device(f"cuda:{4}" if torch.cuda.is_available() else 'cpu')

    mdlm = Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config,
        tokenizer=tokenizer
    ).eval().to(device)

    guidance = SolubilityGuider(config, device, mdlm) # Pass Diffusion model to prevent loading twice

    print("loaded models...")

    # Get 100 random sequence lengths to generate
    #sequence_lengths = [random.randint(50, 500) for _ in range(700)] 
    sequence_lengths = [100]

    generation_results = []
    for seq_length in tqdm(sequence_lengths, desc=f"Generating sequences: "):
        og_sequence, og_logits = generate_sequence(seq_length, tokenizer, mdlm)
        tokens = tokenizer(og_sequence, return_tensors='pt')
        ids, masks = tokens['input_ids'], tokens['attention_mask']
        og_preds = guidance.embed_and_predict_solubility(ids, masks)['solubility_predictions']
        og_solubility = guidance.compute_frac_soluble(ids, og_preds)

        if optimize:
            optimized_sequence, optimized_solubility = guidance.optimized_sampling(og_sequence, og_logits)
        
        og_ppl = mdlm.compute_masked_perplexity([og_sequence], masked_sequence)
        optimized_ppl = mdlm.compute_masked_perplexity([optimized_sequence], masked_sequence)
        og_ppl, optimized_ppl = round(og_ppl, 4), round(optimized_ppl, 4)

        generation_results.append([og_sequence, og_ppl, optimized_sequence, optimized_ppl])

        print(f"original ppl: {og_sequence} | origional solubility: {og_solubility} | length: {seq_length} | generated sequence: {og_sequence}")
        print(f"optimized ppl: {optimized_ppl} | optimized solubility: {optimized_solubility} | length: {seq_length} | generated sequence: {optimized_sequence}")
        print('\n')
        sys.stdout.flush()

    df = pd.DataFrame(generation_results, columns=['OG Sequence', 'OG PPL', 'Optimized Sequence', 'Optimized PPL'])
    df.to_csv(path + f'/benchmarks/de_novo_generation_results.csv', index=False)

    
if __name__ == "__main__":
    main()
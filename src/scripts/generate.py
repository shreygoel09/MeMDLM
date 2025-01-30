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


cfg_pth = '/workspace/sg666/MeMDLM/MeMDLM/configs'


def _print_og(og_sol, og_ppl, og_seq):
    print(f"OG solubility: {og_sol} | OG PPL: {og_ppl} | OG Sequence: {og_seq}")

def _print_optim(optim_sol, optim_ppl, optim_seq):
    print(f'Optim solubility: {optim_sol} | Optim PPL: {optim_ppl} | Optim Seq: {optim_seq[5:-5].replace(" ", "")}')


@torch.no_grad()
def generate_sequence(prior: str, tokenizer, mdlm: Diffusion):
    inputs = tokenizer(prior, return_tensors="pt").to(mdlm.device)
    ids = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    generated_sequence = tokenizer.decode(ids.squeeze())[5:-5].replace(" ", "") # bos/eos tokens & spaces between residues

    return generated_sequence, inputs

@hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
def main(config):
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    device = torch.device(f"cuda:{6}" if torch.cuda.is_available() else 'cpu')

    mdlm = Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config,
        tokenizer=tokenizer
    ).eval().to(device)

    guidance = SolubilityGuider(config, device, mdlm) # Pass Diffusion model to prevent loading twice

    print("loaded models...")

    # Get 100 random sequence lengths to generate
    sequence_lengths = [random.randint(150, 500) for _ in range(100)]

    generation_results = []
    for length in tqdm(sequence_lengths, desc=f"Generating sequences: "):
        seq_res = []

        # Generate a prior of all masked tokens
        masked_seq = mask_for_de_novo(length)

        # Unconditionally generate a sequence and calculate its perplexity and solubility
        og_seq, og_tokens = generate_sequence(masked_seq, tokenizer, mdlm)
        og_solubility_preds = guidance.embed_and_predict_solubility(
            input_ids=og_tokens['input_ids'],
            attention_masks=og_tokens['attention_mask']
        )['solubility_predictions']
        og_solubility = round(guidance.compute_frac_soluble(og_tokens["input_ids"], og_solubility_preds), 4)
        og_ppl = round(mdlm.compute_masked_perplexity([og_seq], masked_seq), 4)

        seq_res.append(og_seq)
        seq_res.append(og_ppl)
        seq_res.append(og_solubility)
        _print_og(og_solubility, og_ppl, og_seq)

        # Do guidance steps if original sequence is insoluble
        if og_solubility < config.value.guidance.sequence_thresh:
            try:
                optim_tokens, optim_seq = guidance.sample_guidance(x_0=og_tokens, guidance=True)
                optim_solubility_preds = guidance.embed_and_predict_solubility(
                    input_ids=optim_tokens,
                    attention_masks=torch.ones_like(optim_tokens)
                )["solubility_predictions"]
                optim_solubility = round(guidance.compute_frac_soluble(optim_tokens, optim_solubility_preds), 4)
                
                try:
                    optim_ppl = round(mdlm.compute_masked_perplexity([optim_seq], masked_seq), 4)
                except:
                    optim_seq = ""
                    optim_solubility = 0
                    optim_ppl = 0

                seq_res.append(optim_seq)
                seq_res.append(optim_ppl)
                seq_res.append(optim_solubility)
                _print_optim(optim_solubility, optim_ppl, optim_seq)
                print("\n")
            except:
                seq_res.append("")
                seq_res.append(0)
                seq_res.append(0)
                print("No optimization required.")
                print("\n")
                continue

        else:
            seq_res.append("")
            seq_res.append(0)
            seq_res.append(0)
            print("No optimization required.")
            print("\n")

        generation_results.append(seq_res)

    df = pd.DataFrame(generation_results, columns=['OG Sequence', 'OG PPL', 'OG Solubility', 'Optimized Sequence', 'Optimized PPL', 'Optimized Solubility'])
    df.to_csv('/workspace/sg666/MeMDLM/MeMDLM/benchmarks/results/de_novo.csv', index=False)

    
if __name__ == "__main__":
    main()
import sys
import os

import hydra
import random
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from generate_utils import mask_for_de_novo, mask_for_scaffold, calculate_perplexity
from MeMDLM.src.diffusion.diffusion import Diffusion
# from MeMDLM.src.guidance.guidance import SolubilityGuider


cfg_pth = '/raid/sg666/MeMDLM/MeMDLM/configs'
data_path = "/raid/sg666/MeMDLM/MeMDLM/data/membrane/test.csv"


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
    csv_save_path = f'/raid/sg666/MeMDLM/MeMDLM/benchmarks/results/de_novo/mdlm/{config.wandb.name}'
    os.makedirs(csv_save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    #device = torch.device(f"cuda:{7}" if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    mdlm = Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        config=config,
        tokenizer=tokenizer
    ).eval().to(device)

    #guidance = SolubilityGuider(config, device, mdlm) # Pass Diffusion model to prevent loading twice

    esm_model = AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(device)
    esm_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

    print("loaded models...")

    # Get 100 random sequence lengths to generate
    seq_lengths = [random.randint(50, 250) for _ in range(200)]

    generation_results = []
    for seq_len in tqdm(seq_lengths, desc=f"Generating sequences: "):
        seq_res = []

        # Generate a prior of all masked tokens
        masked_seq = mask_for_de_novo(seq_len) # mask non-soluble residues
        
        og_seq, og_tokens = generate_sequence(masked_seq, tokenizer, mdlm)

        #test_seq_ppl = round(mdlm.compute_masked_perplexity([og_seq], masked_seq), 4)

        pseudo_ppl = calculate_perplexity(esm_model, esm_tokenizer, og_seq, [i for i in range(len(og_seq))])
        
        # og_solubility_preds = guidance.embed_and_predict_solubility(
        #     input_ids=og_tokens['input_ids'],
        #     attention_masks=og_tokens['attention_mask']
        # )['solubility_predictions']
        # og_solubility = round(guidance.compute_frac_soluble(og_tokens["input_ids"], og_solubility_preds), 4)
        
        seq_res.append(og_seq)
        seq_res.append(pseudo_ppl)
        #seq_res.append(og_solubility)

        # Unconditionally generate a sequence and calculate its perplexity and solubility
        # og_seq, og_tokens = generate_sequence(masked_seq, tokenizer, mdlm)
        # og_solubility_preds = guidance.embed_and_predict_solubility(
        #     input_ids=og_tokens['input_ids'],
        #     attention_masks=og_tokens['attention_mask']
        # )['solubility_predictions']
        # og_solubility = round(guidance.compute_frac_soluble(og_tokens["input_ids"], og_solubility_preds), 4)
        # og_ppl = round(mdlm.compute_masked_perplexity([og_seq], masked_seq), 4)

        # seq_res.append(og_seq)
        # seq_res.append(og_ppl)
        # seq_res.append(og_solubility)
        # _print_og(og_solubility, og_ppl, og_seq)

        # Do guidance steps if original sequence is insoluble
        # if og_solubility < config.value.guidance.sequence_thresh:
        #     try:
        #         optim_tokens, optim_seq = guidance.sample_guidance(x_0=og_tokens, guidance=True)
        #         optim_solubility_preds = guidance.embed_and_predict_solubility(
        #             input_ids=optim_tokens,
        #             attention_masks=torch.ones_like(optim_tokens)
        #         )["solubility_predictions"]
        #         optim_solubility = round(guidance.compute_frac_soluble(optim_tokens, optim_solubility_preds), 4)
                
        #         try:
        #             optim_ppl = round(mdlm.compute_masked_perplexity([optim_seq], masked_seq), 4)
        #         except:
        #             optim_seq = ""
        #             optim_solubility = 0
        #             optim_ppl = 0

        #         seq_res.append(optim_seq)
        #         seq_res.append(optim_ppl)
        #         seq_res.append(optim_solubility)
        #         _print_optim(optim_solubility, optim_ppl, optim_seq)
        #         print("\n")
        #     except Exception as e:
        #         print("ERROR: ", e)
        #         seq_res.append("")
        #         seq_res.append(0)
        #         seq_res.append(0)
        #         print("No optimization required.")
        #         print("\n")
        #         continue

        # else:
        #     seq_res.append("")
        #     seq_res.append(0)
        #     seq_res.append(0)
        #     print("No optimization required.")
        #     print("\n")

        generation_results.append(seq_res)

    # df = pd.DataFrame(generation_results, columns=['OG Sequence', 'OG PPL', 'Unguided Sequence', 'Unguided PPL', 'Unguided Solubility', 'Optimized Sequence', 'Optimized PPL', 'Optimized Solubility'])
    df = pd.DataFrame(generation_results, columns=['Generated Sequence', 'Pseudo Perplexity'])
    df.to_csv(csv_save_path + "/denovo_seqs.csv", index=False)

    
if __name__ == "__main__":
    main()
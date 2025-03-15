import torch
import torch.nn.functional as F
import math
import random
import sys
import pandas as pd
from generate_utils import mask_for_scaffold, calculate_cosine_sim, calculate_hamming_dist
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, pipeline
from evodiff.utils import Tokenizer
from evodiff.pretrained import OA_DM_38M
# from evodiff_utils import compute_masked_perplexity
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.guidance import SolubilityGuider
import numpy as np

device = torch.device("cuda:1")

# @torch.no_grad()
def compute_pseudo_perplexity(model, tokenizer, protein_seq): 
    '''
    Computes the pseudo-perplexity of a protein sequence using the ESM-2-650M pLM.
    '''
    tensor_input = tokenizer(protein_seq,
    return_tensors='pt').to(model.device)["input_ids"]
    # print(tensor_input.size(), tensor_input)

    total_loss = 0

    # Loop through each token in the sequence
    for i in range(-len(protein_seq)-1, -1):

        # Create a copy of the original tensor  
        masked_input = tensor_input.clone()
        # print(masked_input.size())
        # Mask one token at a time
        masked_input[:, i] = tokenizer.mask_token_id

        # Create labels
        labels = torch.full(tensor_input.shape,-100).to(model.device)
        labels[:, i] = tensor_input[:,i]

        # Get model prediction and loss
        with torch.no_grad():
            outputs = model(masked_input, labels=labels)
            total_loss += outputs.loss.item()
            # print(outputs.loss)

        # Calculate the average loss
        avg_loss = total_loss / len(protein_seq)

    # Calculate pseudo perplexity
    pseudo_perplexity = np.exp(avg_loss)
    # print(pseudo_perplexity)
    return pseudo_perplexity

# default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_evodiff_scaffold_from_sequence(model, motif_seq, tokenizer, batch_size=1, device=device):
    mask = tokenizer.mask_id

    sample = torch.as_tensor(tokenizer.tokenize([motif_seq])).to(device)

    # Create input motif + scaffold
    loc = torch.arange(0, len(motif_seq)).to(device)[sample==mask].cpu().numpy()
    np.random.shuffle(loc)
    
    sample = sample.to(device).unsqueeze(0)
    og_sample = sample.clone()
    
    with torch.no_grad():
        for i in loc:
            timestep = torch.tensor([0] * batch_size).to(device)  # placeholder but not called in model
            timestep = timestep.to(device)
            prediction = model(sample, timestep)
            p = prediction[:, i, :len(tokenizer.all_aas) - 6]  # only canonical
            p = F.softmax(p, dim=1)  # softmax over categorical probs
            p_sample = torch.multinomial(p, num_samples=1)
            sample[:, i] = p_sample.squeeze()
    output = [tokenizer.untokenize(s) for s in sample]

    return output[0] if batch_size==1 else output, og_sample, loc


cfg_pth = '/workspace/sg666/MeMDLM/MeMDLM/configs'


@torch.no_grad()
def generate_scaffold_mdlm(sequence: str, generate_case: str, tokenizer, mdlm: Diffusion):
    masked_sequence = mask_for_scaffold(sequence, generate_case)
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    # logits = mdlm.forward(inputs)
    # print(tokenizer.decode(logits.squeeze(), skip_special_tokens=True))

    return tokenizer.decode(logits.squeeze()), masked_sequence


@hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
def mdlm_motif_benchmark(config):
    path = "/workspace/sg666/MeMDLM/MeMDLM"
    
    test_sequences = pd.read_csv(path + "/data/test2.csv")['Sequence'].tolist()

    checkpoint = OA_DM_38M()
    model, collater, tokenizer, scheme = checkpoint
    model = model.to(device)
    
    # device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
    
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    mlm_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)

    # mdlm = Diffusion.load_from_checkpoint(
    #     config.eval.checkpoint_path,
    #     config=config,
    #     tokenizer=esm_tokenizer
    # ).eval().to(device)

    # guidance = SolubilityGuider(config, device, mdlm)

    # esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D").to(device) # model used for functionality testing
    
    print("load models...")

    for generate_case in ['uppercase', 'lowercase']:
        case_results = []
        for original_sequence in tqdm(test_sequences, desc=f"scaffolding ({generate_case}): "):

            masked_input = mask_for_scaffold(original_sequence, generate_case, mask_key="#")
            original_input = torch.as_tensor(tokenizer.tokenize([original_sequence.upper()])).to(device=device)
            
            generated_sequence, masked_seq, loc = generate_evodiff_scaffold_from_sequence(model, masked_input, tokenizer)
            
            # generated_tokens = esm_tokenizer(generated_sequence)
            
            # og_solubility_preds = guidance.embed_and_predict_solubility(
            #     input_ids=generated_tokens['input_ids'],
            #     attention_masks=generated_tokens['attention_mask']
            # )['solubility_predictions']
            # og_solubility = round(guidance.compute_frac_soluble(generated_tokens["input_ids"], og_solubility_preds), 4)
            
            # perplexity = compute_pseudo_perplexity(mlm_model, esm_tokenizer, generated_sequence)
            cos_sim = calculate_cosine_sim(original_sequence, generated_sequence, esm_tokenizer, esm_model, device)
            # hamming_distance = calculate_hamming_dist(original_sequence, generated_sequence)
        
            # case_results.append([original_sequence, generated_sequence, perplexity, cos_sim])
            case_results.append([original_sequence, generated_sequence, cos_sim])

            # print("perplexity: ", perplexity, "cos sim: ", cos_sim)
            sys.stdout.flush()

        # df = pd.DataFrame(case_results, columns=['Original Sequence', 'Generated Sequence', 'Perplexity', 'Cosine Similarity'])
        df = pd.DataFrame(case_results, columns=['Original Sequence', 'Generated Sequence', 'Cosine Similarity'])
        df.to_csv(path + f'/benchmarks/oldtestdata/evodiff_{generate_case}_results.csv', index=False)
        

    
if __name__ == "__main__":
    mdlm_motif_benchmark()
    

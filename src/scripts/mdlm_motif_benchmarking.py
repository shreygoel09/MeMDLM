import torch
import torch.nn.functional as F
import math
import random
import sys
import pandas as pd
from mlm_generate_utils import mask_for_scaffold, calculate_cosine_sim, calculate_hamming_dist
from MeMDLM.src.models.diffusion import Diffusion
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline

def masking_test(sequence: str, generate_case: str, tokenizer, mask_prob: float = 0.50):
    """
    Masks 50% of the tokens in the sequence.
    """
    tokens = list(sequence.upper()) 
    num_tokens_to_mask = int(mask_prob * len(tokens))  # Select some fraction of the tokens
    print(num_tokens_to_mask,len(tokens))
    
    # Get random indices to mask
    mask_indices = random.sample(range(len(tokens)), num_tokens_to_mask)
    
    for idx in mask_indices:
        tokens[idx] = tokenizer.mask_token  # Replace with mask token
    
    return ''.join(tokens)



@torch.no_grad()
def generate_scaffold_mdlm(sequence: str, generate_case: str, tokenizer, mdlm: Diffusion):
    # # Mask soluble or transmembrane domains
    # masked_sequence = mask_for_scaffold(sequence, generate_case)

    # # Test out different masking rates
    # masked_sequence = masking_test(sequence, generate_case, tokenizer)

    # 100% masking rate, de novo generation
    masked_sequence = len(sequence) * "<mask>"

    print(masked_sequence)

    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    # logits = mdlm.forward(inputs)
    # print(tokenizer.decode(logits.squeeze(), skip_special_tokens=True))

    return tokenizer.decode(logits.squeeze()), masked_sequence


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    path = "/workspace/sg666/MDpLM"
    
    test_sequences = pd.read_csv(path + "/data/membrane/test.csv")['Sequence'].tolist()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    mdlm_model = Diffusion.load_from_checkpoint(config.eval.checkpoint_path, config=config, tokenizer=tokenizer)
    esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D") # model used for functionality testing
    
    mdlm_model.eval()
    esm_model.eval()
    
    print("loaded models...")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mdlm_model.to(device)
    esm_model.to(device)

    for generate_case in ['uppercase', 'lowercase']:
        case_results = []
        for original_sequence in tqdm(test_sequences, desc=f"scaffolding ({generate_case}): "):

            generated_sequence, masked_input = generate_scaffold_mdlm(original_sequence, generate_case, tokenizer, mdlm_model)
            generated_sequence = generated_sequence[5:-5].replace(" ", "") # Remove bos/eos tokens
            
            perplexity = mdlm_model.compute_masked_perplexity([original_sequence], masked_input)
            cos_sim = calculate_cosine_sim(original_sequence, generated_sequence, tokenizer, esm_model, device)
            hamming_distance = calculate_hamming_dist(original_sequence, generated_sequence)
        
            case_results.append([original_sequence, generated_sequence, perplexity, cos_sim, hamming_distance])

            print("perplexity: ", perplexity, "cos sim: ", cos_sim, "hamming: ", hamming_distance)
            print(f"generated sequence: {generated_sequence}")
            print(f"original sequence: {original_sequence.upper()}")
            sys.stdout.flush()

        df = pd.DataFrame(case_results, columns=['Original Sequence', 'Generated Sequence', 'Perplexity', 'Cosine Similarity', 'Hamming Distance'])
        df.to_csv(path + f'/benchmarks/MLM/mlm_{generate_case}_results.csv', index=False)
        

    
if __name__ == "__main__":
    main()
import torch
import torch.nn.functional as F
import math
import random
import sys
import pandas as pd
from mlm_generate_utils import mask_for_scaffold, calculate_cosine_sim, calculate_hamming_dist
from diffusion import Diffusion
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline

# def mask_for_scaffold(sequence: str, generate_case: str, tokenizer, mask_prob: float = 0.50):
#     """
#     Masks 50% of the tokens in the sequence.
#     """
#     tokens = list(sequence.upper())  # Assuming the sequence is tokenized as individual characters
#     num_tokens_to_mask = int(mask_prob * len(tokens))  # 50% of the tokens
#     print(num_tokens_to_mask,len(tokens))
    
#     # Get random indices to mask
#     mask_indices = random.sample(range(len(tokens)), num_tokens_to_mask)
    
#     for idx in mask_indices:
#         tokens[idx] = tokenizer.mask_token  # Replace with mask token
    
#     # print(tokens)
    
#     # print(' '.join(tokens))

#     return ''.join(tokens)



@torch.no_grad()
def generate_scaffold_mdlm(sequence: str, generate_case: str, tokenizer, mdlm: Diffusion):
    masked_sequence = mask_for_scaffold(sequence, generate_case)
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    
    logits = mdlm._sample(x_input=inputs) # using sample, change config.sampling.steps to determine robustness
    # logits = mdlm.forward(inputs)
    # print(tokenizer.decode(logits.squeeze(), skip_special_tokens=True))

    return tokenizer.decode(logits.squeeze()), masked_sequence


@hydra.main(version_base=None, config_path='configs', config_name='config')
def mdlm_motif_benchmark(config):
    path = "/workspace/vnt/MeMDLM"
    
    test_sequences = pd.read_csv(path + "/data/membrane/test.csv")['Sequence'].tolist()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    mlm_model = Diffusion.load_from_checkpoint(config.CKPT_DIR + "/best.ckpt", config=config, tokenizer=tokenizer)
    ## config.training.esm_model_path = config.CKPT_DIR + "/best.ckpt"
    # mlm_model = Diffusion(config, tokenizer=tokenizer) # should autoload the checkpoint with config
    print(mlm_model.backbone.model.esm.encoder.layer[0].attention.self.query.weight)
    mlm_model.eval()
    # mlm_model = pipeline("fill-mask", model="facebook/esm2_t30_150m_UR50D")   
    # esm_model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D") # model used for functionality testing
    
    print("load models...")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mlm_model.to(device)
    esm_model.to(device)

    for generate_case in ['uppercase', 'lowercase']:
        case_results = []
        for original_sequence in tqdm(test_sequences, desc=f"scaffolding ({generate_case}): "):

            generated_sequence, masked_input = generate_scaffold_mdlm(original_sequence, generate_case, tokenizer, mlm_model)
            # print(generated_sequence)
            
            perplexity = mlm_model.compute_masked_perplexity([original_sequence], masked_input)
            # cos_sim = calculate_cosine_sim(original_sequence, generated_sequence, tokenizer, esm_model, device)
            # hamming_distance = calculate_hamming_dist(original_sequence, generated_sequence)
        
            # case_results.append([original_sequence, generated_sequence, perplexity, cos_sim, hamming_distance])

            # print("perplexity: ", perplexity, "cos sim: ", cos_sim, "hamming: ", hamming_distance)
            print("perplexity: ", perplexity)
            sys.stdout.flush()

        df = pd.DataFrame(case_results, columns=['Original Sequence', 'Generated Sequence', 'Perplexity', 'Cosine Similarity', 'Hamming Distance'])
        df.to_csv(path + f'/benchmarks/MLM/mlm_{generate_case}_results.csv', index=False)
        

    
if __name__ == "__main__":
    mdlm_motif_benchmark()
    

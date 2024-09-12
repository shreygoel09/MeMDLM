import torch
import torch.nn.functional as F
import math
import sys
import pandas as pd
from mlm_generate_utils import mask_for_scaffold, calculate_cosine_sim, calculate_hamming_dist
from diffusion import Diffusion
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


@torch.no_grad()
def generate_scaffold_mdlm(sequence: str, generate_case: str, tokenizer, mdlm: Diffusion):
    masked_sequence = mask_for_scaffold(sequence, generate_case)
    inputs = tokenizer(masked_sequence, return_tensors="pt").to(mdlm.device)
    
    logits = mdlm.forward(inputs.input_ids, sigma=torch.tensor([1], device=mdlm.device), attention_mask=inputs.attention_mask)
    
    mask_token_indices = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    logits_at_masks = logits[0, mask_token_indices]

    pred_tokens = []
    for i in range(len(mask_token_indices)):
        topk_logits, topk_indices = logits_at_masks[i].topk(k=3, dim=-1)
        probabilities = torch.nn.functional.softmax(topk_logits, dim=-1)
        predicted_index = torch.distributions.categorical.Categorical(probabilities).sample()
        predicted_token_id = topk_indices[predicted_index].item()
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=True)

        pred_tokens.append('G' if predicted_token == '' else predicted_token)

    generated_sequence = masked_sequence
    for token in pred_tokens:
        generated_sequence = generated_sequence.replace("<mask>", token, 1)

    return generated_sequence, mask_token_indices


def calculate_perplexity(model: Diffusion, tokenizer, generated_sequence, mask_token_indices):
    total_loss = 0.0
    tensor_input = tokenizer.encode(generated_sequence, return_tensors='pt').to(model.device)

    for i in mask_token_indices:
        masked_input = tensor_input.clone()
        masked_input[0, i] = tokenizer.mask_token_id
    
        labels = torch.full(tensor_input.shape, -100).to(model.device)
        labels[0, i] = tensor_input[0, i]

        with torch.no_grad():
            outputs = model.forward(masked_input, sigma=torch.tensor([1], device=model.device), attention_mask=None)
            loss = F.cross_entropy(outputs.squeeze(), labels.squeeze())
            total_loss += loss.item()
    
    num_mask_tokens = len(mask_token_indices)
    if num_mask_tokens == 0:
        perplexity = 10000
    else:
        avg_loss = total_loss / num_mask_tokens
        perplexity = math.exp(avg_loss)

    return perplexity

@hydra.main(version_base=None, config_path='configs', config_name='config')
def mdlm_motif_benchmark(config):
    path = "/workspace/vnt/MeMDLM"
    
    test_sequences = pd.read_csv(path + "/data/membrane/test.csv")['Sequence'].tolist()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # mlm_model = Diffusion.load_from_checkpoint(config.eval.checkpoint_path, config=config, tokenizer=tokenizer)
    mlm_model = Diffusion(config, tokenizer=tokenizer)
    mlm_model.eval()
    # esm_model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    print("load models...")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mlm_model.to(device)
    esm_model.to(device)

    for generate_case in ['uppercase', 'lowercase']:
        case_results = []
        for original_sequence in tqdm(test_sequences, desc=f"scaffolding ({generate_case}): "):

            generated_sequence, mask_token_idx = generate_scaffold_mdlm(original_sequence, generate_case, tokenizer, mlm_model)
            perplexity = calculate_perplexity(mlm_model, tokenizer, generated_sequence, mask_token_idx)
            cos_sim = calculate_cosine_sim(original_sequence, generated_sequence, tokenizer, esm_model, device)
            hamming_distance = calculate_hamming_dist(original_sequence, generated_sequence)
        
            case_results.append([original_sequence, generated_sequence, perplexity, cos_sim, hamming_distance])

            # print(case_results)
            sys.stdout.flush()

        df = pd.DataFrame(case_results, columns=['Original Sequence', 'Generated Sequence', 'Perplexity', 'Cosine Similarity', 'Hamming Distance'])
        df.to_csv(path + f'/benchmarks/MLM/mlm_{generate_case}_results.csv', index=False)
        

    
if __name__ == "__main__":
    mdlm_motif_benchmark()
    

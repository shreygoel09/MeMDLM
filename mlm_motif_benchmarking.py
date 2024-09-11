import torch
import config
import sys
import pandas as pd
from mlm_generate_utils import generate_scaffold, calculate_perplexity, calculate_cosine_sim, calculate_hamming_dist
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer

def motif_benchmarking():
    path = "/workspace/sg666/MDpLM"
    
    test_sequences = pd.read_csv(path + "/data/membrane/test.csv")['Sequence'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(config.CKPT_DIR + "/best_model_epoch")
    mlm_model = AutoModelForMaskedLM.from_pretrained(config.CKPT_DIR + "/best_model_epoch")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    mlm_model.to(device)
    esm_model.to(device)

    for generate_case in ['uppercase', 'lowercase']:
        case_results = []
        for original_sequence in test_sequences:
            generated_sequence, mask_token_idx = generate_scaffold(original_sequence, generate_case, tokenizer, mlm_model)
            perplexity = calculate_perplexity(mlm_model, tokenizer, generated_sequence, mask_token_idx)
            cos_sim = calculate_cosine_sim(original_sequence, generated_sequence, tokenizer, esm_model, device)
            hamming_distance = calculate_hamming_dist(original_sequence, generated_sequence)
        
            case_results.append([original_sequence, generated_sequence, perplexity, cos_sim, hamming_distance])

            print(case_results)
            sys.stdout.flush()

        df = pd.DataFrame(case_results, columns=['Original Sequence', 'Generated Sequence', 'Perplexity', 'Cosine Similarity', 'Hamming Distance'])
        df.to_csv(path + f'/benchmarks/MLM/mlm_{generate_case}_results.csv', index=False)


if __name__ == "__main__":
    motif_benchmarking()
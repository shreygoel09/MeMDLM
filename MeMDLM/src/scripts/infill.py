import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from MeMDLM.src.scripts.generate_utils import calculate_perplexity
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.guidance import SolubilityGuider


class Infiller:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
        self.mdlm = Diffusion.load_from_checkpoint(self.cfg.eval.checkpoint_path,
                                                   config=self.cfg,
                                                   tokenizer=self.tokenizer).eval().to(self.device)
        self.esm_model = AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').to(self.device)
        self.guider = SolubilityGuider(self.cfg, self.device, self.mdlm)
    
    def _encode_sequence(self, sequence):
        return self.tokenizer(sequence, return_tensors='pt').to(self.mdlm.device)
    
    def _decode_sequence(self, token_ids):
         # Remove bos/eos tokens & spaces between residues
        return self.tokenizer.decode(token_ids.squeeze())[5:-5].replace(" ", "")

    @torch.no_grad()
    def generate_sequence(self, input_ids):
        generated_ids = self.mdlm._sample(x_input=input_ids)
        generated_sequence = self._decode_sequence(generated_ids)
        return generated_sequence, generated_ids

    def ppl(self, sequence, mask_idxs: list):
        return calculate_perplexity(self.esm_model, self.tokenizer, sequence, mask_idxs)

    def infill(self, sequence: str, mask_indices: list, conserved_idx: list, infill: bool, guidance: bool):
        tokens = self._encode_sequence(sequence.upper())
        og_tokens = tokens['input_ids'].clone()
        tokens['input_ids'] = tokens['input_ids'][:, 1:-1] # 1 x 231
        tokens['attention_mask'] = tokens['attention_mask'][:, 1:-1]

        # Replace normal input id with mask token id
        id_mask = torch.tensor(mask_indices, dtype=torch.long)
        tokens['input_ids'][0, id_mask] = self.tokenizer.mask_token_id

        #infilled_seq, _ = self.generate_sequence(tokens)
        #infilled_ppl = self.ppl(infilled_seq, mask_indices)

        if guidance:
            _, optim_seq =self.guider.sample_guidance(x_0=tokens, og_tokens=og_tokens,
                                                      guidance=True, infill=infill,
                                                      conserved_indices=conserved_idx)
            print(optim_seq)
            print(sequence.upper())
            optim_ppl = self.ppl(optim_seq, mask_indices)
        
        # print(conserved_idx)
        # bads = []
        # for i in conserved_idx:
        #     if sequence[i] != optim_seq[i]:
        #         bads.append(i)
        # print(f'mismatches for seq {sequence}')
        # print(bads, '\n')

        return optim_seq, optim_ppl



# @hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
# def main(config):
#     tokenizer = 
#     device = torch.device(f"cuda:{2}" if torch.cuda.is_available() else 'cpu')

#     mdlm = 

#     esm_model = 
#     esm_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')

#     guidance =  # Pass Diffusion model to prevent loading twice
#     generator = Generator(tokenizer, mdlm, esm_model, device)

#     print("loaded models...")

#     # Get 100 random sequence lengths to generate
#     sequences = read_fasta(data_path)

#     generation_results = []
#     for seq_id in tqdm(sequences, desc=f"Infilling and optimizing sequences: "):
#         sequence = sequences[seq_id]
        
#         seq_res = []

#         # Unconditionally infill insoluble residues
#         insoluble_seq, insoluble_ppl = generator.infill(sequence, generate_type='uppercase')

#         # Unconditionally infill soluble residues
#         soluble_seq, soluble_ppl = generator.infill(sequence, generate_type='lowercase')

#         # Conduct guidance on insoluble residues
#         insoluble_masked_seq = mask_for_scaffold(sequence, generate_type='uppercase')
#         insoluble_tokens = tokenizer(insoluble_masked_seq, return_tensors='pt').to(device)
#         optim_tokens, optim_seq = guidance.sample_guidance(x_0=insoluble_tokens, guidance=True)
#         optim_ppl = calculate_perplexity(
#             esm_model,
#             tokenizer,
#             optim_seq,
#             [i for i, char in enumerate(sequence) if char.isupper()]
#         )

#         # Unconditionally generate a sequence and calculate its perplexity and solubility
#         # og_seq, og_tokens = generate_sequence(masked_seq, tokenizer, mdlm)
#         # og_solubility_preds = guidance.embed_and_predict_solubility(
#         #     input_ids=og_tokens['input_ids'],
#         #     attention_masks=og_tokens['attention_mask']
#         # )['solubility_predictions']
#         # og_solubility = round(guidance.compute_frac_soluble(og_tokens["input_ids"], og_solubility_preds), 4)
#         # og_ppl = round(mdlm.compute_masked_perplexity([og_seq], masked_seq), 4)

#         # seq_res.append(og_seq)
#         # seq_res.append(og_ppl)
#         # seq_res.append(og_solubility)
#         # _print_og(og_solubility, og_ppl, og_seq)

#         # Do guidance steps if original sequence is insoluble
#         # if og_solubility < config.value.guidance.sequence_thresh:
#         #     try:
#         #         optim_tokens, optim_seq = guidance.sample_guidance(x_0=og_tokens, guidance=True)
#         #         optim_solubility_preds = guidance.embed_and_predict_solubility(
#         #             input_ids=optim_tokens,
#         #             attention_masks=torch.ones_like(optim_tokens)
#         #         )["solubility_predictions"]
#         #         optim_solubility = round(guidance.compute_frac_soluble(optim_tokens, optim_solubility_preds), 4)
                
#         #         try:
#         #             optim_ppl = round(mdlm.compute_masked_perplexity([optim_seq], masked_seq), 4)
#         #         except:
#         #             optim_seq = ""
#         #             optim_solubility = 0
#         #             optim_ppl = 0

#         #         seq_res.append(optim_seq)
#         #         seq_res.append(optim_ppl)
#         #         seq_res.append(optim_solubility)
#         #         _print_optim(optim_solubility, optim_ppl, optim_seq)
#         #         print("\n")
#         #     except Exception as e:
#         #         print("ERROR: ", e)
#         #         seq_res.append("")
#         #         seq_res.append(0)
#         #         seq_res.append(0)
#         #         print("No optimization required.")
#         #         print("\n")
#         #         continue

#         # else:
#         #     seq_res.append("")
#         #     seq_res.append(0)
#         #     seq_res.append(0)
#         #     print("No optimization required.")
#         #     print("\n")
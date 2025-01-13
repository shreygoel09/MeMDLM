import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from main import SolubilityClassifier
from MeMDLM.src.diffusion.diffusion import Diffusion


class SolubilityGuider:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)

        self.diffusion = Diffusion(self.config, self.tokenizer)
        self.diffusion = self.diffusion.load_from_checkpoint(self.config.eval.checkpoint_path,
                                                             self.tokenizer,
                                                             config=config).eval()
        self.memdlm = AutoModel.from_pretrained(self.config.value.training.pretrained_model).eval()
        self.medlm_lm = AutoModelForMaskedLM.from_pretrained(self.config.value.training.pretrained_model).eval()

        self.classifier_model = SolubilityClassifier(config).eval()
        ckpt_path = os.path.join(config.value.training.ckpt_path, "best_model.ckpt")
        state_dict = self.classifier_model._get_state_dict(ckpt_path)
        self.classifier_model.load_state_dict(state_dict)
        
        self.eps = config.value.guidance.epsilon
        self.temperature = config.value.guidance.temperature
        self.topk = config.value.guidance.topk

        self.residue_thresh = config.value.guidance.residue_thresh
        self.sequence_density = config.value.guidance.sequence_thresh
    
    def tokenize_sequence(self, sequence):
        """Helper method to tokenize a sequence"""
        tokens = self.tokenizer(sequence, return_tensors='pt')
        return tokens['input_ids'], tokens['attention_mask']
    
    def embed_and_predict_solubility(self, input_ids, attention_masks):
        """Obtain sequence embeddings and per-residue solubility predictions"""
        # Get sequence embeddings using diffusion model hidden states
        with torch.no_grad():
            outputs = self.memdlm(input_ids=input_ids, attention_mask=attention_masks)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0)

        # Get per-residue solubility predictions (of shape [seq_len]) from trained classifier model
        solubility_preds = self.classifier_model(embeds=sequence_embeddings, mask=attention_masks)
        
        return {
            "solubility_predictions": solubility_preds.requires_grad_(True), # Enable gradients for backprop
            "sequence_embeddings": sequence_embeddings 
        }


    def compute_saliency(self, input_ids, attention_masks):
        """
        Compute a saliency map using gradients as defined in LaMBO-2 (https://arxiv.org/pdf/2305.20009)
        Args:
            sequence (str): generated, unoptimized protein sequence
        Returns:
            saliency (torch.Tensor): normalized saliency scores [seq_len]
            input_ids (torch.Tensor): squeezed token IDS [seq_len]
        """
        # get sequence embeddings and per-residue predictions
        outs = self.embed_and_predict_solubility(input_ids, attention_masks)

        # Compute gradients of value function wrt hidden states
        gradients = torch.autograd.grad(
            outputs=outs['solubility_predictions'],
            inputs=outs['sequence_embeddings'],
            create_graph=True,
        )[0].squeeze(0)  # shape [seq_length x hidden_dim=640] since we optimize 1 sequence at a time

        # Creating a saliency map (Eq.5 in LaMBO-2 paper)
        saliency = gradients.abs().sum(dim=-1)  # summation across hidden dim. Abs value for directionality only
        saliency = saliency.pow(1.0 / self.temperature) # temperature scaling and numerical stability
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) # normalization
        saliency = torch.clamp(saliency, min=self.eps) # numerical stability

        return saliency

    def determine_edit_positions(self, saliency_map, solubility_preds):
        """"Determine edit positions for a generated sequence.
        We define low-value residues as those with low solubility scores
        that do not contribute to the saliency of the sequence.
        Args:
            - sequence (str): protein sequence
        Returns:
            - edit_positions (torch.Tensor): token IDs with low-value residues remasked

        """
        # Selection procedure to determine low-value residues
        combined_scores = saliency_map * solubility_preds # interpolate between saliency and solubility pred
        probabilities = F.softmax(combined_scores, dim=0) # probability distribution
        probabilities[(solubility_preds == 1.0)] = 0.0 # zero out selection chance for soluble residues
        topk_edit_probs, topk_edit_pos = torch.topk(probabilities, self.topk) # select edit positions

        return probabilities, topk_edit_probs
    
    def mask_sequence(self, input_ids, solubility_probs, topk_edit_probs):
        """Mask out low-vaue residues"""
        return torch.where(
            solubility_probs > topk_edit_probs,
            input_ids,
            torch.full_like(input_ids, self.tokenizer.mask_token_id)
        )

    def compute_frac_soluble(self, input_ids, solubility_preds):
        """Helper method to compute density of soluble residues in a sequence"""
        soluble_mask = torch.gt(solubility_preds, self.residue_thresh)
        num_soluble = torch.count_nonzero(solubility_preds, soluble_mask).sum().item()
        frac_soluble = (num_soluble / len(input_ids))
        return frac_soluble
    

    def update_logits(self, p_ht, solubility_preds):
        """
        Shift logits distribution towards only soluble residues.
        Implementation of SGLD update step (Eq. 4 in LaMBO-2 paper).
        Given logits ht, we apply update: ht_next <-- ht - grad(loss(ht)) + stoch
        Args:
            - p_ht (torch.Tensor): diffusion model logits matrix [seq_len x vocab_size]
            - solubility_preds (torch.Tensor): per-residue solubility predictions
        Returns:
            - updated_logits (torch.Tensor): updated diffusion logits
        """

        neta = self.config.value.guidance.step_size
        eps = torch.randn_like(p_ht) # eps ~ N(0, 1)

        # ~p_theta(w_hat | h_t)
        p_ht_prime = torch.pow(p_ht, solubility_preds.unsqueeze(-1).expand_as(p_ht)) # Scale logits by solubility predictions
        p_ht_prime = p_ht / p_ht_prime.sum(dim=-1, keep_dim=True) # valid probability distribution over vocabulary

        # neta * KL(p_theta(w_hat | ht_prime) || p_theta(w_hat | h_t))
        kl_div = F.kl_div(p_ht_prime.log(), p_ht, reduction='none') # no reduction for per-residue computation

        updated_seq = self._decode(p_ht_prime)
        updated_token_ids = self.tokenizer(updated_seq, return_tensors='pt')['input_ids']
        
        v_theta_ht_prime = self.embed_and_predict_solubility(
            input_ids=updated_token_ids,
            attention_masks=torch.ones_like(updated_token_ids)
        )['solubility_predictions']

        ht_prime_grad = torch.autograd.grad(
            outputs= (kl_div - v_theta_ht_prime).requires_grad_(True), # loss term in Eq 4.
            inputs=p_ht_prime, # w.r.t updated logits
            create_graph=True,
            retain_graph=True
        )[0]

        # Stochastic term for SGLD
        stoch_term = torch.sqrt(2 * neta * self.temperature) * eps

        # apply updates
        ht_prime_next = p_ht_prime - (neta * ht_prime_grad) + stoch_term
        return ht_prime_next
    
    def generate_and_optimize(self, ids, logits):
        """Combine the diffusion sampling and logits update"""
        with torch.no_grad():
            logits = self.diffusion._sample(x_input=ids)
        updated_logits = self.update_logits(logits)
        generated_sequence = self._decode(updated_logits)
        return generated_sequence
    
    def optimized_sampling(self, sequence):
        """Main entry point to optimize a generated sequence."""
        # Tokenize sequence
        input_ids, attention_masks = self.tokenize_sequence(sequence)
        squeezed_ids, squeezed_masks = input_ids.squeeze(0), attention_masks.squeeze(0)

        outs = self.embed_and_predict_solubility(input_ids, attention_masks)
        solubility_preds = outs['solubility_predictions']
        og_embeddings = outs['sequence_embeddings']

        frac_soluble = self.compute_frac_soluble(squeezed_ids, solubility_preds)

        # Iterate until we hit a certain density of soluble residues
        while (frac_soluble < self.sequence_density):
            # Compute saliency map and edit positions
            saliency_map = self.compute_saliency(squeezed_ids, squeezed_masks)
            solubility_preds, top_edit_probs = self.determine_edit_positions(saliency_map, solubility_preds)

            # Mask low-value residues and get updated logits
            remasked_seq_ids = self.mask_sequence(squeezed_ids, solubility_preds, top_edit_probs)
            p_ht = self.medlm_lm(input_ids=remasked_seq_ids, attention_mask=attention_masks)

            # Generate and optimize the new sequence
            optimized_sequence = self.generate_and_optimize(ids=remasked_seq_ids, logits=p_ht)

            # Recompute solubility density of optimized
            input_ids, attention_masks = self.tokenize_sequence(optimized_sequence)
            solubility_preds = self.embed_and_predict_solubility(input_ids, attention_masks)['solubility_predictions']
            frac_soluble = self.compute_frac_soluble(input_ids.squeeze(0), solubility_preds)

        return optimized_sequence

    
    def _decode(self, logits):
        """Helper method to decode a sequence from logits matrix."""
        return self.tokenizer.decode(logits.squeeze())[5:-5].replace(" ", "")
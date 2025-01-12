import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from MeMDLM.src.diffusion.diffusion import Diffusion
from main import SolubilityClassifier


class SolubilityGuider:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)

        self.diffusion = Diffusion(self.config, self.tokenizer)
        self.diffusion = self.diffusion.load_from_checkpoint(self.config.eval.checkpoint_path,
                                                             self.tokenizer,
                                                             config=config).eval()
        self.memdlm = AutoModel.from_pretrained(self.config.value.training.pretrained_model)

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
        tokens = self.tokenizer(sequence, return_tensors='pt')
        return tokens['input_ids'], tokens['attention_mask']
    
    def embed_and_predict_solubility(self, input_ids, attention_masks):
        # Get sequence embeddings using diffusion model hidden states
        with torch.no_grad():
            outputs = self.mdlm_model(input_ids=input_ids, attention_mask=attention_masks)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0)

        # Get per-residue solubility predictions (of shape [seq_len]) from trained classifier model
        solubility_preds = self.classifier_model(embeds=sequence_embeddings,
                                           mask=attention_masks)
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
        return torch.where(
            solubility_probs > topk_edit_probs,
            input_ids,
            torch.full_like(input_ids, self.tokenizer.mask_token_id)
        )

    def compute_frac_soluble(self, input_ids, solubility_preds):
        soluble_mask = torch.gt(solubility_preds, self.residue_thresh)
        num_soluble = torch.count_nonzero(solubility_preds, soluble_mask).sum().item()
        frac_soluble = (num_soluble / len(input_ids))
        return frac_soluble
    

    def update_logits(self, p_ht, solubility_preds):
        """
        Core logic to shift distribution towards soluble residues.
        Eq. 4 in LaMBO-2 paper.
        Args:
            - ph_t (torch.Tensor): diffusion model logits matrix [seq_len x vocab_size]
            - solubility_preds (torch.Tensor): per-residue solubility predictions
        Returns:
            - updated_logits (torch.Tensor): updated diffusion logits
        """
        solubility_preds = solubility_preds.unsqueeze(-1).expand_as(p_ht)

        neta = self.config.value.guidance.step_size
        eps = torch.randn_like(p_ht) # eps ~ N(0, 1)

        # Scale logits by solubility predictions: ~p_theta(w_hat | h_t)
        p_ht_prime = torch.pow(p_ht, solubility_preds)
        p_ht_prime = p_ht / p_ht_prime.sum(dim=-1, keep_dim=True)

        # update logits with KL Divergence
        kl_div = F.kl_div(p_ht_prime.log(), p_ht, reduction='batchmean')  # KL(p_theta(w_hat | ht_prime) || p_theta(w_hat | h_t))

        kl_grad = torch.autograd.grad(
            outputs=kl_div,
            inputs=p_ht,
            grad_outputs=torch.ones_like(kl_div),
            retain_graph=True,
            create_graph=True,
        )[0]

        v_theta_grad = torch.autograd.grad(
            outputs=solubility_preds,
            inputs=p_ht,
            grad_outputs=torch.ones_like(solubility_preds),
            create_graph=True,
            retain_graph=True,
        )[0]

        logits_update = (kl_grad - v_theta_grad) * neta

        # stoachasticity term
        stoch_term = math.sqrt(2 * neta * self.temperature) * eps

        # apply updates
        updated_logits = p_ht_prime - logits_update + stoch_term

        return updated_logits
    
    def generate_and_optimize(self, input_ids):
        # Generate an initial protein sequence
        with torch.no_grad():
            logits = self.diffusion._sample(x_input=input_ids)
        logits = self.update_logits(logits)
        generated_sequence = self.tokenizer.decode(logits.squeeze())[5:-5].replace(" ", "")
        return generated_sequence
    
    def optimized_sampling(self, sequence):
        """
        while (# soluble residues < 0.80):
            compute saliency and mask out bad residues
            tokenize sequence
            update logits
            re-generate sequence
            classify residues
            
        """
        # Tokenize sequence
        input_ids, attention_masks = self.tokenize_sequence(sequence)
        squeezed_ids, squeezed_masks = input_ids.squeeze(0), attention_masks.squeeze(0)

        outs = self.embed_and_predict_solubility(input_ids, attention_masks)
        solubility_preds = outs['solubility_predictions']
        frac_soluble = self.compute_frac_soluble(squeezed_ids, solubility_preds)

        # Iterate until we hit a certain density of soluble residues
        while (frac_soluble < self.sequence_density):
            # Compute saliency map and edit positions
            saliency_map = self.compute_saliency(squeezed_ids, squeezed_masks)
            solubility_preds, top_edit_probs = self.determine_edit_positions(saliency_map, solubility_preds)

            # Mask low-value residues
            remasked_seq_ids = self.mask_sequence(squeezed_ids, solubility_preds, top_edit_probs)

            # Optimize and generate the new sequence
            optimized_sequence = self.generate_and_optimize(remasked_seq_ids)

            # Recompute solubility density of sequence
            input_ids, attention_masks = self.tokenize_sequence(optimized_sequence)
            solubility_preds = self.embed_and_predict_solubility(input_ids, attention_masks)['solubility_predictions']
            frac_soluble = self.compute_frac_soluble(input_ids.squeeze(0), solubility_preds)

        return optimized_sequence
import os
import math
import torch
import torch.nn.functional as F
import torch.autograd as GRD
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from MeMDLM.src.guidance.main import SolubilityClassifier
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.utils import _print


class SolubilityGuider:
    def __init__(self, config, device, mdlm):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)

        self.diffusion = mdlm
        self.memdlm = AutoModel.from_pretrained(config.value.training.pretrained_model).eval().to(self.device)
        self.medlm_lm = AutoModelForMaskedLM.from_pretrained(self.config.value.training.pretrained_model).eval().to(self.device)

        #ckpt_path = os.path.join(config.value.training.ckpt_path, "best_model.ckpt")
        ckpt_path = "/workspace/sg666/MeMDLM/MeMDLM/checkpoints/classifier/steps50k_lr3e-4_bsz16_heads2_drpt0.5_layers4_mask0.50"
        self.classifier_model = SolubilityClassifier(config, sampling=True).eval().to(self.device)
        state_dict = self.classifier_model._get_state_dict(ckpt_path + "/best_model.ckpt")
        self.classifier_model.load_state_dict(state_dict)
        
        self.eps = config.value.guidance.epsilon
        self.topk = config.value.guidance.topk
        self.temperature = config.value.guidance.temperature
        self.residue_thresh = config.value.guidance.residue_thresh
        self.sequence_density = config.value.guidance.sequence_thresh

    
    def tokenize_sequence(self, sequence):
        """Helper method to tokenize a sequence"""
        tokens = self.tokenizer(sequence, return_tensors='pt').to(self.device)
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

    def _decode(self, logits):
        """Helper method to decode a sequence from logits."""
        return 
    
    def embed_and_predict_solubility(self, input_ids, attention_masks):
        """Obtain sequence embeddings and per-residue solubility predictions"""
        # Get sequence embeddings using diffusion model hidden states
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_masks.dim() == 1:
            attention_masks = attention_masks.unsqueeze(0)

        input_ids, attention_masks = input_ids.to(self.device), attention_masks.to(self.device)
        
        with torch.no_grad():
            outputs = self.memdlm(input_ids=input_ids, attention_mask=attention_masks)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0)

        # Get per-residue solubility predictions (of shape [seq_len]) from trained classifier model
        batch = {"embeddings": sequence_embeddings, "attention_mask": attention_masks}
        solubility_preds = self.classifier_model(batch)
        
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
        embeddings = outs['sequence_embeddings']
        preds = outs['solubility_predictions']

        def jacobian_fn(embeddings):
            batch = {"embeddings": embeddings, "attention_mask": attention_masks}
            return self.classifier_model.forward(batch)

        jacobian = GRD.functional.jacobian(jacobian_fn, embeddings, create_graph=True)

        # Creating a saliency map (Eq.5 in LaMBO-2 paper)
        saliency = jacobian.abs().sum(dim=-1)  # summation across hidden dim. Abs value for directionality only
        saliency = saliency.pow(1.0 / self.temperature)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        saliency = saliency.clamp(min=self.eps).to(self.device).diag()
        return saliency

        # # Compute gradients of value function wrt hidden states
        # gradients = torch.autograd.grad(
        #     outputs=outs['solubility_predictions'],
        #     inputs=outs['sequence_embeddings'],
        #     grad_outputs=torch.ones_like(outs['solubility_predictions']),
        #     create_graph=True,
        # )[0].squeeze(0)  # shape [seq_length x hidden_dim=640] since we optimize 1 sequence at a time


    def determine_edit_positions(self, saliency_map, solubility_preds):
        """"Determine edit positions for a generated sequence.
        We define low-value residues as those with low solubility scores
        that do not contribute to the saliency of the sequence.
        Args:
            - sequence (str): protein sequence
        Returns:
            - edit_positions (torch.Tensor): token IDs with low-value residues remasked

        """
        # # Selection procedure to determine low-value residues
        # combined_scores = saliency_map * solubility_preds # interpolate between saliency and solubility
        # probabilities = F.softmax(combined_scores, dim=0) # aggressive probability distribution w/ softmax
        # probabilities[(solubility_preds == 1.0)] = 0.0 # zero out selection chance for soluble residues
        # #topk_edit_probs, topk_edit_pos = torch.topk(probabilities, self.topk) # select edit positions

        # return probabilities.to(self.device) #, topk_edit_probs
        probabilities = saliency_map / saliency_map.sum()
        probabilities = probabilities.masked_fill(solubility_preds <= self.residue_thresh, 0.0)
        return probabilities
    
    def mask_sequence(self, input_ids, interpolated_solubility):
        """Mask out low-vaue residues"""
        return torch.where(
            interpolated_solubility > 0.5,
            input_ids.to(self.device),
            torch.full_like(input_ids, self.tokenizer.mask_token_id).to(self.device)
        ).unsqueeze(0)

    def compute_frac_soluble(self, input_ids, solubility_preds):
        """Helper method to compute density of soluble residues in a sequence"""
        num_soluble = (solubility_preds > self.residue_thresh).sum().item()
        frac_soluble = (num_soluble / solubility_preds.numel())
        return frac_soluble

    def update_logits(self, p_ht_prime, p_ht, updated_ids, first_iter: bool):
        """
        Shift logits distribution towards only soluble residues.
        Implementation of SGLD update step (Eq. 4 in LaMBO-2 paper).
        Given logits p_ht, we apply update: ht_next <-- ht - grad(loss(ht)) + stoch
        Args:
            - p_ht (torch.Tensor): diffusion model logits matrix [seq_len x vocab_size]
            - solubility_preds (torch.Tensor): per-residue solubility predictions
        Returns:
            - updated_logits (torch.Tensor): updated diffusion logits
        """

        neta = self.config.value.guidance.step_size
        lamb = self.config.value.guidance.reg_strength
        eps = torch.normal(mean=0.0, std=1.0, size=(1,)).clamp(-3.0, 3.0).item() # eps ~ N(0, 1)

        # ~p_theta(w_hat | h_t)
        # p_ht_prime = torch.pow(p_ht, solubility_preds.unsqueeze(-1).expand_as(p_ht)) # Scale logits by solubility predictions
        # p_ht_prime = p_ht / p_ht_prime.sum(dim=-1, keepdim=True) # valid probability distribution over vocabulary

        # First round of iteration has same logits
        if first_iter:
            p_ht_prime = p_ht

        # Force valid probability distributions over vocab
        p_ht_prime, p_ht = F.log_softmax(p_ht_prime, dim=-1), F.softmax(p_ht, dim=-1)

        v_ht_prime = self.embed_and_predict_solubility(
            input_ids=updated_ids,
            attention_masks=torch.ones_like(updated_ids)
        )['solubility_predictions']

        def loss_fn(p_ht_prime):
            """Jacobian helper function to implement Eq. 4"""
            kl_div = F.kl_div(p_ht_prime, p_ht, reduction='none').sum(dim=-1) # Aggregate KL term over the vocab
            return (lamb * kl_div - v_ht_prime).requires_grad_(True) # loss term
        
        ht_prime_grad = GRD.functional.jacobian(loss_fn, p_ht_prime, create_graph=True)

        # Remove bsz dimension and aggregate over seq_len
        ht_prime_grad = ht_prime_grad.squeeze(0).squeeze(1).sum(dim=1)
        p_ht_prime = p_ht_prime.squeeze(0)

        # Stochastic term
        stoch_term = math.sqrt(2 * neta * self.temperature) * eps

        # SGLD update
        p_ht_prime_next = p_ht_prime - (neta * ht_prime_grad) + stoch_term
        return p_ht_prime_next
    
    
    def optimized_sampling(self, og_seq, og_preds, og_solubility):
        """Main entry point to optimize a generated sequence."""
        # if og_solubility >= self.sequence_density:
        #     return og_seq, og_solubility
    
        squeezed_ids, squeezed_masks = self.tokenize_sequence(og_seq)
        ids, masks = squeezed_ids.unsqueeze(0), squeezed_masks.unsqueeze(0)
        og_logits = self.medlm_lm(ids, masks).logits

        # Init the update variables
        solubility_preds = og_preds
        frac_soluble = og_solubility
        optimized_sequence = og_seq
        iter=1

        print(f"OG Sequence: {og_seq}")
        print(f'OG Frac Soluble: {frac_soluble}')
        print(f'OG Logits: {og_logits}')
        
        # Continuous optimization until solubility density threshold is reached
        while (frac_soluble < self.sequence_density):

            # Compute saliency map and edit positions
            saliency_map = self.compute_saliency(squeezed_ids, squeezed_masks)
            edit_positions = self.determine_edit_positions(saliency_map, solubility_preds)

            print(f'saliency map: {saliency_map}')
            print(f'init preds: {solubility_preds}')
            print(f'edit pos: {edit_positions}')

            # Mask low-value residues and get updated logits
            remasked_seq_ids = self.mask_sequence(squeezed_ids, edit_positions)
            updated_logits = self.diffusion.get_logits(
                x=remasked_seq_ids,
                attention_mask=torch.ones_like(remasked_seq_ids)
            )

            # Optimize and generate the new sequence
            updated_logits = self.update_logits(
                p_ht_prime=updated_logits,
                p_ht=og_logits,
                updated_ids=remasked_seq_ids,
                first_iter=(iter == 1)
            )

            optimized_sequence = self.tokenizer.decode(updated_logits.argmax(dim=-1))[5:-5].replace(" ", "")

            # Recompute solubility density of optimized sequence
            squeezed_ids, squeezed_masks = self.tokenize_sequence(optimized_sequence)
            solubility_preds = self.embed_and_predict_solubility(
                input_ids=squeezed_ids.unsqueeze(0),
                attention_masks=squeezed_masks.unsqueeze(0)
            )['solubility_predictions']
            frac_soluble = self.compute_frac_soluble(squeezed_ids, solubility_preds)

            iter += 1

            print(f'optim seq: {optimized_sequence}')
            print(f'optim sol: {frac_soluble}')

        return optimized_sequence, frac_soluble
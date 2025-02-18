import os
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as GRD
from tqdm import tqdm, trange
from argparse import Namespace
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from MeMDLM.src.guidance.main import SolubilityClassifier
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.utils import _print


def _sample_categorical(categorical_probs):
  gumbel_norm = (1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


class SolubilityGuider:
    def __init__(self, config, device, mdlm):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)

        self.diffusion = mdlm
        self.memdlm = AutoModel.from_pretrained(config.value.training.pretrained_model).eval().to(self.device)
        self.medlm_lm = AutoModelForMaskedLM.from_pretrained(
            self.config.value.training.pretrained_model,
            output_hidden_states=True
        ).eval().to(self.device)

        ckpt_path = os.path.join(config.value.training.ckpt_path, "best_model.ckpt")
        self.classifier_model = SolubilityClassifier(config, sampling=True).eval().to(self.device)
        state_dict = self.classifier_model._get_state_dict(ckpt_path)
        self.classifier_model.load_state_dict(state_dict)
        
        self.eps = config.value.guidance.epsilon
        self.topk = config.value.guidance.topk
        self.temperature = config.value.guidance.temperature
        self.residue_thresh = config.value.guidance.residue_thresh
        self.sequence_density = config.value.guidance.sequence_thresh

    def mask_sequence(self, input_ids, interpolated_solubility):
        """Mask out low-vaue residues"""
        return torch.where(
            interpolated_solubility==0,
            input_ids.to(self.device),
            torch.full_like(input_ids, self.tokenizer.mask_token_id).to(self.device)
        ).unsqueeze(0)

    def compute_frac_soluble(self, input_ids, solubility_preds):
        """Calculatedensity of soluble residues in a sequence"""
        num_soluble = (solubility_preds > self.residue_thresh).sum().item()
        frac_soluble = (num_soluble / solubility_preds.numel())
        return frac_soluble

    def compute_logits_value_from_hidden(self, hidden_states, attention_mask):
        """Obtain logits and solubility predictions from hidden states"""
        batch = {"embeddings": hidden_states.squeeze(), "attention_mask": attention_mask.squeeze()}
        solubility_preds = self.classifier_model(batch)
        return self.medlm_lm.lm_head(hidden_states), solubility_preds

    def embed_and_predict_solubility(self, input_ids, attention_masks):
        """Get sequence embeddings and solubility predictions"""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_masks.dim() == 1:
            attention_masks = attention_masks.unsqueeze(0)

        input_ids, attention_masks = input_ids.to(self.device), attention_masks.to(self.device)
        
        with torch.no_grad():
            outputs = self.memdlm(input_ids=input_ids, attention_mask=attention_masks)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0)

        batch = {"embeddings": sequence_embeddings, "attention_mask": attention_masks}
        solubility_preds = self.classifier_model(batch)
        
        return {
            "solubility_predictions": solubility_preds.requires_grad_(True), # Enable gradients for backprop
            "sequence_embeddings": sequence_embeddings 
        }


    def compute_saliency(self, embeddings, attention_masks, saliency_grads):
        """
        Compute a saliency map using gradients as defined in LaMBO-2 (https://arxiv.org/pdf/2305.20009)
        """
        optim = torch.optim.SGD([saliency_grads], lr=1) # lr=1 to apply full effect of gradients
        
        batch = {"embeddings": (embeddings + saliency_grads).squeeze(), "attention_mask": attention_masks.squeeze()}
        out = self.classifier_model.forward(batch).sum() 
                
        out.backward(retain_graph=True)
        optim.step() # stores gradients in delta
        optim.zero_grad()
        
        # Creating the saliency map (Eq.5 in LaMBO-2 paper)
        saliency = saliency_grads.abs().sum(dim=-1)  # Summation across hidden dim. Abs value for mangitude only
        saliency = saliency.pow(1.0 / self.temperature).clamp(min=self.eps).to(self.device)
        return saliency.squeeze()
    

    def determine_edit_positions(self, saliency_map, solubility_preds):
        """
        Create a one-hot mask that indicates the top-k low-value residue positions.
        We defind low-value positions as those with high saliency scores and
        thus a high edit probability.
        """
        probabilities = saliency_map.masked_fill(solubility_preds >= self.residue_thresh, 0.0) # exclude already soluble tokens
        probabilities = probabilities / probabilities.sum()
        nonzero = probabilities.count_nonzero().item()
        _, topk_edit_pos = torch.topk(probabilities, min(self.topk, nonzero))
        mask = torch.zeros_like(probabilities).scatter(0, topk_edit_pos, torch.ones_like(probabilities))
        return mask.unsqueeze(-1)

    def update_logits(self, og_hidden, og_logits, hidden_state_mask, attention_mask, optimizer, delta):
        """
        Shift logits distribution towards only soluble residues by applying
        the explore-exploit loss.
        """
        lamb = self.config.value.guidance.reg_strength
        
        h_current = og_hidden + hidden_state_mask * delta
        new_logits, v_ht_prime = self.compute_logits_value_from_hidden(h_current, attention_mask)
        
        # Summing across dim instead of mean (this is used in the official implementation)
        loss = lamb * F.kl_div(new_logits, og_logits) - v_ht_prime.sum()
        
        # Gradient parameter in Langevin dynamics update (Eq. 4) wrt hidden states
        loss.backward(retain_graph=True)
        optimizer.step() # Populate delta with the gradient of the loss function * step size
        optimizer.zero_grad()
        
        return delta
    
    def optimized_sampling(self, og_logits, og_hidden, attention_mask, n_steps, plot_saliency=None):
        """Main entry point to optimize a generated sequence."""
        neta = self.config.value.guidance.step_size

        # Calculate initial solubility predictions
        batch = {"embeddings": og_hidden.squeeze(), "attention_mask": attention_mask.squeeze()}
        solubility_preds = self.classifier_model(batch)
        
        delta_saliency = nn.Parameter(torch.zeros_like(og_hidden), requires_grad=True)
        delta = nn.Parameter(torch.zeros_like(og_hidden), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=neta)
        optimizer.zero_grad()
        
        # Continuous optimization until solubility density threshold is reached
        with torch.enable_grad():
            for n in range(n_steps):

                # Compute saliency map and edit positions using updated hidden states
                saliency_map = self.compute_saliency(og_hidden + delta.data, attention_mask, delta_saliency)
                if plot_saliency:
                    if n == n_steps-1:
                        return saliency_map
                mask = self.determine_edit_positions(saliency_map, solubility_preds)

                # Optimize and generate the new sequence
                delta = self.update_logits(
                    og_hidden=og_hidden,
                    og_logits=og_logits,
                    hidden_state_mask=mask,
                    attention_mask=attention_mask,
                    optimizer=optimizer,
                    delta=delta
                )

        # Final hidden state update
        h_new = og_hidden + delta.data 
        new_logits, _new_sol = self.compute_logits_value_from_hidden(h_new, attention_mask)
        
        return new_logits
    
    def sample_guidance(self, x_0 = None, num_steps=None, eps=1e-5, bsz=1, guidance=True):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        
        # If input sequence is given (e.g. inpainting task)
        if x_0 is not None:
            x = x_0.input_ids.to(self.device)
            attention_mask = x_0.attention_mask.to(self.device)
        else: # Generate a prior of just mask tokens
            x = self.diffusion._sample_prior(bsz, self.config.model.length).to(self.device)
            attention_mask = torch.ones_like(x, device=self.device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        
        dt = (1 - eps) / num_steps
        
        for i in trange(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            logits, hidden_states = self.diffusion.backbone.forward_hidden(x, attention_mask)
            if guidance:
                logits = self.optimized_sampling(logits, hidden_states, attention_mask,
                                                 n_steps=self.config.value.guidance.n_steps)
            logits_wrapper = Namespace() # Workaround for SUBS as it accesses logits.logits
            logits_wrapper.logits = logits
            p_x0 = self.diffusion._subs_parameterization(logits=logits_wrapper, xt=x)
            
            # computing move chance categoricals
            move_chance_t = t[:, None, None]
            move_chance_s = (t - dt)[:, None, None]
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.diffusion.mask_index] = move_chance_s[:, :, 0]
            _x = _sample_categorical(q_xs) # Gumbel-max sampling of categoricals
            
            copy_flag = (x != self.diffusion.mask_index).to(x.dtype)
            x = (copy_flag * x + (1 - copy_flag) * _x).squeeze(0)

        generated_sequence = self.tokenizer.decode(x.squeeze())[5:-5].replace(" ", "")
        
        return x, generated_sequence
            
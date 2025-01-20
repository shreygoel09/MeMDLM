import os
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as GRD
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from MeMDLM.src.guidance.main import SolubilityClassifier
from MeMDLM.src.diffusion.diffusion import Diffusion
from MeMDLM.src.guidance.utils import _print
from tqdm import tqdm, trange
from argparse import Namespace


def _sample_categorical(categorical_probs):
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


class SolubilityGuider:
    def __init__(self, config, device, mdlm):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)

        self.diffusion = mdlm
        self.memdlm = AutoModel.from_pretrained(config.value.training.pretrained_model).eval().to(self.device)
        self.medlm_lm = AutoModelForMaskedLM.from_pretrained(self.config.value.training.pretrained_model, output_hidden_states=True).eval().to(self.device)

        #ckpt_path = os.path.join(config.value.training.ckpt_path, "best_model.ckpt")
        ckpt_path = "/workspace/sg666/MeMDLM/MeMDLM/checkpoints/classifier/steps50k_lr3e-5_bsz16_heads2_drpt0.5_layers4_mask0.50"
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


    def compute_saliency(self, embeddings, attention_masks):
        """
        Compute a saliency map using gradients as defined in LaMBO-2 (https://arxiv.org/pdf/2305.20009)
        Args:
            sequence (str): generated, unoptimized protein sequence
        Returns:
            saliency (torch.Tensor): normalized saliency scores [seq_len]
            input_ids (torch.Tensor): squeezed token IDS [seq_len]
        """
        # get sequence embeddings and per-residue predictions
        
        grads = nn.Parameter(torch.zeros_like(embeddings), requires_grad=True)
        optim = torch.optim.SGD([grads], lr=1) # 1x the gradient value = gradient value itself i think
        
        batch = {"embeddings": (embeddings + grads).squeeze(), "attention_mask": attention_masks.squeeze()}
        out = self.classifier_model.forward(batch).sum() # note: sum and mean should scale same due to normalization of salience?
                
        out.backward(retain_graph=True)
        optim.step() # stores gradients in delta
        
        # jacobian = GRD.functional.jacobian(jacobian_fn, embeddings, create_graph=True)
        
        # Creating a saliency map (Eq.5 in LaMBO-2 paper)
        saliency = grads.abs().sum(dim=-1)  # summation across hidden dim. Abs value for directionality only
        saliency = saliency.pow(1.0 / self.temperature)
        # saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        saliency = saliency.clamp(min=self.eps).to(self.device)
        return saliency.squeeze()

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
        probabilities = saliency_map.masked_fill(solubility_preds >= self.residue_thresh, 0.0) # exclude already soluble sequences
        probabilities = probabilities / probabilities.sum()
        nonzero = probabilities.count_nonzero().item()
        # print(probabilities)
        topk_edit_probs, topk_edit_pos = torch.topk(probabilities, min(self.topk, nonzero))
        # print(topk_edit_pos)
        mask = torch.zeros_like(probabilities).scatter(0, topk_edit_pos, torch.ones_like(probabilities))
        
        return mask
    
    def mask_sequence(self, input_ids, interpolated_solubility):
        """Mask out low-vaue residues"""
        return torch.where(
            interpolated_solubility==0,
            input_ids.to(self.device),
            torch.full_like(input_ids, self.tokenizer.mask_token_id).to(self.device)
        ).unsqueeze(0)

    def compute_frac_soluble(self, input_ids, solubility_preds):
        """Helper method to compute density of soluble residues in a sequence"""
        num_soluble = (solubility_preds > self.residue_thresh).sum().item()
        frac_soluble = (num_soluble / solubility_preds.numel())
        return frac_soluble

    def compute_logits_value_from_hidden(self, hidden_states, attention_mask):
        batch = {"embeddings": hidden_states.squeeze(), "attention_mask": attention_mask.squeeze()}
        solubility_preds = self.classifier_model(batch)
        # print(self.medlm_lm.__dict__)
        return self.medlm_lm.lm_head(hidden_states), solubility_preds

    def update_logits(self, p_ht_prime, og_logits, hidden_state_mask, attention_mask, optimizer, delta):
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

        # neta = self.config.value.guidance.step_size
        lamb = self.config.value.guidance.reg_strength
        # eps = torch.normal(mean=0.0, std=1.0, size=(1,)).clamp(-3.0, 3.0).item() # eps ~ N(0, 1)

        hidden_state_mask = hidden_state_mask.unsqueeze(-1)
        
        h_current = p_ht_prime + hidden_state_mask * delta
        new_logits, v_ht_prime = self.compute_logits_value_from_hidden(h_current, attention_mask)
        
        loss = lamb * F.kl_div(new_logits, og_logits) - v_ht_prime.sum() # summing across dim instead of mean (this is used in the official implementation)
        
        loss.backward(retain_graph=True) # computing gradients relative to the input hidden states (the gradient param in the langevin dynamics eq)
        optimizer.step() # populates delta with the gradient of the loss function * the step size!!!!!
        optimizer.zero_grad()
        
        # stoch_term = math.sqrt(2 * neta * self.temperature) * eps
        
        # h_new = (h_current + (delta.data * hidden_state_mask)).detach()
        # updated_logits, _vht1 = self.compute_logits_value_from_hidden(h_new, attention_mask)
        
        return delta
    
    def optimized_sampling(self, og_logits, og_hidden, attention_mask, n_steps=None, use_stoch_term=False):
        """Main entry point to optimize a generated sequence."""
        # if og_solubility >= self.sequence_density:
        #     return og_seq, og_solubility
        if n_steps is None: n_steps = self.config.value.guidance.n_steps

        # cur_hidden = og_hidden
        # cur_logits = og_logits
        batch = {"embeddings": og_hidden.squeeze(), "attention_mask": attention_mask.squeeze()}
        solubility_preds = self.classifier_model(batch)
        
        neta = self.config.value.guidance.step_size
        
        delta = nn.Parameter(torch.zeros_like(og_hidden), requires_grad=True)
        optimizer = torch.optim.Adagrad([delta], lr=neta) # here we initialize an sgd per iteration, but the official implementation does an adagrad, possibly for smaller scaling updates over time?
        optimizer.zero_grad()
        
        # Continuous optimization until solubility density threshold is reached
        with torch.enable_grad():
            for i in range(n_steps):

                # Compute saliency map and edit positions
                saliency_map = self.compute_saliency(og_hidden + delta.data, attention_mask)
                mask = self.determine_edit_positions(saliency_map, solubility_preds)

                # Optimize and generate the new sequence
                delta = self.update_logits(
                    p_ht_prime=og_hidden,
                    og_logits=og_logits,
                    hidden_state_mask=mask,
                    attention_mask=attention_mask,
                    optimizer=optimizer,
                    delta=delta
                )

        h_new = og_hidden + delta.data
        new_logits, _new_sol = self.compute_logits_value_from_hidden(h_new, attention_mask)
                        
        return new_logits
    
    def sample_guidance(self, x_0 = None, num_steps=None, eps=1e-5, bsz=1, guidance=True):
        if num_steps is None: num_steps = self.config.sampling.steps
        
        # if a prior sequence is given
        if x_0 is not None:
            x = x_0.input_ids.to(self.device)
            attention_mask = x_0.attention_mask.to(self.device)
        else: # generate prior of just mask tokens
            x = self.diffusion._sample_prior(bsz, self.config.model.length).to(self.device)
            attention_mask = torch.ones_like(x, device=self.device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        
        dt = (1 - eps) / num_steps
        
        for i in trange(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            # sigma_t, _ = self.diffusion.noise(t)
            logits, hidden_states = self.diffusion.backbone.forward_hidden(x, attention_mask)
            if guidance: logits = self.optimized_sampling(logits, hidden_states, attention_mask)
            logits_wrapper = Namespace() # quick fix to be compatible with subs function, bc it accesses logits.logits for some reason
            logits_wrapper.logits = logits
            p_x0 = self.diffusion._subs_parameterization(logits=logits_wrapper, xt=x)
            
            # computing move chance categoricals
            move_chance_t = t[:, None, None]
            move_chance_s = (t - dt)[:, None, None]
            q_xs = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.diffusion.mask_index] = move_chance_s[:, :, 0]
            _x = _sample_categorical(q_xs) # sampling move chances in reverse process
            
            copy_flag = (x != self.diffusion.mask_index).to(x.dtype)
            x = (copy_flag * x + (1 - copy_flag) * _x).squeeze(0)
                    
        # if self.config.sampling.noise_removal:
        #     t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        #     unet_conditioning
        
        return x, self.tokenizer.decode(x.squeeze())
            
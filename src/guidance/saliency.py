import os
import torch

from transformers import AutoModel, AutoTokenizer

from main import SolubilityClassifier


class SolubilityGuider:
    def __init__(self, config):
        self.config = config
        self.classifier_model = SolubilityClassifier(config).eval()
        self.memdlm = AutoModel.from_pretrained(self.config.value.training.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.value.training.pretrained_model)

        ckpt_path = os.path.join(self.config.value.training.ckpt_path, "best_model.ckpt")
        state_dict = self.classifier_model._get_state_dict(ckpt_path)
        self.classifier_model.load_state_dict(state_dict)

        self.eps = self.config.value.guidance.epsilon
        self.temperature = self.config.value.guidance.temperature
        self.topk = self.config.value.guidance.topk

    def __call__(self, sequence):
        return self.determine_edit_positions(sequence)

    def compute_saliency(self, sequence):
        """
        Compute a saliency map using gradients as defined in LaMBO-2 (https://arxiv.org/pdf/2305.20009)
        Args:
            sequence (str): generated, unoptimized protein sequence
        Returns:
            saliency (torch.Tensor): normalized saliency scores [seq_len]
            input_ids (torch.Tensor): squeezed token IDS [seq_len]
        """

        # tokenize input protein sequence
        tokens = self.tokenizer(sequence, return_tensors='pt')
        input_ids = tokens['input_ids']
        attention_masks = tokens['attention_mask']

        # Get sequence embeddings using diffusion model hidden states
        with torch.no_grad():
            outputs = self.mdlm_model(input_ids=input_ids, attention_mask=attention_masks)
            sequence_embeddings = outputs.last_hidden_state.squeeze(0)
        sequence_embeddings.requires_grad_(True) # enable gradient for backprop

        # Get per-residue solubility predictions (of shape [seq_len]) from trained classifier model
        self.solubility_preds = self.classifier_model(embeds=sequence_embeddings,
                                           mask=attention_masks).requires_grad_(True)

        # Compute gradients of value function wrt hidden states
        gradients = torch.autograd.grad(
            outputs=self.solubility_preds,
            inputs=sequence_embeddings,
            create_graph=True,
        )[0].squeeze(0)  # shape [seq_length, hidden_dim=640] since we optimize 1 sequence at a time

        # Creating a saliency map (Eq.5 in LaMBO-2 paper)
        saliency = gradients.abs().sum(dim=-1)  # summation across hidden dim. Abs value for directionality only
        saliency = saliency.pow(1.0 / self.temperature) # temperature scaling and numerical stability
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) # normalization
        saliency = torch.clamp(saliency, min=self.eps) # numerical stability

        return saliency, input_ids.squeeze(0)


    def determine_edit_positions(self, sequence):
        """"Determine edit positions for a generated sequence.
        We define low-value residues as those with low solubility scores
        that do not contribute to the saliency of the sequence.
        
        Args:
            - sequence (str): protein sequence
        Returns:
            - edit_positions (torch.Tensor): token IDs with low-value residues remasked

        """
        # Create a sequence-level saliency map
        saliency_map, sequence_tokens = self.compute_saliency(sequence)

        # Selection procedure to determine low-value residues
        combined_scores = saliency_map * self.solubility_preds # interpolate between saliency and solubility pred
        probabilities = torch.nn.functional.softmax(combined_scores, dim=0) # probability distribution
        probabilities[(self.solubility_preds == 1.0)] = 0.0 # zero out selection chance for soluble residues
        topk_edit_probs, topk_edit_pos = torch.topk(probabilities, self.topk) # select edit positions

        # Optimize the protein sequence
        edited_sequence = torch.where(
            probabilities > topk_edit_probs, # retain high-scoring residues
            sequence_tokens,
            torch.full_like(sequence_tokens, self.tokenizer_mask_token_id) # Mask low-value residues
        )

        return edited_sequence
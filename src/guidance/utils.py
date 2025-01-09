import sys
import torch
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler


def _print(message):
    print(message)
    sys.stdout.flush()


class NoisingScheduler:
    def __init__(self, config, tokenizer, eps=1e-3):
        self.eps = eps
        self.config = config
        self.tokenizer = tokenizer
        self.sampling_eps = self.config.training.sampling_eps
    
    def sample_t(self, bsz, device):
        _eps_t = torch.rand(bsz, device=device)
        offset = torch.arange(bsz, device=device) / bsz
        _eps_t = (_eps_t / bsz + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        return t

    def get_noise(self, t):
        rate_noise = (1 - self.eps) / (1 - (1 - self.eps) * t)
        move_chance = 1 - torch.exp(-rate_noise[:, None])
        return move_chance

    def q_xt(self, x, move_chance):
        max_masks = self.config.training.max_mask_rate
        actual_seq_length = (x != 1).sum(dim=1, keepdim=True)
        max_mask_length = (actual_seq_length * max_masks).long()
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        restricted_move_indices = torch.zeros_like(move_indices, dtype=torch.bool)

        for i in range(x.shape[0]):
            true_positions = torch.where(move_indices[i])[0]
            if len(true_positions) > max_mask_length[i]:
                selected_positions = true_positions[:max_mask_length[i].item()]
                restricted_move_indices[i, selected_positions] = True
            else:
                restricted_move_indices[i] = move_indices[i]
        xt = torch.where(restricted_move_indices, self.tokenizer.mask_token_id, x)
        return xt

    def __call__(self, x0, attention_mask):
        t = self.sample_t(x0.shape[0], x0.device)
        noise = self.get_noise(t)
        xt = self.q_xt(x0, noise)
        xt = xt * attention_mask + (1 - attention_mask) # dont noise pad tokens
        return xt
    

class CosineWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_ratio = eta_ratio  # The ratio of minimum to maximum learning rate
        super(CosineWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = (1 - self.eta_ratio) * cosine_decay + self.eta_ratio

        return [decayed_lr * base_lr for base_lr in self.base_lrs]
import torch
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler
from src.diffusion.diffusion import q_xt

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

class NoisingScheduler:
    def __init__(self, config, eps=1e-3):
        self.eps = eps
        self.config = config
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

    def forward(self, x0, attention_mask):
        t = self.sample_t(x0.shape[0], x0.device)
        noise = self.get_noise(t)
        xt = q_xt(x0, noise)
        xt = xt * attention_mask + (1 - attention_mask) # dont noise pad tokens
        return xt
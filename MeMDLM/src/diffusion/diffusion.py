import itertools
import math
import os
import sys
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch.nn as nn
import torch
import time
import gc
import torchmetrics
import transformers

from torch import Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModelForMaskedLM, AutoModel, EsmConfig, AutoTokenizer, EsmForMaskedLM

from MeMDLM.src.diffusion import pl_data_loader
from MeMDLM.src.diffusion import ema
from MeMDLM.src.diffusion import utils
from MeMDLM.src.diffusion import noise_schedule

LOG2 = math.log(2)

def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor

class NLL(torchmetrics.aggregation.MeanMetric):
    pass

class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension."""
        return self.mean_value / self.weight / LOG2

class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity."""
        return torch.exp(self.mean_value / self.weight)



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


class WrapESM(nn.Module):
    def __init__(self, model_path, esm_model):
        super(WrapESM, self).__init__()

        default_config = EsmConfig.from_pretrained(esm_model)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model)
        
        esm_config = EsmConfig(
            vocab_size=default_config.vocab_size,
            hidden_size=896, #default_config.hidden_size,
            num_hidden_layers=default_config.num_hidden_layers,
            num_attention_heads=14, #default_config.num_attention_heads,
            intermediate_size=3584, #default_config.intermediate_size,
            max_position_embeddings=2048,
            hidden_dropout_prob=default_config.hidden_dropout_prob,
            attention_probs_dropout_prob=default_config.attention_probs_dropout_prob,
            position_embedding_type=default_config.position_embedding_type,
            pad_token_id=self.tokenizer.pad_token_id
        )

        self.model = EsmForMaskedLM(esm_config).to(self.device)
        #self.model = AutoModelForMaskedLM.from_pretrained('facebook/esm2_t30_150M_UR50D')

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def unfreeze_attn_layers(self):
        model_layers = len(self.model.esm.encoder.layer)
        for i, layer in enumerate(self.model.esm.encoder.layer):
            if i >= model_layers - 5:  # Fine-tune only last 5 layers
                for module in [layer.attention.self.key, layer.attention.self.query, layer.attention.self.value]:
                    for param in module.parameters():
                        param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
        
    def reset_weights(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = torch.empty_like(param).normal_()

    def forward(self, inputs, sigma, attention_mask):
        output = self.model(input_ids=inputs, attention_mask=attention_mask)
        return output.logits

    def forward_hidden(self, inputs, attention_mask):
        output = self.model(input_ids=inputs, attention_mask=attention_mask)
        return output.logits, output.hidden_states[-1]

    def save_model(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def load_model(self, load_dir):
        self.model = AutoModel.from_pretrained(load_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        self.parameterization = self.config.parameterization
        self.subs_masking = self.config.subs_masking
        self.T = self.config.T
        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)

        self.tokenizer = tokenizer
        self.mask_index = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.sampling.predictor

        self.gen_ppl_eval_model_name_or_path = self.config.eval.gen_ppl_eval_model_name_or_path
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        
        # Initialize backbone for full-parameter training starting from random weights
        if 'pretrain' in self.config.backbone:
            bert_model = self.config.training.esm_model_path
        elif 'fine_tune' in self.config.backbone:
            bert_model = self.config.checkpointing.pretrained_esm_mdlm_automodel_path
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")
        self.backbone = WrapESM(model_path=bert_model, esm_model="facebook/esm2_t30_150M_UR50D")

        # Metrics (automatically reset at end of epoch)
        metrics = torchmetrics.MetricCollection({"nll": NLL(), "bpd": BPD(), "ppl": Perplexity(),})
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Generative perplexity
        self.gen_ppl_metric = Perplexity()
        self.eval_model_tokenizer = AutoTokenizer.from_pretrained(self.gen_ppl_eval_model_name_or_path)
        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token = self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id = self.eval_model_tokenizer.eos_token_id

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

        self.curriculum = self.config.training.curriculum_learning
        self.max_epochs = self.config.trainer.max_epochs
        self.init_mask_rate = self.config.training.initial_mask_rate
        self.mask_increment = self.config.training.mask_increment
        self.max_mask_rate = self.config.training.max_mask_rate

        # Exponential Moving Average (EMA)
        if self.config.training.ema > 0:
            self.ema = ema.ExponentialMovingAverage(self.backbone.parameters(),
                                                    self.config.training.ema)
        else:
            self.ema = None


    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        
        if self.parameterization == "sedd":
            assert not self.importance_sampling
            assert not self.change_of_variables
        
        if self.parameterization == "d3pm":
            assert self.T > 0
        
        if self.T > 0:
            assert self.parameterization in {"d3pm", "subs"}
        
        if self.subs_masking:
            assert self.parameterization == "d3pm"


    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["completed"] = (
            checkpoint["loops"]["fit_loop"]["epoch_loop.automatic_optimization.optim_progress"]
            ["optimizer"]["step"]["total"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["completed"] = (
            checkpoint["loops"]["fit_loop"]["epoch_loop.automatic_optimization.optim_progress"]
            ["optimizer"]["step"]["current"]["completed"]
            * self.trainer.accumulate_grad_batches
        )

        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"]["_batches_that_stepped"] = (
            checkpoint["loops"]["fit_loop"]["epoch_loop.automatic_optimization.optim_progress"]
            ["optimizer"]["step"]["total"]["completed"]
        )

        if "sampler" not in checkpoint.keys():
            checkpoint["sampler"] = {}

        if hasattr(self.trainer.train_dataloader.sampler, "state_dict"):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get("random_state", None)
        else:
            checkpoint["sampler"]["random_state"] = None

        self.backbone.save_model(self.config.checkpointing.pretrained_esm_mdlm_automodel_path)


    def on_train_start(self):
        torch.cuda.empty_cache()
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    #     # Adapted from:
    #     # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    #     distributed = (
    #         self.trainer._accelerator_connector.use_distributed_sampler
    #         and self.trainer._accelerator_connector.is_distributed)
    #     if distributed:
    #         sampler_cls = pl_data_loader.FaultTolerantDistributedSampler
    #     else:
    #         sampler_cls = pl_data_loader.RandomFaultTolerantSampler

    #     updated_dls = []
    #     for dl in self.trainer.fit_loop._combined_loader.flattened:
    #         if hasattr(dl.sampler, 'shuffle'):
    #             dl_sampler = sampler_cls(
    #                 dl.dataset, shuffle=dl.sampler.shuffle)
    #         else:
    #             dl_sampler = sampler_cls(dl.dataset)
    #         if (distributed
    #             and self.fast_forward_epochs is not None
    #             and self.fast_forward_batches is not None):
    #             dl_sampler.load_state_dict({
    #                 'epoch': self.fast_forward_epochs,
    #                 'counter': (self.fast_forward_batches
    #                             * self.config.loader.batch_size)})

    #         from functools import partial
    #         from pl_data_loader import collate_fn
    #         collate_partial = partial(collate_fn, tokenizer=self.tokenizer)
    #         torch.cuda.empty_cache()

    #         updated_dls.append(
    #             torch.utils.data.DataLoader(
    #                 dl.dataset,
    #                 batch_size=self.config.loader.batch_size,
    #                 num_workers=self.config.loader.num_workers,
    #                 pin_memory=self.config.loader.pin_memory,
    #                 sampler=dl_sampler,
    #                 shuffle=False,
    #                 persistent_workers=False,
    #                 collate_fn=collate_partial))

    #     self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()
        if self.ema:
            self.ema.update(self.backbone.parameters())
    
    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == 'ar'
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _init_epoch_metrics(self):
        mode_batches = {'train': self.config.training.train_samples,
                        'val': self.config.training.val_samples,
                        'test': self.config.training.test_samples}

        if self.mode not in mode_batches:
            raise NotImplementedError(f'Mode {self.mode} unsupported.')
    
        self.n_batches = mode_batches[self.mode]
        self.curr_epoch = self.trainer.current_epoch
        self._compute_mask_rates()

    def _compute_mask_rates(self, eps=1e-3):
        """"Pre-defined masking rates"""
        # Compute min and max masking rates for the epoch
        self.epoch_rates = [self.config.training.initial_mask_rate + (i * self.mask_increment)
                            for i in range(self.max_epochs)]
        self.min_mask_rate = self.epoch_rates[self.curr_epoch]
        self.max_mask_rate = self.epoch_rates[min(self.curr_epoch + 1, self.max_epochs - 1)]

        # Cosine noise. Compute total noise and the noise for each step
        t = torch.linspace(eps, 1, self.n_batches) # Generate step values to distribute masking rate across the epoch
        self.mask_rates = self.min_mask_rate + 0.5 * (self.max_mask_rate - self.min_mask_rate) * (1 - torch.cos(torch.pi * t))
        self.dsigmas = torch.cumsum(self.mask_rates, dim=0) * (t[1] - t[0])

    def on_train_epoch_start(self):
        self.mode = 'train'

        # print(f'curr epoch: {self.curr_epoch}')
        # print(f'min: {self.min_mask_rate}')
        # print(f'max: {self.max_mask_rate}')
        # print(f'rates: {self.mask_rates}')
        # print(f'dsigmas: {self.dsigmas}')

        self._init_epoch_metrics()
        self.backbone.train()

    def training_step(self, batch, batch_idx):
        # Initialize throughput calculation
        start_time = time.time() 

        loss = self._compute_loss(batch, batch_idx, prefix='train')
        self.log(name='trainer/loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=False)
        
        # Calculate throughput
        elapsed_time = time.time() - start_time
        total_tokens = batch['input_ids'].numel()
        throughput = total_tokens / elapsed_time
        self.log(name='trainer/throughput',
                value=throughput,
                on_step=True,
                on_epoch=False,
                sync_dist=False)
    
        # Log gradients
        tot_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm(2)
                tot_norm += param_norm.item() ** 2
        tot_norm = tot_norm ** 0.5
        self.log(name='trainer/grad_norm',
                 value=tot_norm,
                 on_step=True,
                 on_epoch=False,
                 sync_dist=False)

        return loss

    def on_validation_epoch_start(self):
        self.mode = 'val'
        self._init_epoch_metrics()
        gc.collect()
        torch.cuda.empty_cache()
        if self.ema:
            self.ema.store(self.backbone.parameters())
            self.ema.copy_to(self.backbone.parameters())
        self.backbone.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx, prefix='val')
        self.log(name='val/loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=False)

        return loss

    def on_validation_epoch_end(self):
        gc.collect()
        torch.cuda.empty_cache()
        if self.ema:
            self.ema.restore(self.backbone.parameters())

    def on_test_epoch_start(self):
        self.mode = 'test'
        self._init_epoch_metrics()
        if self.ema:
            self.ema.store(self.backbone.parameters())
            self.ema.copy_to(self.backbone.parameters())
        self.backbone.eval()
        self.test_metrics.reset()

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, batch_idx, prefix='test')
        self.log('test/loss',
                value=loss.item(),
                on_step=False,
                on_epoch=True,
                sync_dist=False)

    def on_test_epoch_end(self):
        if self.ema:
            self.ema.restore(self.backbone.parameters())
        
        for metric_name, metric_value in self.test_metrics.compute().items():
            self.log(metric_name, metric_value, sync_dist=False)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )

        self.total_steps = self.config.trainer.max_steps
        scheduler = CosineWarmup(optimizer,
                                warmup_steps=self.config.lr_scheduler.num_warmup_steps,
                                total_steps=self.total_steps)

        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'frequency': 1,
                          'monitor': 'val/loss',
                          'name': 'trainer/lr'}

        return [optimizer], [scheduler_dict]


    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt with pre-defined max masking rate.
        Args:
            x: int torch.Tensor with shape (batch_size, diffusion_model_input_length), input. 
            move_chance: float torch.Tensor with shape (batch_size, 1).
        Returns:
            xt: Tensor with the same shape as x, containing the noisy sample.
        """
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
        
        xt = torch.where(restricted_move_indices, self.mask_index, x)
        return xt


    def scheduled_q_xt(self, x, mask_rate):
        """Noise a sequence via pre-scheduled forward noising process."""
        move_indices = torch.rand(*x.shape, device=x.device) < mask_rate
        xt = torch.where(move_indices, self.mask_index, x)
        return xt
    

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits = logits.logits
        logits[:, :, self.mask_index] += self.neg_infinity
        # logits[:, :, self.tokenizer.eos_token_id] += self.neg_infinity
        # logits[:, :, self.tokenizer.cls_token_id] += self.neg_infinity
        # logits[:, :, self.tokenizer.pad_token_id] += self.neg_infinity

        # Normalize the logits such that x.exp() is prob distribution over vocab_size.
        # logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True) 

        logsumexp_vals = torch.logsumexp(logits, dim=-1, keepdim=True)
        logsumexp_vals = logsumexp_vals.masked_fill(torch.isinf(logsumexp_vals), 0)
        logits = logits - logsumexp_vals

        #print(f'after logsumexp transformation: {logits.exp().sum(dim=-1)}')

        #print('nans' if logits.isnan().any() else 'nans after logsumexp')

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values to -inf 
        # except for the indices corresponding to the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        
        return logits

    def forward(self, x, attention_mask, print_logits=False):
        """Returns log score."""
        # sigma = self._process_sigma(sigma)
        logits = self.backbone(x, attention_mask)
        if self.parameterization == 'subs':
            return self._subs_parameterization(logits=logits, xt=x)
        return logits


    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t


    def _forward_pass_diffusion(self, x0, attention_mask, batch_idx):
        # Curriculum learning enforces t ~ Uniform(min_rate, max_rate), otherwise, t ~ Uniform(0, 1)
        if self.curriculum:
            # Compute move chance and noise sample directly
            move_chance = torch.empty_like(x0, dtype=torch.float, device=x0.device).uniform_(self.min_mask_rate, self.max_mask_rate)
            xt = self.scheduled_q_xt(x0, move_chance)
            #print(f'frac masked: {(xt==self.mask_index).sum().item() / xt.numel()}')

            # Forward pass through model – NO SUBS
            model_output = self.backbone.forward(inputs=xt, sigma=None, attention_mask=attention_mask)
        
        else:
            # Compute timesteps and parameterize noise
            t = self._sample_t(x0.shape[0], x0.device)
            sigma, dsigma = self.noise(t)

            # Forward noising process
            move_chance = (1 - torch.exp(-sigma[:, None]))#.clamp(max=self.max_mask_rate)
            xt = self.q_xt(x0, move_chance)

            # Forward pass through model with SUBS parameterization
            model_output = self.forward(x=xt, attention_mask=attention_mask)


        # # Check that the masking was applied correctly
        # num_masked = (xt == self.mask_index).sum().item()
        # tot_tokens = x0.numel()
        # frac_masked = (num_masked / tot_tokens) * 100
        # print(f'num_masked: {num_masked}')
        # print(f'tot tokens: {tot_tokens}')
        # print(f'frac masked: {frac_masked}')
        
        #print(f'move_chance: {move_chance}')
        
        # if mask is None:
        #     xt = self.q_xt(x0, move_chance)
        # else:
        #     xt = x0.where(mask == 1, torch.full_like(x0, self.tokenizer.mask_token_id))
        

        utils.print_nans(model_output, 'model_output')
        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)

        # NELBO simplifies to MLM loss with pre-defined masking rates, so only need log probs
        # Otherwise, standard MDLM. NELBO = log_probs * (alpha_t' / (1 - alpha_t))
        if self.curriculum:
            return - log_p_theta
        else:
            return - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]


    def _loss(self, x0, attention_mask, batch_idx):
        loss = self._forward_pass_diffusion(x0, attention_mask, batch_idx)

        loss_positions = (x0 == self.mask_index) if self.curriculum else attention_mask
        nlls = loss * loss_positions
        token_nll = nlls.sum() / loss_positions.sum()

        # # MLM loss computed just at mask positions, while MLDM considers all tokens
        # if self.curriculum:
        #     mask_positions = (x0 == self.mask_index)
        #     nlls = loss * mask_positions
        #     token_nll = nlls.sum() / mask_positions.sum()
        # else:
        #     nlls = loss * attention_mask
        #     token_nll = nlls.sum() / attention_mask.sum()

        return Loss(loss=token_nll, nlls=nlls, token_mask=loss_positions)


    def _compute_loss(self, batch, batch_idx, prefix):
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']
        else:
            attention_mask = None
        
        # if 'mask' in batch:
        #     mask = batch['mask']
        # else:
        #     mask = None
        
        losses = self._loss(batch['input_ids'], attention_mask, batch_idx)
        loss = losses.loss

        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')

        self.log_dict(metrics,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False)
        return loss
    


    """
    SAMPLING METHODS.
    STILL NEED TO UPDATE FOR COSINE-SCHEDULED MASKING
    """
    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None, attention_mask=None):
        assert self.config.noise.type == 'loglinear'
        
        # Get the noise for the given time step
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        # Compute move chances based on the current and previous time steps
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        
        # Forward pass
        if p_x0 is None:
            p_x0 = self.forward(x, attention_mask).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        
        # Calculate move chance categoricals
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        
        # Gumbel-max sampling from categorical distribution
        _x = _sample_categorical(q_xs)
        
        # Create a copy flag to retain non-masked values from x
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, x_input=None):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.eval.perplexity_batch_size
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if x_input is not None:
            x = x_input.input_ids
            attention_mask = x_input.attention_mask
        else:
            x = self._sample_prior(batch_size_per_gpu, self.config.model.length).to(self.device)
            attention_mask = torch.ones_like(x)
        
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)

            p_x0_cache, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache, attention_mask=attention_mask)
            if (not torch.allclose(x_next, x) or self.time_conditioning):
                # Disable caching
                p_x0_cache = None
            x = x_next
            # print(self.tokenizer.decode(x.squeeze()))

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            unet_conditioning = self.noise(t)[0]
            x = self.forward(x, attention_mask, print_logits=True).argmax(dim=-1)
            # print(self.tokenizer.decode(x.squeeze()))
        
        return x

    @torch.no_grad()
    def compute_masked_perplexity(self, sequences, masked):
        """Compute the pseudo-perplexity of the generated protein sequences.
        
        Args:
            sequences: List of generated protein sequences.
            masked: Masked version of the sequences for evaluation.
            
        Returns:
            pseudo_perplexity: Computed generative perplexity.
        """
        total_nll = 0
        total_tokens = 0

        for sequence in sequences:
            # Tokenize the sequence
            input_ids = self.tokenizer(masked, return_tensors="pt").input_ids.to(self.device)
            gt_ids = self.tokenizer(sequence.upper(), return_tensors="pt").input_ids.to(self.device)

            # Forward pass through the ESM model
            if self.config.mode in ['train', 'ppl_eval']:
                outputs = self.backbone.model.forward(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            elif self.config.mode == "sample_eval":
                outputs = self.backbone.model.forward(input_ids)

            # Reshape logits and true tokens (disregard batch as we handle one sequence at a time)
            logits = outputs.logits.squeeze(0)  # Shape: L, V 
            gt_ids = gt_ids.where(input_ids == 32, torch.full_like(input_ids, -100)).squeeze(0)  # Shape: L

            # Accumulate cross-entropy loss and tokens
            loss = F.cross_entropy(logits, gt_ids, reduction='sum')
            total_nll += loss.item()
            total_tokens += input_ids.ne(self.tokenizer.pad_token_id).sum().item()  # Count tokens excluding padding

        # Compute generative perplexity
        pseudo_perplexity = torch.exp(torch.tensor(total_nll / total_tokens))
        self.gen_ppl_metric.update(pseudo_perplexity)

        return pseudo_perplexity.item()


    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        if self.ema:
            self.ema.store(self.backbone.parameters())
            self.ema.copy_to(self.backbone.parameters())
        
        self.backbone.eval()
        
        samples = self._sample(num_steps=num_steps, eps=eps)
        
        if self.ema:
            self.ema.restore(self.backbone.parameters())
        
        self.backbone.train()
        
        return samples


    @torch.no_grad
    def sample_subs_guidance(self, n_samples, stride_length, num_strides, dt=0.001):
        ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self._sample_prior(n_samples, self.config.model.length).to(self.device)
            
            if target is not None:
                x[:, :-stride_length] = target
                
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ddpm_caching_update(x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
                
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = ((np.concatenate(intermediate_tokens, axis=1)[:, 1:] == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
        
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(self.tokenizer.batch_decode(np.concatenate(intermediate_tokens[:i], axis=1)))
            
        return (sampling_steps, intermediate_text_samples, sequence_lengths)


    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(self.backbone.parameters())
            self.ema.copy_to(self.backbone.parameters())
            
        self.backbone.eval()
        
        (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides, 
            dt=dt
        )
        
        if self.ema:
            self.ema.restore(self.backbone.parameters())
            
        self.backbone.train()
        
        return sampling_steps, samples, sequence_lengths


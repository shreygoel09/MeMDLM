import os
import wandb
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import pl_data_loader as dataloader
from diffusion import Diffusion
import utils

from lightning.pytorch.strategies import DDPStrategy
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset

# wandb.login(key="your_key")
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def _load_from_checkpoint(config, tokenizer):
    if 'hf' in config.backbone:
        return Diffusion(config, tokenizer=tokenizer).to('cuda')
    else:
        model = Diffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            config=config)
    return model

@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure."""
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(f"{config.checkpointing.save_dir}/config_tree.txt", 'w') as fp:
            rich.print(tree, file=fp)

@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
    for dl_type, dl in [('train', train_ds)]:
        print(f'Printing {dl_type} dataloader batch.')
        batch = next(iter(dl))
        print('Batch input_ids.shape', batch['input_ids'].shape)
        first = batch['input_ids'][0, :k]
        last = batch['input_ids'][0, -k:]
        print(f'First {k} tokens:', tokenizer.decode(first))
        print('ids:', first)
        print(f'Last {k} tokens:', tokenizer.decode(last))
        print('ids:', last)

def generate_samples(config, logger, tokenizer):
    logger.info('Generating samples.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps)
            text_samples = intermediate_samples[-1]
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
    print('Text samples:', text_samples)
    if not config.sampling.semi_ar:
        print('Generative perplexity:', model.gen_ppl_metric.compute())
    return text_samples

def _ppl_eval(config, logger, tokenizer, data_module):
    logger.info('Starting Zero Shot Eval.')
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger)
    trainer.test(model, data_module)

def _train(config, logger, tokenizer, data_module):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb)
    if (config.checkpointing.resume_from_ckpt and \
        config.checkpointing.resume_ckpt_path is not None and \
        utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
            ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        accelerator='cuda',
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy="ddp",
        logger=wandb_logger)
    
    model = Diffusion(config, tokenizer=tokenizer)
    # model = torch.compile(model)
    
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    print(torch.cuda.device_count())
    _print_config(config, resolve=True, save_cfg=True)
    
    logger = utils.get_logger(__name__)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    if config.backbone == "vanilla_esm_pretrain":
        train_dataset = load_dataset('csv', data_files=config.data.train.vanilla_esm_train_path)
        val_dataset = load_dataset('csv', data_files=config.data.valid.vanilla_esm_valid_path)
        test_dataset = load_dataset('csv', data_files=config.data.test.vanilla_esm_test_path)
    elif config.backbone == "membrane_esm_finetune" or config.backbone == "dit":
        train_dataset = load_dataset('csv', data_files=config.data.train.membrane_esm_train_path)
        val_dataset = load_dataset('csv', data_files=config.data.valid.membrane_esm_valid_path)
        test_dataset = load_dataset('csv', data_files=config.data.test.membrane_esm_test_path)

    # lst = [i for i in range(1, 200)]

    train_dataset = train_dataset['train']#.select(lst)
    val_dataset = val_dataset['train']#.select(lst)
    test_dataset = test_dataset['train']#.select(lst)
    
    print("batch size: ", config.loader.batch_size)
    
    if config.training.focus_mask:
        collator = dataloader.membrane_collate_fn
    elif config.data.batching == "wrapping":
        collator = dataloader.wrap_collate_fn
    elif config.data.batching == "padding":
        collator = dataloader.collate_fn

    data_module = dataloader.CustomDataModule(
        train_dataset, val_dataset, test_dataset,
        tokenizer, 
        batch_size=config.loader.batch_size,
        collate_fn=collator
       )

    if config.mode == 'sample_eval':
        generate_samples(config, logger, tokenizer)
    elif config.mode == 'ppl_eval':
        _ppl_eval(config, logger, tokenizer, data_module)
    else:
        _train(config, logger, tokenizer, data_module)

if __name__ == '__main__':
    main()

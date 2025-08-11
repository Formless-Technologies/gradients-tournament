#!/usr/bin/env python3
import os
import argparse
import yaml
from datetime import datetime, timedelta, timezone
import torch
from transformers import EarlyStoppingCallback
from trl import (
    SFTConfig,
    SFTTrainer,
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
)
from training_helpers.custom_callbacks import TimeLimitCallback
from training_helpers.dataset_helpers import (
    load_sft_datasets,
    load_sft_pretrain_datasets,
    load_dpo_datasets,
    load_grpo_datasets,
    load_tokenizer,
)
from training_helpers.model_helpers import load_model, get_lora_adapter
from training_helpers.trainer_helpers import build_trainer_args, reward_functions

def parse_args():
    parser = argparse.ArgumentParser(description="Train a causal LM with SFT, DPO, or GRPO")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_trainer(config: dict, model, peft_config, tokenizer, train_ds, eval_ds):

    #### Callbacks ####
    callbacks = []
    if config.get('early_stopping', True):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.get('early_stopping_patience', 3), early_stopping_threshold=1e-4)
        )
    # Calculate time left for job
    time_remaining = datetime.fromisoformat(config['required_finish_time']) - datetime.now(timezone.utc)
    seconds_remaining = max(0.0, time_remaining.total_seconds())

    if seconds_remaining is not None:
        callbacks.append(TimeLimitCallback(seconds_remaining*0.95))
    ###################


    ##### Training Arguments ####
    trainer_kwargs = build_trainer_args(config)

    #####################################
    # SFT
    if config['rl'] == "sft":
        trainer_args = SFTConfig(
            **trainer_kwargs,
        )
        return SFTTrainer(
            model=model,
            args=trainer_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config
        )
    # DPO
    if config['rl'] == "dpo":
        trainer_args = DPOConfig(
            **trainer_kwargs,
        )
        return DPOTrainer(
            model=model,
            ref_model=None,
            args=trainer_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config
        )
    # GRPO
    elif config['rl'] == "grpo":
        trainer_args = GRPOConfig(
            **trainer_kwargs,
        )
        return GRPOTrainer(
            model=model,
            args=trainer_args,
            reward_funcs=reward_functions(config),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
            peft_config=peft_config
        )
        


def run_training(config_path: str) -> None:
    """Run the training loop using the provided YAML config path."""
    config = load_config(config_path)

    # Performance flags
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # after loading config...
    tokenizer = load_tokenizer(config['base_model'], config)

    if config['sft_pretrain']:
        train_dataset, eval_dataset = load_sft_pretrain_datasets(config)
        config['rl'] = "sft"
    elif config['rl'] == "sft":
        train_dataset, eval_dataset = load_sft_datasets(config)
    elif config['rl'] == "dpo":
        train_dataset, eval_dataset = load_dpo_datasets(config)
    elif config['rl'] == "grpo":
        train_dataset, eval_dataset = load_grpo_datasets(config)
        

    model = load_model(config['base_model'], config)
    num_model_parameters = model.num_parameters()
    config['model_params_count'] = num_model_parameters

    if config.get('adapter') == "lora":
        peft_config = get_lora_adapter(model, config)
    else:
        peft_config = None

    trainer = build_trainer(config, model, peft_config, tokenizer, train_dataset, eval_dataset)

    print(f"Starting Training with: \n")
    print(f"Training Type: {config['rl']}")
    print(f"Max Steps: {config['max_steps']}")
    print(f"Eval Steps: {config['eval_steps']}")
    print(f"Save Steps: {config['save_steps']}")
    print(f"Main Training Run: {config['main_training_run']}")
    print(f"Optimizer: {config['optimizer']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Model Architecture: {config['model_architecture']}")
    print(f"Flash Attention Enabled: {config['use_flash_attn']}")
    print(f"Output Directory: {config['output_dir']}")

    if config["eval_probe_run"]:
        metrics = trainer.evaluate()
        print(metrics)
    else:
        trainer.train()


    if config['sft_pretrain'] and not (config['eval_probe_run'] or config['throughput_probe_run']):
        model_obj = getattr(trainer, "model", None)
        try:
            if model_obj is not None and hasattr(model_obj, "merge_and_unload"):
                print(f"Saving Final Model To: {config['output_dir']}")
                merged = model_obj.merge_and_unload()
                merged.save_pretrained(config['output_dir'])
                tokenizer.save_pretrained(config['output_dir'])
            else:
                print(f"Saving Final Model To: {config['output_dir']}")
                trainer.save_model(config['output_dir'])
                tokenizer.save_pretrained(config['output_dir'])
        except Exception as e:
            print(f"Merge-and-unload save failed; falling back to trainer.save_model(): {e}")
            trainer.save_model(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
    elif config['main_training_run']:
        print(f"Saving Final Model To: {config['output_dir']}")
        trainer.save_model(config['output_dir'])
        tokenizer.save_pretrained(config['output_dir'])


def main():
    args = parse_args()
    run_training(args.config)


if __name__ == '__main__':
    main()

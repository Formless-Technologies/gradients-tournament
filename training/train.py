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
            EarlyStoppingCallback(early_stopping_patience=config.get('early_stopping_patience', 4), early_stopping_threshold=1e-4)
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
    print("Initializing Trainer")
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
    
    print(f"Loaded config from {config_path}")
    
    # after loading config...
    tokenizer = load_tokenizer(config['base_model'], config)

    if config['rl'] == "sft":
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

    print("Starting Full Model Training...")
    trainer = build_trainer(config, model, peft_config, tokenizer, train_dataset, eval_dataset)

    trainer.train()


def main():
    args = parse_args()
    run_training(args.config)


if __name__ == '__main__':
    main()

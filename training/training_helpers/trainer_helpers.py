from transformers import SchedulerType
from trl.trainer.grpo_trainer import RewardFunc
import os
import importlib
import sys
import inspect

def build_trainer_args(config: dict):

    lr_scheduler=SchedulerType.COSINE

    # Build Main Invariant Training Arguments
    trainer_kwargs = {
        # Training Length Args
        'max_steps': int(config['max_steps']),
        'logging_steps': int(config['logging_steps']),

        # Optimizer Args
        'optim': config['optimizer'],
        'weight_decay': float(config['weight_decay']),
        'gradient_checkpointing': config['gradient_checkpointing'],
        'gradient_checkpointing_kwargs': {'use_reentrant':False},

        # LR Args
        'learning_rate': float(config['learning_rate']),
        'lr_scheduler_type': lr_scheduler,
        'warmup_steps': config['warmup_steps'],

        # Batch and Memory Args
        'per_device_train_batch_size': int(config['micro_batch_size']),
        'per_device_eval_batch_size': int(config['micro_batch_size']),
        'gradient_accumulation_steps': int(config['gradient_accumulation_steps']),

        # Evaluation and Saving Args
        'eval_strategy': 'steps', 
        'save_strategy': 'best',
        'eval_steps': int(config['eval_steps']),
        'save_steps': int(config['save_steps']),
        'save_total_limit': int(config['save_total_limit']),

        # Best Metric Args
        'metric_for_best_model': config['metric_for_best_model'],
        'load_best_model_at_end': True,

        # Optimization Args
        'bf16': True,
        'use_liger_kernel': config['use_liger_kernel'],
        'auto_find_batch_size': True,

        # Misc Args
        'output_dir': config['output_dir'],
        'report_to': "wandb"
    }

    # Training Type Specific Args
    type_spec_args = {}
    if config['rl'] == "sft":
        type_spec_args = {
            'greater_is_better': False,
            'packing': config['packing'],
            'eval_packing': config['packing'],
            'neftune_noise_alpha': 5 
        }
    elif config['rl'] == "dpo":
        type_spec_args = {
            'beta': float(config['beta']),
            'label_smoothing': float(config['label_smoothing']),
            'greater_is_better': False,
        }
    elif config['rl'] == "grpo":
        type_spec_args = {
            'beta': float(config['beta']),
            'num_generations': int(config['trl']['num_generations']),
            'max_completion_length': int(config['trl']['max_completion_length']),
            'reward_weights': config['trl']['reward_weights'],
            'use_vllm': False,
            'loss_type': "dr_grpo",
            'mask_truncated_completions': True,
            'greater_is_better': True,
            'gradient_checkpointing': False,
        }

    trainer_kwargs |= type_spec_args

    return trainer_kwargs


#### GRPO Specific ####
CONFIG_DIR = os.path.abspath("/workspace/configs/")


##### Custom Funcs for getting GRPO reward functions #####
def reward_functions(config):
    """
    Collects and returns a list of functions for GRPOTrainer.
    """
    funcs = []
    for fqn in config['trl']['reward_funcs']:
        funcs.append(get_reward_func(fqn))
    return funcs


def get_reward_func(reward_func_fqn: str) -> RewardFunc | str:
    """
    Try to load <module>.py from CONFIG_DIR and return its <func>.
    If the file doesn’t exist, just return the original string (HF model path).
    """
    module_name, func_name = reward_func_fqn.rsplit(".", 1)
    module_path = os.path.join(CONFIG_DIR, f"{module_name}.py")
    print(f"→ looking for {module_name!r} at {module_path!r}, exists? {os.path.isfile(module_path)}")
    # 1) if we have an on-disk file, dynamically import it
    if os.path.isfile(module_path):
        # drop any cached module so we always load the newest version
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # get the function
        if not hasattr(module, func_name):
            raise AttributeError(
                f"Module {module_name!r} has no attribute {func_name!r}"
            )
        reward_func = getattr(module, func_name)

        # sanity check signature
        sig = inspect.signature(reward_func)
        if len(sig.parameters) < 2:
            raise ValueError(
                "Reward function must accept at least two arguments: "
                "prompts: list and completions: list"
            )

        return reward_func

    # 2) otherwise fall back to treating the FQN string as a model-path
    return reward_func_fqn

from pydantic import BaseModel
from pydantic import Field
import uuid
from datetime import datetime
from enum import Enum
from datetime import timedelta
import os
import uuid
import re
import toml
import yaml
from transformers import AutoTokenizer
from transformers import AutoConfig

##### HELPERS #####

class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset
    S3 = "s3"

class TaskType(str, Enum):
    INSTRUCTTEXTTASK = "InstructTextTask"
    IMAGETASK = "ImageTask"
    DPOTASK = "DpoTask"
    GRPOTASK = "GrpoTask"
    CHATTASK = "ChatTask"

    def __hash__(self):
        return hash(str(self))


class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""
    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            "\"\"\"Reward function that favors shorter, more concise answers.\"\"\"",
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]"
        ]
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None

class InstructTextDatasetType(BaseModel):
    system_prompt: str | None = ""
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None

class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []
    extra_column: str | None = None

class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"

TextDatasetType = InstructTextDatasetType | DpoDatasetType | GrpoDatasetType

def create_dataset_entry(
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType | GrpoDatasetType,
    is_eval: bool = False,
) -> dict:
    dataset_entry = {"path": dataset}

    if isinstance(dataset_type, InstructTextDatasetType):
        dataset_entry.update(_process_instruct_dataset_fields(dataset_type))
    elif isinstance(dataset_type, DpoDatasetType):
        dataset_entry.update(_process_dpo_dataset_fields(dataset_type))
    elif isinstance(dataset_type, GrpoDatasetType):
        dataset_entry.update(_process_grpo_dataset_fields(dataset_type))
    else:
        raise ValueError("Invalid dataset_type provided.")

    return dataset_entry

def _process_grpo_dataset_fields(dataset_type: GrpoDatasetType) -> dict:
    field_prompt = dataset_type.field_prompt
    extra_column = dataset_type.extra_column

    full_template_config = {"field_prompt": field_prompt, "extra_column": extra_column}

    return full_template_config


def _process_dpo_dataset_fields(dataset_type: DpoDatasetType) -> dict:

    field_prompt = dataset_type.field_prompt
    field_chosen = dataset_type.field_chosen
    field_rejected = dataset_type.field_rejected
    full_template_config = {"field_prompt": field_prompt,  "field_chosen": field_chosen, "field_rejected": field_rejected}

    return full_template_config


def _process_instruct_dataset_fields(dataset_type: InstructTextDatasetType) -> dict:
    field_instruction = dataset_type.field_instruction
    field_input = dataset_type.field_input
    field_output = dataset_type.field_output
    full_template_config = {"field_instruction": field_instruction,  "field_input": field_input, "field_output": field_output}

    return full_template_config

def create_reward_funcs_file(reward_funcs: list[str], task_id: str) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = f"/workspace/configs/{filename}.py"

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names

def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)


def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


####################

def update_model_info(config: dict, model: str, task_id: str = "", expected_repo_name: str | None = None):
    # update model info
    model_path = f"/cache/models/{model.replace('/', '--')}"
    config['base_model'] = model_path
    config['model_params_count'] = 0

    # Calculate sequence length and model architecture
    model_config = AutoConfig.from_pretrained(model_path)
    architectures = model_config.architectures
    if len(architectures) > 1:
            config['model_architecture'] = "Multiple architectures"
    config['model_architecture'] = architectures[0].strip().lower()
    model_max_sequence_length = model_config.max_position_embeddings
    largest_trainable_sequence_length = config['sequence_len']
    config['sequence_len'] = min(model_max_sequence_length, largest_trainable_sequence_length)


    # Model specific configs
    if any(k in model.lower() for k in ("meta-llama-3.1")):
        config['packing'] = False
        config["use_liger_kernel"] = False

    liger_model_architectures = [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm",
    ]

    if config['model_architecture'] in liger_model_architectures:
        config["use_liger_kernel"] = True
    else:
        config["use_liger_kernel"] = False

    no_flash_attention_architectures = [
        "phi3forcausallm",
        "falconforcausallm"
    ]
    if config['model_architecture'] in no_flash_attention_architectures:
        config["use_flash_attn"] = False
    else:
        config["use_flash_attn"] = True
        
    return config

def add_throughput_information(config_path: str, steps_per_minute: float):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config['steps_per_minute'] = steps_per_minute

    save_config(config, config_path)


def add_eval_time_information(config_path: str, seconds_per_eval: float):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config['seconds_per_eval'] = seconds_per_eval

    save_config(config, config_path)

def modify_model_location(config_path: str, new_model_location: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config['base_model'] = new_model_location

    save_config(config, config_path)


def setup_config(
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    task_id: str,
    expected_repo_name: str | None,
    required_finish_time: str | None,
    testing: bool | None
):
    if testing:
        with open("/workspace/configs/base_testing.yml", "r") as file:
            config = yaml.safe_load(file)
        config['testing'] = True
    else:
        with open("/workspace/configs/base.yml", "r") as file:
            config = yaml.safe_load(file)
        config['testing'] = False
    
    # Useful config
    config['task_id'] = task_id
    config['required_finish_time'] = required_finish_time
    
    # RL specific config
    # SFT
    if isinstance(dataset_type, InstructTextDatasetType):
        config['rl'] = "sft"
        config['learning_rate'] = 1e-4
    # DPO
    elif isinstance(dataset_type, DpoDatasetType):
        config['rl'] = "dpo"
        config['learning_rate'] = 5e-7
        config['label_smoothing'] = 0.0
        config['beta'] = 0.04

    # GRPO
    elif isinstance(dataset_type, GrpoDatasetType):
        config['rl'] = "grpo"
        config['learning_rate'] = 5e-6
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
        )
        config['eval_steps'] = 100
        config['save_steps'] = 100
        config['trl'] = {}
        config['trl']['beta'] = 0.04
        config['trl']['max_completion_length'] = 64
        config['trl']['use_vllm'] = False
        config['trl']['num_generations'] = 2
        config['trl']['reward_funcs'] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config['trl']['reward_weights'] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]
        config['rl_beta'] = 0.1
        config['beta'] = 0.04


    # Setup Datasets    
    config['datasets'] = []
    dataset_entry = create_dataset_entry(dataset, dataset_type)
    config['datasets'].append(dataset_entry)

    # Update model specific config
    config = update_model_info(config, model, task_id, expected_repo_name)
    
    # Setup Lora if it is used
    if config['adapter'] == "lora":
        # Looks silly right now but useful if want to modify based on rl type
        config['lora_r'] = config['lora_r']
        config['lora_alpha'] = config['lora_alpha']
        config['lora_dropout'] = config['lora_dropout']

    # Setup WandB
    log_wandb = True
    WANDB_LOGS_DIR = "/cache/wandb_logs"
    if log_wandb:
        os.environ['WANDB_RUN_ID'] = f"{task_id}_{expected_repo_name}"
        os.environ['WANDB_NAME'] = f"{task_id}_{expected_repo_name}"
        os.environ['WANDB_MODE'] = "offline"
        os.makedirs(WANDB_LOGS_DIR, exist_ok=True)

    # Setup output dir
    output_dir = f"/app/checkpoints/{task_id}/{expected_repo_name}"
    config['output_dir'] = output_dir

    # Modify Config and save
    config_path = f"/workspace/configs/{task_id}.yml"

    save_config(config, config_path)

    return config

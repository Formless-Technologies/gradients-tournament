from pathlib import Path
import os
import configs.trainer_constants as train_cst
from enum import Enum
from pydantic import BaseModel
from pydantic import Field

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

class RewardFunction(BaseModel):
    """Model representing a reward function with its metadata"""

    reward_func: str = Field(
        ...,
        description="String with the python code of the reward function to use",
        examples=[
            "def reward_func_conciseness(completions, **kwargs):",
            '"""Reward function that favors shorter, more concise answers."""',
            "    return [100.0/(len(completion.split()) + 10) for completion in completions]",
        ],
    )
    reward_weight: float = Field(..., ge=0)
    func_hash: str | None = None
    is_generic: bool | None = None

class GrpoDatasetType(BaseModel):
    field_prompt: str | None = None
    reward_functions: list[RewardFunction] | None = []

class DpoDatasetType(BaseModel):
    field_prompt: str | None = None
    field_system: str | None = None
    field_chosen: str | None = None
    field_rejected: str | None = None
    prompt_format: str | None = "{prompt}"
    chosen_format: str | None = "{chosen}"
    rejected_format: str | None = "{rejected}"

class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"

def get_checkpoints_output_path(task_id: str, repo_name: str) -> str:
    return str(Path(train_cst.OUTPUT_CHECKPOINTS_PATH) / task_id / repo_name)

def get_image_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    base_path = str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)
    if os.path.isdir(base_path):
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(base_path, files[0])
    return base_path

def get_image_training_images_dir(task_id: str) -> str:
    return str(Path(train_cst.IMAGE_CONTAINER_IMAGES_PATH) / task_id / "img")

def get_image_training_config_template_path(model_type: str) -> str:
    model_type = model_type.lower()
    if model_type == ImageModelType.SDXL.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_sdxl.toml")
    elif model_type == ImageModelType.FLUX.value:
        return str(Path(train_cst.IMAGE_CONTAINER_CONFIG_TEMPLATE_PATH) / "base_diffusion_flux.toml")

def get_image_training_zip_save_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_tourn.zip")

def get_text_dataset_path(task_id: str) -> str:
    return str(Path(train_cst.CACHE_DATASETS_DIR) / f"{task_id}_train_data.json")

def get_axolotl_dataset_paths(dataset_filename: str) -> tuple[str, str]:
    data_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["data"]) / dataset_filename)
    root_path = str(Path(train_cst.AXOLOTL_DIRECTORIES["root"]) / dataset_filename)
    return data_path, root_path

def get_axolotl_base_config_path(dataset_type) -> str:
    root_dir = Path(train_cst.AXOLOTL_DIRECTORIES["root"])
    if isinstance(dataset_type, (InstructTextDatasetType, DpoDatasetType)):
        return str(root_dir / "base.yml")
    elif isinstance(dataset_type, GrpoDatasetType):
        return str(root_dir / "base_grpo.yml")
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")

def get_text_base_model_path(model_id: str) -> str:
    model_folder = model_id.replace("/", "--")
    return str(Path(train_cst.CACHE_MODELS_DIR) / model_folder)
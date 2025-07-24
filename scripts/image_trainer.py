#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import os
import subprocess
import sys
import shutil
import zipfile
import toml
from enum import Enum

# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import configs.core_constants as cst
import configs.constants as train_cst

class ImageModelType(str, Enum):
    FLUX = "flux"
    SDXL = "sdxl"

def prepare_dataset(
    training_images_zip_path: str,
    training_images_repeat: int,
    instance_prompt: str,
    class_prompt: str,
    job_id: str,
    regularization_images_dir: str = None,
    regularization_images_repeat: int = None,
):
    extraction_dir = f"{cst.DIFFUSION_DATASET_DIR}/tmp/{job_id}/"
    os.makedirs(extraction_dir, exist_ok=True)
    with zipfile.ZipFile(training_images_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    extracted_items = [entry for entry in os.listdir(extraction_dir)]
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(extraction_dir, extracted_items[0])):
        training_images_dir = os.path.join(extraction_dir, extracted_items[0])
    else:
        training_images_dir = extraction_dir

    output_dir = f"{cst.DIFFUSION_DATASET_DIR}/{job_id}/"
    os.makedirs(output_dir, exist_ok=True)

    training_dir = os.path.join(
        output_dir,
        f"img/{training_images_repeat}_{instance_prompt} {class_prompt}",
    )

    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)

    shutil.copytree(training_images_dir, training_dir)

    if regularization_images_dir is not None:
        regularization_dir = os.path.join(
            output_dir,
            f"reg/{regularization_images_repeat}_{class_prompt}",
        )

        if os.path.exists(regularization_dir):
            shutil.rmtree(regularization_dir)
        shutil.copytree(regularization_images_dir, regularization_dir)

    if not os.path.exists(os.path.join(output_dir, "log")):
        os.makedirs(os.path.join(output_dir, "log"))

    if not os.path.exists(os.path.join(output_dir, "model")):
        os.makedirs(os.path.join(output_dir, "model"))

    if os.path.exists(extraction_dir):
        shutil.rmtree(extraction_dir)

    if os.path.exists(training_images_zip_path):
        os.remove(training_images_zip_path)

def save_config_toml(config: dict, config_path: str):
    with open(config_path, "w") as file:
        toml.dump(config, file)


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path

def create_config(task_id, model, model_type, expected_repo_name):
    """Create the diffusion config file"""
    # In Docker environment, adjust paths
    if os.path.exists("/workspace/configs"):
        config_path = "/workspace/configs"
        sdxl_path = f"{config_path}/base_diffusion_sdxl.toml"
        flux_path = f"{config_path}/base_diffusion_flux.toml"
    else:
        sdxl_path = cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL
        flux_path = cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX

    # Load appropriate config template
    if model_type == ImageModelType.SDXL.value:
        with open(sdxl_path, "r") as file:
            config = toml.load(file)
    elif model_type == ImageModelType.FLUX.value:
        with open(flux_path, "r") as file:
            config = toml.load(file)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Update config
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = f"/dataset/images/{task_id}/img/"
    output_dir = f"{train_cst.IMAGE_CONTAINER_SAVE_PATH}{task_id}/{expected_repo_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Save config to file
    config_path = os.path.join("/dataset/configs", f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    training_command = [
        "accelerate", "launch",
        "--dynamo_backend", "no",
        "--dynamo_mode", "default",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "2",
        f"/app/sd-scripts/{model_type}_train_network.py",
        "--config_file", config_path
    ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux"], help="Model type")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    args = parser.parse_args()

    # Create required directories
    os.makedirs("/dataset/configs", exist_ok=True)
    os.makedirs("/dataset/outputs", exist_ok=True)
    os.makedirs("/dataset/images", exist_ok=True)

    model_folder = args.model.replace("/", "--")
    model_path = get_model_path(f"{train_cst.CACHE_PATH}/{args.task_id}/models/{model_folder}")

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
    )

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    # Set DIFFUSION_DATASET_DIR to environment variable if available
    original_dataset_dir = cst.DIFFUSION_DATASET_DIR
    if os.environ.get("DATASET_DIR"):
        cst.DIFFUSION_DATASET_DIR = os.environ.get("DATASET_DIR")

    prepare_dataset(
        training_images_zip_path=f"{train_cst.CACHE_PATH}/{args.task_id}/datasets/{args.task_id}.zip",
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
    )

    # Restore original value
    cst.DIFFUSION_DATASET_DIR = original_dataset_dir

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())

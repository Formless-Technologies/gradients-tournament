#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import torch
import pathlib
import yaml
from transformers import AutoTokenizer
from datetime import datetime, timedelta, timezone
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
from configs.serverless_config_handler import setup_config
from configs.serverless_config_handler import TaskType, FileFormat
from configs.serverless_config_handler import InstructTextDatasetType, DpoDatasetType, GrpoDatasetType


def patch_wandb_symlinks(base_dir: str):
    """Handle WandB symlinks by converting to real files."""
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)
            if os.path.islink(full_path):
                target_path = os.readlink(full_path)
                try:
                    os.unlink(full_path)
                    if os.path.exists(target_path):
                        shutil.copy(target_path, full_path)
                    else:
                        pathlib.Path(full_path).touch()
                except Exception as e:
                    print(f"Symlink patch failed: {e}")


def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass


def run_hpo(config_path: str):
    cmd = [
        "python", 
        "/workspace/training/hpo.py", 
        "--config", config_path
    ]
    
    # Run the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)

    print("HPO subprocess completed successfully.", flush=True)


async def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", help="Number of hours to complete the training")
    args = parser.parse_args()

    # Setup Datasets
    try:
        dataset_type_dict = json.loads(args.dataset_type)

        if args.task_type == TaskType.DPOTASK.value:
            dataset_type = DpoDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.INSTRUCTTEXTTASK.value:
            dataset_type = InstructTextDatasetType(**dataset_type_dict)
        elif args.task_type == TaskType.GRPOTASK.value:
            dataset_type = GrpoDatasetType(**dataset_type_dict)
        else:
            sys.exit(f"Unsupported task type: {args.task_type}")
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = f"/cache/datasets/{args.task_id}_train_data.json"

    # Setup correct output directories
    output_dir = f"/app/checkpoints/{args.task_id}/{args.expected_repo_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Calculate required finish time
    required_finish_time_dt  = datetime.now(timezone.utc) + timedelta(hours=int(args.hours_to_complete)) - timedelta(minutes=15) # Add 15 minute leeway for docker build and setup
    required_finish_time = required_finish_time_dt.isoformat()

    # Build Config File
    config_path = f"/workspace/configs/{args.task_id}.yml"
    setup_config(dataset_path, args.model, dataset_type, args.task_id, args.expected_repo_name, required_finish_time)

    # Run HPO and write best config to _best.yml
    #run_hpo(config_path)

    # Start Training
    path_to_train_file = "/workspace/training/train.py"
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        training_command = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
            path_to_train_file,
            "--config", str(config_path),
        ]

    else:
        training_command = [
            "accelerate", "launch",
            "--multi_gpu",
            "--mixed_precision", "bf16",
            "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
            path_to_train_file,
            "--config", str(config_path),
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


    WANDB_LOGS_DIR = "/cache/wandb_logs"
    patch_wandb_symlinks(WANDB_LOGS_DIR)
    patch_model_metadata(output_dir, args.model)


if __name__ == "__main__":
    asyncio.run(main())
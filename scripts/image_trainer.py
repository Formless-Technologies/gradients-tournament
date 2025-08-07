#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import os
import subprocess
import sys
import configs.core_constants as cst
import configs.trainer_constants as train_cst
import configs.training_paths as train_paths
from configs.training_paths import ImageModelType
import shutil
import zipfile
import toml
import time


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)


def get_available_vram_gb() -> float:
    """Detect available GPU VRAM in GB with multiple fallbacks."""
    # Try torch
    try:
        import torch  # local import to avoid hard dependency when CPU-only
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024 ** 3), 1)
    except Exception:
        pass
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        out = (result.stdout or "").strip().splitlines()
        if out:
            return round(float(out[0]), 1)
    except Exception:
        pass
    # Fallback safe default
    return 12.0


def parse_resolution(reso: str) -> tuple[int, int]:
    try:
        w, h = reso.split(",")
        return int(w.strip()), int(h.strip())
    except Exception:
        return (1024, 1024)


def choose_batch_settings(vram_gb: float, resolution: tuple[int, int], model_type: str) -> tuple[int, int]:
    """Heuristic mapping from VRAM and resolution to (train_batch_size, gradient_accumulation_steps)."""
    max_dim = max(resolution)
    if vram_gb >= 39:
        bs = 48
    elif vram_gb >= 23:
        bs = 32
    elif vram_gb >= 15:
        bs = 24
    elif vram_gb >= 11:
        bs = 16
    elif vram_gb >= 9:
        bs = 8
    else:
        bs = 2

    if max_dim > 1024:
        bs = max(1, bs - 1)

    return int(bs), 1


def adjust_config_for_memory(config: dict, model_type: str, vram_gb: float) -> dict:
    """Mutate config with VRAM-aware settings; return the config."""
    reso_str = config.get("resolution", "1024,1024")
    reso = parse_resolution(reso_str)
    bs, gas = choose_batch_settings(vram_gb, reso, model_type)
    config["train_batch_size"] = int(bs)
    config["gradient_accumulation_steps"] = int(gas)

    # Low-VRAM resolution/bucket fallbacks
    if vram_gb < 10:
        config["resolution"] = "768,768"
        config["max_bucket_reso"] = min(int(config.get("max_bucket_reso", 1024)), 1024)
        config["bucket_reso_steps"] = min(int(config.get("bucket_reso_steps", 64)), 32)
    if vram_gb < 8:
        config["resolution"] = "640,640"
        config["max_bucket_reso"] = min(int(config.get("max_bucket_reso", 896)), 896)

    return config


def update_config_file_inplace(config_path: str, updater_fn):
    """Load toml config, apply updater, write back, and return new dict."""
    with open(config_path, "r") as f:
        cfg = toml.load(f)
    cfg = updater_fn(cfg)
    save_config_toml(cfg, config_path)
    return cfg


def _line_has_oom(s: str) -> bool:
    s = s.lower()
    return ("out of memory" in s) or ("cuda oom" in s) or ("cuda error: out of memory" in s)


def prepare_dataset(
    training_images_zip_path: str,
    training_images_repeat: int,
    instance_prompt: str,
    class_prompt: str,
    job_id: str,
    regularization_images_dir: str = None,
    regularization_images_repeat: int = None,
    output_dir: str = None,
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

    if output_dir is None:
        output_dir = f"{cst.DIFFUSION_DATASET_DIR}/{job_id}/"
    else:
        output_dir = f"{output_dir}/{job_id}/"
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

    if os.path.exists(training_images_zip_path) and "tourn" not in os.path.basename(training_images_zip_path):
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
    config_template_path = train_paths.get_image_training_config_template_path(model_type)

    with open(config_template_path, "r") as file:
        config = toml.load(file)

    # Update basic paths
    config["pretrained_model_name_or_path"] = model
    config["train_data_dir"] = train_paths.get_image_training_images_dir(task_id)
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Detect VRAM and adjust memory-related settings
    vram_gb = get_available_vram_gb()
    print(f"[auto-config] Detected VRAM ≈ {vram_gb} GB", flush=True)
    config = adjust_config_for_memory(config, model_type, vram_gb)
    print(f"[auto-config] Using train_batch_size={config.get('train_batch_size')} "
          f"gradient_accumulation_steps={config.get('gradient_accumulation_steps')} "
          f"resolution={config.get('resolution')} max_bucket_reso={config.get('max_bucket_reso')}", flush=True)

    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    max_retries = 3
    attempt = 0

    while attempt <= max_retries:
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
            print(f"Starting training subprocess (attempt {attempt + 1}/{max_retries + 1})...\n", flush=True)
            process = subprocess.Popen(
                training_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            lines = []
            oom_seen = False
            for line in process.stdout:
                print(line, end="", flush=True)
                lines.append(line)
                if not oom_seen and _line_has_oom(line):
                    oom_seen = True

            return_code = process.wait()
            if return_code == 0 and not oom_seen:
                print("Training subprocess completed successfully.", flush=True)
                return

            # Handle OOM and retry by shrinking memory footprint
            combined = "".join(lines)
            if oom_seen or _line_has_oom(combined):
                attempt += 1
                if attempt > max_retries:
                    print("OOM persisted after maximum retries.", flush=True)
                    raise subprocess.CalledProcessError(return_code or 1, training_command)

                def updater(cfg: dict):
                    bs = int(cfg.get("train_batch_size", 1))
                    gas = int(cfg.get("gradient_accumulation_steps", 1))
                    if bs > 1:
                        new_bs = max(1, bs // 2)
                        print(f"[retry {attempt}] Reducing train_batch_size {bs} -> {new_bs}", flush=True)
                        cfg["train_batch_size"] = new_bs
                    else:
                        new_gas = gas + 1
                        print(f"[retry {attempt}] Increasing gradient_accumulation_steps {gas} -> {new_gas}", flush=True)
                        cfg["gradient_accumulation_steps"] = new_gas
                        # Step down resolution gradually
                        reso = cfg.get("resolution", "1024,1024")
                        w, h = parse_resolution(reso)
                        if max(w, h) > 768:
                            cfg["resolution"] = "768,768"
                            cfg["max_bucket_reso"] = min(int(cfg.get("max_bucket_reso", 1024)), 1024)
                            cfg["bucket_reso_steps"] = min(int(cfg.get("bucket_reso_steps", 64)), 32)
                            print(f"[retry {attempt}] Reducing resolution to {cfg['resolution']} and max_bucket_reso={cfg['max_bucket_reso']}", flush=True)
                        elif max(w, h) > 640:
                            cfg["resolution"] = "640,640"
                            cfg["max_bucket_reso"] = min(int(cfg.get("max_bucket_reso", 896)), 896)
                            print(f"[retry {attempt}] Reducing resolution to {cfg['resolution']} and max_bucket_reso={cfg['max_bucket_reso']}", flush=True)
                        else:
                            cfg["resolution"] = "512,512"
                            cfg["max_bucket_reso"] = min(int(cfg.get("max_bucket_reso", 768)), 768)
                            print(f"[retry {attempt}] Reducing resolution to {cfg['resolution']} and max_bucket_reso={cfg['max_bucket_reso']}", flush=True)
                    return cfg

                update_config_file_inplace(config_path, updater)
                print("[retry] Updated config to mitigate OOM; restarting training...", flush=True)
                time.sleep(1.0)
                continue

            # If non-OOM error, raise as before
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, training_command)

            print("Training subprocess completed successfully.", flush=True)
            return

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
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model_type,
        args.expected_repo_name,
    )

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
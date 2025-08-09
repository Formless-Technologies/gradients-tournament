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
import re
import gc
import time
import psutil
from transformers import AutoTokenizer
from datetime import datetime, timedelta, timezone
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
from configs.serverless_config_handler import setup_config, add_throughput_information, modify_model_location
from configs.serverless_config_handler import TaskType, FileFormat
from configs.serverless_config_handler import InstructTextDatasetType, DpoDatasetType, GrpoDatasetType

TESTING = False

DO_SFT_PRETRAIN = True
SFT_PRETRAIN_TIME = 30
DO_THROUGHPUT_PROBE = True
THROUGHPUT_PROBE_TIME = 5
DO_HPO = True
GPU_CLEANUP_WAIT_TIME = 5

if TESTING:
    DO_SFT_PRETRAIN = False
    SFT_PRETRAIN_TIME = 1
    DO_THROUGHPUT_PROBE = True
    THROUGHPUT_PROBE_TIME = 1
    DO_HPO = False

def cleanup_resources():
    """
    Force cleanup of GPU/CPU memory and zombie child processes.
    Mirrors the logic used in HPO to stabilize repeated launches.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        time.sleep(2)  # give processes time to terminate
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        print(f"Resource cleanup error: {e}", flush=True)

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
    """
    Launch the HPO pipeline. If it produces a _best.yml config, return that path.
    If no best config is produced, return None. Raises on subprocess failure.
    """
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

    # Check for optimised config emitted by HPO script
    best_cfg_path = config_path.replace(".yml", "_best.yml")
    if os.path.exists(best_cfg_path):
        print(f"Found optimised config: {best_cfg_path}\n", flush=True)
        return best_cfg_path
    else:
        print("No optimised _best.yml found; will fall back to base config.", flush=True)
        return None

def run_probe(base_config_path: str, minutes: int = 5):
    """
    Run a short time-limited training to estimate steps-per-minute using the base YAML config.
    The throughput is computed over the LAST 80% of the probe duration to reduce warmup bias.

    This implementation parses tqdm progress lines (e.g.,
      "  0%|          | 130/1000000000 [04:48<...,  2.19s/it]")
    to extract (step, elapsed_time) pairs, which are printed even when HF logging lines
    are buffered until the end.

    Args:
        base_config_path: Path to the base YAML config to use for the probe.
        minutes: Duration of the probe (default 5 minutes).

    Returns:
        steps_per_minute: float
    """
    # Load base config and build a probe variant
    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    probe_cfg = dict(cfg)

    # Make time the limiter
    now = datetime.now(timezone.utc)
    probe_cfg["required_finish_time"] = (now + timedelta(minutes=minutes)).isoformat()

    # Isolate output directory to avoid interfering with real checkpoints
    out_dir = probe_cfg.get("output_dir", f"/app/checkpoints/{uuid.uuid4().hex}")
    probe_cfg["output_dir"] = os.path.join(out_dir, "probe")

    # Avoid checkpointing overhead during probe; increase step cap; disable eval; increase logging frequency
    probe_cfg["max_steps"] = int(1e9)
    probe_cfg["save_steps"] = int(1e9)
    probe_cfg["eval_steps"] = int(1e9)
    probe_cfg["save_total_limit"] = 1
    probe_cfg["main_training_run"] = False
    try:
        probe_cfg["logging_steps"] = min(int(probe_cfg.get("logging_steps", 10)), 10)
    except Exception:
        probe_cfg["logging_steps"] = 10

    # Write the probe config alongside the base config
    probe_cfg_path = base_config_path.replace(".yml", "_probe.yml")
    with open(probe_cfg_path, "w") as f:
        yaml.safe_dump(probe_cfg, f)

    # Build the accelerate command (mirror scripts/text_trainer launching approach)
    path_to_train_file = "/workspace/training/train.py"
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        training_command = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_processes", str(num_gpus),
            path_to_train_file,
            "--config", str(probe_cfg_path),
        ]
    else:
        training_command = [
            "accelerate", "launch",
            "--multi_gpu",
            "--mixed_precision", "bf16",
            "--num_processes", str(num_gpus),
            path_to_train_file,
            "--config", str(probe_cfg_path),
        ]

    # Regex for tqdm lines: capture current step and elapsed time inside [...]
    # Example matched: "  0%| ... | 130/1000000000 [04:48<...,  2.19s/it]"
    tqdm_re = re.compile(r"\s*\d+%.*\|\s*(\d+)/(\d+).*?\[([0-9:]+)<")

    def _elapsed_to_seconds(token: str) -> float | None:
        # token is "MM:SS" or "HH:MM:SS"
        parts = token.strip().split(":")
        try:
            parts = [int(p) for p in parts]
        except Exception:
            return None
        if len(parts) == 2:
            mm, ss = parts
            return mm * 60 + ss
        if len(parts) == 3:
            hh, mm, ss = parts
            return hh * 3600 + mm * 60 + ss
        return None

    # Keep last observed (elapsed_seconds, step) and baseline after warmup
    last_step = None
    last_elapsed_sec = None
    baseline_step = None
    baseline_elapsed_sec = None

    warmup_seconds = max(0.0, minutes * 60.0 * 0.2)  # first 20% is warmup

    process = subprocess.Popen(
        training_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return_code = 0
    try:
        for line in process.stdout:
            print(line, end="", flush=True)

            # Parse tqdm progress
            m = tqdm_re.search(line)
            if m:
                try:
                    cur_step = int(m.group(1))
                    elapsed_tok = m.group(3)
                    elapsed_sec = _elapsed_to_seconds(elapsed_tok)
                except Exception:
                    cur_step, elapsed_sec = None, None

                if elapsed_sec is not None and cur_step is not None:
                    last_step = cur_step
                    last_elapsed_sec = elapsed_sec
                    if baseline_step is None and elapsed_sec >= warmup_seconds:
                        baseline_step = cur_step
                        baseline_elapsed_sec = elapsed_sec

        return_code = process.wait()
    finally:
        # Ensure process is terminated and resources are cleaned up (like HPO)
        try:
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except Exception:
                    process.kill()
        except Exception:
            pass

        cleanup_resources()
        time.sleep(GPU_CLEANUP_WAIT_TIME)

    # Compute throughput over the last 80% (post-warmup)
    steps_per_minute = None
    if last_step is not None and last_elapsed_sec is not None:
        if baseline_step is None or baseline_elapsed_sec is None:
            # No sample past warmup, use earliest we have as baseline
            baseline_step = baseline_step if baseline_step is not None else 0
            baseline_elapsed_sec = baseline_elapsed_sec if baseline_elapsed_sec is not None else 0.0

        time_delta_min = max(1e-6, (last_elapsed_sec - baseline_elapsed_sec) / 60.0)
        step_delta = max(0, int(last_step) - int(baseline_step))
        steps_per_minute = float(step_delta) / time_delta_min

    # As a last resort, return 0.0 to avoid crashing callers
    if steps_per_minute is None:
        steps_per_minute = 0.0

    if return_code != 0:
        print(f"Probe subprocess exited with code {return_code} (continuing with collected metrics).", flush=True)

    return steps_per_minute

def run_sft_pretrain(base_config_path: str, minutes: int = 15, max_steps: int = 1000) -> str | None:
    """
    Run a short SFT pretraining pass when the primary task is DPO.

    - Builds a derivative SFT config from the base YAML (DPO) with:
      * minimal step/time budget
      * separate output_dir subfolder "sft_pretrain"
      * dataset field mapping inferred from DPO config

    Returns:
      Path to the final saved model
    """
    # Load the existing (DPO) config
    with open(base_config_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    if base_cfg.get("rl") not in ("dpo", "grpo"):
        print("SFT pretrain skipped: base config is not DPO.", flush=True)
        return None

    # Build SFT config derivative
    sft_cfg = dict(base_cfg)

    # Compute short time budget via required_finish_time
    now = datetime.now(timezone.utc)
    sft_cfg["required_finish_time"] = (now + timedelta(minutes=max(1, int(minutes)))).isoformat()

    # Training caps for a short warmup
    sft_cfg["max_steps"] = int(max_steps)

    sft_cfg["save_steps"] = int(1e9)   # avoid checkpoint overhead
    sft_cfg["eval_steps"] = int(1e9)   # avoid eval overhead
    sft_cfg["save_total_limit"] = 1
    sft_cfg["main_training_run"] = False
    sft_cfg["logging_steps"] = 10

    # set a reasonable LR for SFT
    sft_cfg["sft_pretrain"] = True
    sft_cfg["learning_rate"] = 1e-4

    # Separate output dir to avoid interfering with main run
    base_out = "/app/checkpoints"
    sft_cfg["output_dir"] = os.path.join(base_out, "sft_pretrain")

    # Write SFT config
    sft_cfg_path = base_config_path.replace(".yml", "_sft_pretrain.yml")
    with open(sft_cfg_path, "w") as f:
        yaml.safe_dump(sft_cfg, f)

    # Launch a short SFT training
    path_to_train_file = "/workspace/training/train.py"
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        cmd = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_processes", str(num_gpus),
            path_to_train_file,
            "--config", str(sft_cfg_path),
        ]
    else:
        cmd = [
            "accelerate", "launch",
            "--multi_gpu",
            "--mixed_precision", "bf16",
            "--num_processes", str(num_gpus),
            path_to_train_file,
            "--config", str(sft_cfg_path),
        ]

    print("--- STARTING SHORT SFT PRETRAIN ---\n", flush=True)
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        rc = proc.wait()
        if rc != 0:
            print(f"SFT pretrain exited with code {rc} (continuing).", flush=True)
    except Exception as e:
        print(f"SFT pretrain failed: {e}", flush=True)
    finally:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        cleanup_resources()
        time.sleep(GPU_CLEANUP_WAIT_TIME)

    print("--- SFT PRETRAIN FINISHED ---\n", flush=True)
    return sft_cfg["output_dir"]
    

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
    config = setup_config(dataset_path, args.model, dataset_type, args.task_id, args.expected_repo_name, required_finish_time)




    # SFT PRETRAINING STEP FOR DPO ==========================================
    if config['rl'] == "dpo" and DO_SFT_PRETRAIN:
        try:
            new_model_location = run_sft_pretrain(config_path, minutes=SFT_PRETRAIN_TIME)
            modify_model_location(config_path, new_model_location)
        except Exception as e:
            print(f"SFT pretrain encountered an error and will be skipped: {e}", flush=True)
    time.sleep(2)

    # THROUGHPUT PROBE =======================================================
    if DO_THROUGHPUT_PROBE:
        # Run throughput probe to determine our steps per minute and adjust max_steps and warmup ratio
        print("--- STARTING THROUGHPUT PROBE ---\n", flush=True)
        try:
            steps_per_minute = run_probe(config_path, minutes=THROUGHPUT_PROBE_TIME)
            if steps_per_minute != 0.0:
                print(f"THROUGHPUT FOUND: {steps_per_minute} spm\n", flush=True)
                add_throughput_information(config_path, steps_per_minute)
            else:
                print(f"THROUGHPUT NOT FOUND DURING PROBE\n", flush=True)
                add_throughput_information(config_path, 0.0)
        except Exception as e:
            print(f"Throughput Probe encountered an error and will be skipped: {e}", flush=True)
            add_throughput_information(config_path, 0.0)
    else:
        add_throughput_information(config_path, 0.0)
    
    time.sleep(2)

    # HPO STEP ================================================================
    selected_config_path = config_path
    if DO_HPO:
        # Try HPO; if it succeeds and produces a _best.yml, use it; otherwise fall back to base.
        print("--- STARTING HPO PIPELINE ---\n", flush=True)
        try:
            best_cfg_path = run_hpo(config_path)
            if best_cfg_path:
                selected_config_path = best_cfg_path
                print(f"Using HPO-optimized config: {best_cfg_path}\n", flush=True)
            else:
                print("HPO completed but no _best.yml found; using base config.", flush=True)
        except Exception as e:
            print(f"HPO failed: {e}. Falling back to base config.\n", flush=True)

    time.sleep(2)


    # FULL TRAINING RUN =========================================================
    print("--- STARTING FULL TRAINING RUN ---\n", flush=True)

    # Start Training
    path_to_train_file = "/workspace/training/train.py"
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        training_command = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
            path_to_train_file,
            "--config", str(selected_config_path),
        ]

    else:
        training_command = [
            "accelerate", "launch",
            "--multi_gpu",
            "--mixed_precision", "bf16",
            "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
            path_to_train_file,
            "--config", str(selected_config_path),
        ]
    try:
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
#!/usr/bin/env python3
"""
hpo_optuna.py  –  1‑hour Optuna sweep → full training (multi‑GPU compatible)
--------------------------------------------------------------------------
"""
from __future__ import annotations
import argparse, copy, json, os, re, shutil, subprocess, tempfile, uuid, time
from pathlib import Path
import yaml, optuna
from datetime import datetime, timedelta, timezone
from optuna.pruners import HyperbandPruner
from optuna.storages import RDBStorage
import gc
import torch
import psutil
from contextlib import contextmanager


MAX_TRIALS_TO_RUN = 10
PERCENT_TIME_FOR_HPO = 0.25
MAX_MINUTES_PER_TRIAL = 15
GPU_CLEANUP_WAIT_TIME = 5  # seconds to wait for GPU cleanup


# ╭──────────────────────── Hyper‑parameter space ───────────────────────────╮
def sample_space(trial: optuna.Trial, cfg: dict) -> dict:
    # Invariant Params
    params = {
        "optimizer": trial.suggest_categorical("optimizer", ["lion_8bit"]),
    }

    # SFT Params
    if cfg["rl"] == "sft":
        params |= {
            "learning_rate": trial.suggest_float("learning_rate", 5e-6, 2e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "use_neftune": trial.suggest_categorical("use_neftune", [True, False]),
        }
    # DPO Params
    if cfg["rl"] == "dpo":
        params |= {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
            "beta": trial.suggest_float("beta", 0.02, 0.5, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        }
    # GRPO Params
    elif cfg["rl"] == "grpo":
        params |= {
            "learning_rate": trial.suggest_float("learning_rate", 5e-7, 1e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.05),
            "beta": trial.suggest_float("beta", 0.01, 0.1, log=True),
        }
        
    # LORA Params
    if cfg["adapter"] == "lora":
        params |= {
            "lora_r": trial.suggest_int("lora_r", 16, 128, step=16),
            "lora_alpha": trial.suggest_int("lora_alpha", 16, 256, step=16),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.1),
        }

    return params
# ╰──────────────────────────────────────────────────────────────────────────╯

_EVAL_RE = re.compile(r"eval_loss[^0-9]*([0-9]+\.[0-9]+)")
def loss_from_stdout(stdout: str) -> float | None:
    matches = _EVAL_RE.findall(stdout)
    return float(matches[-1]) if matches else None

# ── Stability utilities ─────────────────────────────────────────────────────
def cleanup_resources():
    """Force cleanup of GPU memory and other resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        # Kill any zombie processes
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        time.sleep(2)  # Give processes time to terminate
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception as e:
        print(f"Resource cleanup error: {e}")



# ╭──────────────────────── Objective (single trial) ─────────────────────────╮
def objective(
    trial: optuna.Trial,
    base_cfg: dict,
    hpo_project: str,
    study_name: str,
    storage_path: str,
    time_when_hpo_finished: datetime
) -> float:
    """Run a single trial with enhanced stability and retry logic"""
    
    # Check if we have enough time left
    time_left = (time_when_hpo_finished - datetime.now(timezone.utc)).total_seconds()
    if time_left < 60*10:  # Less than 10 minutes left
        print("Not enough time left for new trial")
        raise optuna.exceptions.OptunaError("Time limit reached")
    
    cfg = copy.deepcopy(base_cfg)
    trial_params = sample_space(trial, cfg)
    cfg.update(trial_params)

    trial_id = f"trial_{trial.number}"
    out_dir = f"./hpo_runs/{trial_id}"
    cfg |= {
        "main_training_run": False,
        "output_dir": str(out_dir),
        "save_steps": 10000,
        "logging_steps": 10,  # More frequent logging for monitoring
        "save_total_limit": 1,  # Save disk space
        "load_best_model_at_end": False,  # Speed up for HPO
    }

    cfg['required_finish_time'] = (
        datetime.now(timezone.utc) + timedelta(minutes=MAX_MINUTES_PER_TRIAL)
    ).isoformat()


    tmp_cfg = Path(tempfile.mkdtemp()) / f"{trial_id}.yml"
    with tmp_cfg.open("w") as f:
        yaml.safe_dump(cfg, f)

    path_to_train_file = "/workspace/training/train.py"
    num_gpus = torch.cuda.device_count()

    if num_gpus == 1:
        training_command = [
            "accelerate", "launch",
            "--mixed_precision", "bf16",
            "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
            path_to_train_file,
            "--config", str(tmp_cfg),
        ]

    else:
        training_command = [
            "accelerate", "launch",
            "--multi_gpu",
            "--mixed_precision", "bf16",
            "--num_processes", str(torch.cuda.device_count()),  # Explicit GPU count
            path_to_train_file,
            "--config", str(tmp_cfg),
        ]

    try:
        print(F"STARTING TRIAL {trial.number} with params: {trial_params} \n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Capture stdout, stream metrics, and report to Optuna for pruning
        stdout_lines = []
        eval_counter = 0
        max_rungs = 10
        for line in process.stdout:
            print(line, end="", flush=True)
            stdout_lines.append(line)

            # Look for eval_loss on this line
            m = _EVAL_RE.search(line)
            if m:
                try:
                    eval_loss = float(m.group(1))
                except Exception:
                    continue

                eval_counter += 1
                # Report rung index as resource step for Hyperband
                try:
                    print(f"Reporting Eval Loss to Optuna: {eval_loss}\n")
                    trial.report(eval_loss, step=eval_counter)
                except Exception as _e:
                    print(f"Optuna report failed at rung {eval_counter}: {_e}\n")

                # Check if we should prune
                try:
                    if trial.should_prune():
                        print(f"Pruning trial {trial.number} at rung {eval_counter} (eval_loss={eval_loss})\n")
                        try:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except Exception:
                                process.kill()
                        finally:
                            cleanup_resources()
                        raise optuna.exceptions.TrialPruned(
                            f"Pruned at rung {eval_counter} with eval_loss={eval_loss}\n"
                        )
                except Exception as _e:
                    print(f"Optuna prune check failed at rung {eval_counter}: {_e}\n")

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)
        
        # Extract eval_loss from captured stdout
        full_stdout = "".join(stdout_lines)
        eval_loss = loss_from_stdout(full_stdout)
        
        if eval_loss is None:
            print("Could not extract eval_loss from stdout, using fallback value")
            return float("inf") if cfg["rl"] != "grpo" else float("-inf")
        
        print(f"Trial {trial.number} completed with eval_loss: {eval_loss}\n")
        return eval_loss

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}\n", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}\n", flush=True)
        try:
            tail_lines = stdout_lines[-10:] if 'stdout_lines' in locals() and stdout_lines else []
            if tail_lines:
                print("Last 10 lines of subprocess output:", flush=True)
                print("".join(tail_lines), end="", flush=True)
            else:
                print("No captured subprocess output available.", flush=True)
        except Exception as tail_err:
            print(f"Failed to print subprocess tail: {tail_err}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

# ╰──────────────────────────────────────────────────────────────────────────╯

# ╭──────────────────────── Run Optuna sweep ─────────────────────────────────╮
def run_optuna(base_cfg_path: str) -> dict:
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    study_name   = base_cfg.get("task_id", "optuna")
    hpo_root     = Path(base_cfg.get("output_root", "./hpo_runs")) / study_name
    hpo_root.mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{hpo_root / 'hpo.db'}"
    base_project = "Gradients"
    hpo_project  = f"{base_project}-HPO-Trials"
    
    # Use more robust storage settings
    storage = RDBStorage(
        url=storage_path, 
        engine_kwargs={
            "connect_args": {
                "timeout": 60,  # Increased timeout
                "check_same_thread": False  # Allow multi-threading
            }, 
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 10
        }
    )

    if base_cfg["rl"] == "grpo":
        direction = "maximize"
    else:
        direction = "minimize"

    # Create study with more aggressive pruning for stability
    study = optuna.create_study(
        direction=direction,
        study_name=base_cfg["task_id"],
        load_if_exists=True,  # Allow resuming interrupted studies
        storage=storage,
        pruner=HyperbandPruner(
            min_resource=1,  # Allow earliest pruning at first rung
            max_resource=10,
            reduction_factor=3
        )
    )
    
    # Calculate time budget
    time_remaining = datetime.fromisoformat(base_cfg['required_finish_time']) - datetime.now(timezone.utc)
    seconds_remaining = max(0.0, time_remaining.total_seconds() * PERCENT_TIME_FOR_HPO)
    time_when_hpo_finished = datetime.now(timezone.utc) + timedelta(seconds=seconds_remaining)

    print(f"Time allocated to HPO Search: {seconds_remaining/3600:.2f}h\n")
    
    # Run optimization with exception handling
    try:
        study.optimize(
            lambda t: objective(t, base_cfg, hpo_project, study_name, storage_path, time_when_hpo_finished),
            timeout=int(seconds_remaining),
            n_trials=MAX_TRIALS_TO_RUN,
            show_progress_bar=True,
            catch=(Exception,),  # Catch all exceptions to prevent study crash
            callbacks=[lambda study, trial: cleanup_resources()]  # Cleanup after each trial
        )
    except Exception as e:
        print(f"Study optimization failed: {e}\n")
        # Try to get best value so far
        if len(study.trials) > 0:
            print("Attempting to use best trial found so far...")
        else:
            raise

    # Final results
    if study.best_trial:
        print(f"HPO finished – best eval_loss {study.best_value:.5f} with params {study.best_params}\n")
            
        return study.best_params
    else:
        raise ValueError("No successful trials completed")
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭──────────────────── Write optimised YAML  ───────────────╮
def write_best_cfg(base_cfg: str, best: dict) -> str:
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg.update(best)
    
    opt_path = base_cfg.replace(".yml", "_best.yml")
    with open(opt_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"💾  Wrote optimised config → {opt_path}")
    return opt_path


# ╭──────────────────────────── CLI entry‑point ──────────────────────────────╮
def main():
    ap = argparse.ArgumentParser(description="HPO then full training")
    ap.add_argument("--config", required=True, help="Base YAML config file")
    ap.add_argument("--resume", action="store_true", help="Resume interrupted HPO study")
    args = ap.parse_args()
    
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    if base_cfg['testing']:
        global MAX_TRIALS_TO_RUN
        global MAX_MINUTES_PER_TRIAL
        MAX_TRIALS_TO_RUN = 2
        MAX_MINUTES_PER_TRIAL = 2
    
    try:
        best_params = run_optuna(args.config)
        optimised_cfg = write_best_cfg(args.config, best_params)
        cleanup_resources()
    except Exception as e:
        print(f"HPO pipeline failed: {e}\n")
        raise

if __name__ == "__main__":
    main()
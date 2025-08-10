import aiohttp
from datasets import load_dataset
from transformers import AutoTokenizer
import hashlib

def load_tokenizer(model_name: str, config: dict):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tok


def load_sft_datasets(config: dict):
    """
    Return (train_ds, eval_ds)
    If config["val_set_size"] is 0 → eval_ds is None.
    """
    # Load **only one** split so we always get a Dataset, never a DatasetDict
    ds_train = load_dataset(
        "json",
        data_files=config["datasets"][0]["path"],
        split="train",          # guarantees Dataset, not DatasetDict
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    def combine_prompt(example):
        # Handles the case when "input" (the context) may be empty
        if example["input"]:
            prompt = f"{example['prompt']}\n{example['input']}"
        else:
            prompt = example["prompt"]
        example["prompt"] = prompt
        return example                  
    
    if config["datasets"][0]["field_input"] is not None:
        # Standardise column names
        ds_train = ds_train.rename_columns({
            config["datasets"][0]["field_instruction"]:   "prompt",
            config["datasets"][0]["field_input"]:   "input",
            config["datasets"][0]["field_output"]:   "completion",
        })
        ds_train = ds_train.map(combine_prompt)
    else:
        # Standardise column names
        ds_train = ds_train.rename_columns({
            config["datasets"][0]["field_instruction"]:   "prompt",
            config["datasets"][0]["field_output"]:   "completion",
        })

    orig_len = len(ds_train)
    _seen = set()                                           # lives outside the lambdas

    ds_train = (
        ds_train
        # 1️⃣ add a stable hash column
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "")
                + ex.get("completion", "")
                ).encode()
            ).hexdigest()},
            num_proc=8
        )
        # 2️⃣ keep only the first row for every hash
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]),
                num_proc=1)                    # single-proc so `_seen` is shared
        # 3️⃣ drop the helper column
        .remove_columns("__hash")
    )
    dup_removed = orig_len - len(ds_train)
    print(f"Removed {dup_removed:,} duplicate rows "
        f"({dup_removed / orig_len:.2%} of original).")
    
    # Optional random split
    val_size = config.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None
    
    return train_ds, eval_ds


def load_dpo_datasets(config: dict):
    """
    Return (train_ds, eval_ds) ready for TRL‑DPO.
    If config["val_set_size"] is 0 → eval_ds is None.
    """
    # Load dataset (guarantees a Dataset, not a DatasetDict)
    ds_train = load_dataset(
        "json",
        data_files=config["datasets"][0]["path"],
        split="train",
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    # Standardise column names
    ds_train = ds_train.rename_columns({
        config["datasets"][0]["field_prompt"]:   "prompt",
        config["datasets"][0]["field_chosen"]:   "chosen",
        config["datasets"][0]["field_rejected"]: "rejected",
    })

    orig_len = len(ds_train)
    _seen = set()                                           # lives outside the lambdas

    ds_train = (
        ds_train
        # 1️⃣ add a stable hash column
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "")
                + ex.get("chosen", "")        # use "rejected" if that’s your column name
                + ex.get("rejected", "")
                ).encode()
            ).hexdigest()},
            num_proc=8
        )
        # 2️⃣ keep only the first row for every hash
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]),
                num_proc=1)                    # single-proc so `_seen` is shared
        # 3️⃣ drop the helper column
        .remove_columns("__hash")
    )
    dup_removed = orig_len - len(ds_train)
    print(f"Removed {dup_removed:,} duplicate rows "
        f"({dup_removed / orig_len:.2%} of original).")

    # Optional random split
    val_size = config.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds


def load_grpo_datasets(config: dict):
    """
    Return (train_ds, eval_ds) ready for TRL‑GRPO.
    If config["val_set_size"] is 0 → eval_ds is None.
    """
    # Load **only one** split so we always get a Dataset, never a DatasetDict
    ds_train = load_dataset(
        "json",
        data_files=config["datasets"][0]["path"],
        split="train",          # guarantees Dataset, not DatasetDict
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    # Standardise column names
    ds_train = ds_train.rename_columns({
        config["datasets"][0]["field_prompt"]:   "prompt",
    })

    if config["datasets"][0]["extra_column"] is not None:
        ds_train = ds_train.rename_columns({
            config["datasets"][0]["extra_column"]:   "extra_data",
        })

    orig_len = len(ds_train)
    _seen = set()                                           # lives outside the lambdas

    ds_train = (
        ds_train
        # 1️⃣ add a stable hash column
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "")
                ).encode()
            ).hexdigest()},
            num_proc=8
        )
        # 2️⃣ keep only the first row for every hash
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]),
                num_proc=1)                    # single-proc so `_seen` is shared
        # 3️⃣ drop the helper column
        .remove_columns("__hash")
    )

    dup_removed = orig_len - len(ds_train)
    print(f"Removed {dup_removed:,} duplicate rows "
        f"({dup_removed / orig_len:.2%} of original).")


    # Optional random split
    val_size = config.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds


def load_sft_pretrain_datasets(config: dict):
    """
    Build an SFT-compatible dataset from a DPO dataset.

    Behavior:
    - When rl == "dpo": maps
        prompt   <- datasets[0]["field_prompt"]
        completion <- datasets[0]["field_chosen"]

    Returns:
        (train_ds, eval_ds) where columns are standardized to ["prompt", "completion"].
        If config["val_set_size"] is 0, eval_ds is None.
    """
    # Always load a single split to ensure a Dataset (not DatasetDict)
    ds_train = load_dataset(
        "json",
        data_files=config["datasets"][0]["path"],
        split="train",
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=1800)}}
    )

    rl_type = config["rl"]

    if rl_type == "dpo":
        # Standardize to SFT-style columns
        ds_train = ds_train.rename_columns({
            config["datasets"][0]["field_prompt"]: "prompt",
            config["datasets"][0]["field_chosen"]: "completion",
        })

    # Deduplicate on (prompt + completion) just like SFT
    orig_len = len(ds_train)
    _seen = set()

    ds_train = (
        ds_train
        .map(
            lambda ex: {"__hash": hashlib.md5(
                (ex.get("prompt", "") + ex.get("completion", "")).encode()
            ).hexdigest()},
            num_proc=8
        )
        .filter(lambda ex: ex["__hash"] not in _seen and not _seen.add(ex["__hash"]), num_proc=1)
        .remove_columns("__hash")
    )

    dup_removed = orig_len - len(ds_train)
    print(
        f"Removed {dup_removed:,} duplicate rows "
        f"({dup_removed / orig_len:.2%} of original)."
    )

    # Optional random split
    val_size = config.get("val_set_size", 0)
    if val_size:
        split = ds_train.train_test_split(test_size=val_size, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = ds_train, None

    return train_ds, eval_ds

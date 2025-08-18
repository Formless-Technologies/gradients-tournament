from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import PartialState


def load_model(model_name: str, config: dict) -> AutoModelForCausalLM:

    if config["attention_type"] == "flash":
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    elif config["attention_type"] == "eager":
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.train()

    return model


def get_lora_adapter(model: AutoModelForCausalLM, cfg: dict) -> AutoModelForCausalLM:
    if get_peft_model is None:
        raise ImportError("peft library is required for LoRA adapters.")

    # Determine target modules for LoRA
    targets = cfg.get('target_modules') or []
    if not targets:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(x in name.lower() for x in ('attn', 'attention')):
                targets.append(name.split('.')[-1])
        targets = list(set(targets))
        if not targets:
            raise ValueError("Could not auto-detect attention modules for LoRA. Please set 'target_modules' in config.")

    peft_config = LoraConfig(
        r=int(cfg.get('lora_r', 16)),
        lora_alpha=int(cfg.get('lora_alpha', 16)),
        target_modules=targets,
        lora_dropout=float(cfg.get('lora_dropout', 0.05)),
        bias='none',
        task_type='CAUSAL_LM'
    )
    return peft_config

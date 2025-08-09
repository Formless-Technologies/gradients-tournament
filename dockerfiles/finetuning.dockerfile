FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

USER root

# System dependencies and performance tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    ccache \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja packaging

# Install ML packages with specific versions for compatibility
RUN pip install -U --no-cache-dir \
    numpy \
    transformers \
    accelerate \
    datasets\
    sentencepiece\
    huggingface_hub \
    wandb \
    peft \
    bitsandbytes \
    safetensors \
    tokenizers

# Install training frameworks
RUN pip install -U --no-cache-dir \
    trl \
    liger-kernel \
    optuna \
    mlflow \
    protobuf

# Install Flash Attention 2 (much faster than flash-attn v1)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install Triton for kernel compilation
RUN pip install --no-cache-dir triton

# Install DeepSpeed
RUN pip install --no-cache-dir -U deepspeed

# Install additional optimizations
RUN pip install --no-cache-dir \
    einops \
    scipy \
    numba \
    toml \
    hf_xet \
    psutil

RUN pip install textstat
RUN pip install detoxify
RUN pip install langcheck
RUN pip install debugpy
RUN pip install cleanlab
RUN pip install sentence-transformers

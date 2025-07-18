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
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

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
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install Triton for kernel compilation
RUN pip install --no-cache-dir triton

# Install DeepSpeed
RUN pip install --no-cache-dir -U deepspeed

# Install additional optimizations
RUN pip install --no-cache-dir \
    einops \
    scipy \
    numba \
    packaging \
    toml \
    hf_xet \
    psutil

RUN pip install textstat
RUN pip install detoxify
RUN pip install langcheck
RUN pip install debugpy

RUN pip install cleanlab
RUN pip install sentence-transformers

# Environment variables for optimal performance
ENV TOKENIZERS_PARALLELISM=false

# Ensure high-speed P2P/NCCL comms and fault tolerance
ENV NCCL_DEBUG=WARN
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
ENV DEEPSPEED_TIMEOUT=3600
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_SHARP_DISABLE=1


WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/outputs /workspace/data /workspace/input_data /workspace/training /workspace/scripts

COPY configs/ /workspace/configs
COPY training/ /workspace/training
COPY training_helpers/ /workspace/training/training_helpers
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_text_trainer.sh
RUN chmod +x /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]

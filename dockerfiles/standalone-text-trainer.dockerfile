FROM phoenixbeaudry/finetuning:v0.0.2

ARG DEBIAN_FRONTEND=noninteractive

USER root

# Environment variables for optimal performance
ENV TOKENIZERS_PARALLELISM=false

# Ensure high-speed P2P/NCCL comms and fault tolerance
ENV NCCL_DEBUG=WARN
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
ENV DEEPSPEED_TIMEOUT=3600
ENV TORCH_NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_SHARP_DISABLE=1
ENV PYTHONUNBUFFERED=1
ENV ACCELERATE_DISABLE_RICH=1

WORKDIR /workspace
RUN mkdir -p /workspace/configs /workspace/training /workspace/scripts /app/checkpoints

COPY text/configs/ /workspace/configs
COPY text/training/ /workspace/training
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_text_trainer.sh
RUN chmod +x /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]

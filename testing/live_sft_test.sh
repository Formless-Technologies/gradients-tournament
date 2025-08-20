#!/bin/bash

# Example configuration for InstructTextTask training

# Unique task identifier
TASK_ID="d1e5976f-ee28-4d36-881c-cd1f73ccafea"

# Model to fine-tune (from HuggingFace)
MODEL="unsloth/SmolLM2-360M"

# Dataset location - can be:
# - S3 URL: "s3://bucket/path/to/dataset.json"
# - Local file: "/path/to/dataset.json"
# - HuggingFace dataset: "username/dataset-name"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/f6c30972447d4848_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250815%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250815T202223Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=1a18749cdf53040c0e70af2e254a85ca3d670c13fde773ee7933413c53d65150"

# Dataset type mapping - maps your dataset columns to expected format
# For InstructTextTask:
# - field_instruction: column containing the instruction/question
# - field_output: column containing the expected output/answer
DATASET_TYPE='{
  "field_instruction":"instruct",
  "field_input": "input",
  "field_output":"output"
}'

# File format: "csv", "json", "hf" (HuggingFace), or "s3"
FILE_FORMAT="s3"

# Optional: Repository name for the trained model (just the model name, not username/model-name)
EXPECTED_REPO_NAME="my-finetuned-model"

# Create secure data directory
DATA_DIR="$(pwd)/testing/secure_data"
mkdir -p "$DATA_DIR"
chmod 777 "$DATA_DIR"

# Build the downloader image
docker build -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
docker build -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

# Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$DATA_DIR:/cache:rw" \
  --name downloader-example \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT"

# Run the training
echo "Starting training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  -e WANDB_LOGS_PATH="/cache/wandb_logs/$TASK_ID" \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --volume "$DATA_DIR:/cache:rw" \
  --name instruct-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --hours-to-complete "4" \
  --testing
#!/bin/bash
TASK_ID="7719761b-73c5-4100-98fb-cbe5a6847737"
MODEL="Qwen/Qwen3-0.6B"
DATASET="https://huggingface.co/datasets/adamo1139/toxic-dpo-natural-v5/resolve/main/toxic_dpo_natural_v5.jsonl?download=true"
DATASET_TYPE='{
  "field_prompt":"prompt",
  "field_chosen":"chosen",
  "field_rejected":"rejected",
  "prompt_format":"{prompt}",
  "chosen_format":"{chosen}",
  "rejected_format":"{rejected}"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=8

# Optional: Repository name for the trained model (just the model name, not username/model-name)
EXPECTED_REPO_NAME="my-finetuned-model"

# Create secure data directory
DATA_DIR="$(pwd)/secure_data"
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
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT"

# Run the training
echo "Starting training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --volume "$DATA_DIR:/cache:rw" \
  --name dpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --hours-to-complete "4"
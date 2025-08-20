#!/bin/bash
TASK_ID="af6d4131-4c12-480c-a041-75a23192581b"
MODEL="furiosa-ai/mlperf-gpt-j-6b"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/b8582b5e0716b68b_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250820%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250820T134140Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=5d7e41465af71ac91b65df533e808a767adea2640ee645ad349fa032b38d98e2"
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
  --task-type "DpoTask" \
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
  --name dpo-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --hours-to-complete "4" \
  --testing
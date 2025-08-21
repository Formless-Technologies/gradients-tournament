#!/bin/bash
TASK_ID="908181eb-b95d-4fea-9a5c-981182e2c0ee"
MODEL="NousResearch/Nous-Hermes-llama-2-7b"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/57e242f2bb0795a7_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250821%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250821T203358Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=09cc60d55468d96bd483d2e9cd6191520196834ac312d826427cefbe94f84024"
DATASET_TYPE='{
  "field_prompt":"prompt",
  "field_chosen":"chosen",
  "field_rejected":"rejected",
  "prompt_format":"{prompt}",
  "chosen_format":"{chosen}",
  "rejected_format":"{rejected}"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=6

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
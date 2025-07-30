#!/bin/bash
TASK_ID="c621c6f1-40be-4a54-add1-38585b4e002f"
MODEL="TinyLlama/TinyLlama_v1.1"
DATASET="https://huggingface.co/datasets/Amod/mental_health_counseling_conversations/resolve/main/combined_dataset.json?download=true"
DATASET_TYPE='{
  "field_prompt":"Context",
  "reward_functions":[
        {"reward_func":"def reward_func(completions, **kwargs):\n    # Count frequency of letter \"e\" in response\n    return [text.count(\"e\") / (len(text) + 1) for text in completions]",
        "reward_weight":0.7,"name":"e_counter"},
        {"reward_func":"def reward_func(completions, **kwargs):\n    # Reward responses that are long but not too long\n    return [min(len(text)/100, 1.0) for text in completions]",
        "reward_weight":0.3,"name":"length_scorer"}
    ]
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
  --task-type "GrpoTask" \
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
  --task-type "GrpoTask" \
  --file-format "$FILE_FORMAT" \
  --expected-repo-name "$EXPECTED_REPO_NAME" \
  --hours-to-complete "4"
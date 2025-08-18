#!/bin/bash
TASK_ID="cfe9aab0-9daa-4ad0-bba4-26c69909b721"
MODEL="UCLA-AGI/Gemma-2-9B-It-SPPO-Iter2"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/30f6460d7b02784d_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250816%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250816T115423Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=23f3ba05d375763f3a0bcac0d11de840c4630757a8fc9aceabcad4d030e291c6"
DATASET_TYPE='{
  "field_prompt": "prompt",
  "reward_functions": [
    {
      "reward_func": "def reward_long_sentences(completions, **kwargs):\n    \"\"\"Rewards text with longer average sentence length.\"\"\"\n    import textstat\n    scores = [textstat.words_per_sentence(comp) for comp in completions]\n    return scores\n",
      "reward_weight": 3.7558904174941588,
      "func_hash": "0dca520c8a5cf852b2b8e1b93c9ac58acccf220ac6c41678f52386675de9f13d",
      "is_generic": true
    },
    {
      "reward_func": "def reward_long_words(completions, **kwargs):\n    \"\"\"Rewards text with more characters per word.\"\"\"\n    import textstat\n    scores = [textstat.avg_character_per_word(comp) for comp in completions]\n    return scores\n",
      "reward_weight": 4.843465021095151,
      "func_hash": "f17b69fa3359d0606755467fe78bdeae20e32926b7469c15c20af7cd46dd27bf",
      "is_generic": true
    },
    {
      "reward_func": "def reward_low_readability(completions, **kwargs):\n    \"\"\"Rewards less readable text using Flesch reading ease score.\"\"\"\n    import textstat\n    scores = [textstat.flesch_reading_ease(comp) for comp in completions]\n    return [-s for s in scores]\n",
      "reward_weight": 6.010304469381054,
      "func_hash": "b6fb681169f0fb6cd0f967f186176c196fd590ca603c1b332a951244730680db",
      "is_generic": true
    }
  ]
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
  --hours-to-complete "4" \
  --testing
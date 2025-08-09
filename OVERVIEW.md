# Project Overview

This is a repo for a tournament about finetuning either text or image models.

The winner is the model with the best evaluation loss.

For text we do either SFT, DPO, or GRPO finetuning. 

For image we finetune either FLUX or SDXL models.

We submit the entire repo as an entry to the tournament.

When we are evaluated the first thing that happens is the dockerimage for the relevant task is built.

This is either dockerfiles/standalone-text-trainer.dockerfile or dockerfiles/standalone-image-trainer.dockerfile

Then the image is run and passed the required arguments.

## Text Finetuning

For text finetuning the relevant files are configs/base.yml configs/serverless_config_handler.py training/ scripts/text_trainer.py and scripts/run_text_trainer.sh

## Image Finetuning 

For image finetuning the relevant files are configs/base_diffusion_sdxl.toml configs/base_diffusion_flux.toml configs/core_constants.py configs/trainer_constants.py configs/training_paths.py scripts/image_trainer.py and scripts/run_image_trainer.sh


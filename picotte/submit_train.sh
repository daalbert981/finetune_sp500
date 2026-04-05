#!/bin/bash
#SBATCH --job-name=gemma4-sp500
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=gemma4_train_%j.log
#SBATCH --error=gemma4_train_%j.err

# ---- Load modules ----
module load anaconda3
conda activate gemma4

# ---- Set paths ----
export REPO_DIR="$HOME/finetune_sp500"
export OUTPUT_DIR="$REPO_DIR/outputs_gemma4"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"   # <-- Replace with your token

# ---- HuggingFace cache (use scratch if available) ----
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- Launch multi-GPU training ----
cd "$REPO_DIR/picotte"

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision fp16 \
    train_gemma4.py

echo "Job complete."

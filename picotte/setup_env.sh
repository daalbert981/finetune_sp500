#!/bin/bash
# One-time setup: create conda environment for Gemma 4 fine-tuning on Picotte
# Run this interactively: bash setup_env.sh

module load anaconda3
conda create -n gemma4 python=3.11 -y
conda activate gemma4

# Install PyTorch with CUDA (V100 = CUDA compute 7.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace stack
pip install transformers --upgrade
pip install datasets accelerate peft trl bitsandbytes
pip install huggingface_hub hf_transfer sentencepiece protobuf

echo ""
echo "Environment 'gemma4' ready."
echo "Before training, set your HuggingFace token:"
echo "  export HF_TOKEN=hf_your_token_here"

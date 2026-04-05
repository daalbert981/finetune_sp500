"""
After training: merge LoRA into base model and export GGUF for llama.cpp inference.
Run on a node with enough RAM (CPU-only is fine, no GPU needed).

Usage:
    python export_gguf.py

Requires: pip install llama-cpp-python
The GGUF can then be used for local eval on Mac or any llama.cpp setup.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)) + "/..")
LORA_DIR = os.path.join(REPO_DIR, "outputs_gemma4", "lora")
MERGED_DIR = os.path.join(REPO_DIR, "outputs_gemma4", "merged")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

print("Loading base model in FP16 (needs ~62GB RAM, no GPU required)...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31B-it",
    torch_dtype=torch.float16,
    device_map="cpu",
    token=HF_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-31B-it", token=HF_TOKEN)

print(f"Loading LoRA from {LORA_DIR}...")
model = PeftModel.from_pretrained(model, LORA_DIR)

print("Merging LoRA into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}...")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print(f"""
Merged model saved to {MERGED_DIR}

To convert to GGUF, install llama.cpp and run:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    python convert_hf_to_gguf.py {MERGED_DIR} --outtype q8_0 --outfile sp500-gemma4-31b-Q8_0.gguf

Then push to HuggingFace:
    huggingface-cli upload daresearch/sp500-exec-classifier-gemma4-gguf sp500-gemma4-31b-Q8_0.gguf
""")

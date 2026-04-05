"""
SP500 Executive Classification — Fine-tune Gemma 4 31B on Picotte (multi-GPU V100)
Uses HuggingFace Transformers + PEFT + TRL with DDP for multi-GPU training.

Launch with:
    accelerate launch --multi_gpu --num_processes NUM_GPUS train_gemma4.py
"""

import os
import json
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-4-31B-it"
REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)) + "/..")
DATA_PATH = os.path.join(REPO_DIR, "Training Datasets", "full_data_n_4977.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(REPO_DIR, "outputs_gemma4"))
HF_TOKEN = os.environ.get("HF_TOKEN", None)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
is_main = local_rank == 0

if is_main:
    print(f"Model: {MODEL_NAME}")
    print(f"Data: {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"GPUs: {torch.cuda.device_count()}")

# ---------------------------------------------------------------------------
# Load model in 4-bit (each GPU gets a full copy)
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,  # V100 does not support BF16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": local_rank},
    torch_dtype=torch.float16,
    token=HF_TOKEN,
    attn_implementation="eager",  # V100 doesn't support flash attention 2
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# Apply LoRA
# ---------------------------------------------------------------------------
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

if is_main:
    model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Load and format training data
# ---------------------------------------------------------------------------
with open(DATA_PATH, "r") as f:
    examples = [json.loads(line) for line in f]

if is_main:
    print(f"Loaded {len(examples)} examples")

random.seed(42)
random.shuffle(examples)
split_idx = int(len(examples) * 0.95)
train_dataset = Dataset.from_list(examples[:split_idx])

def formatting_prompts_func(examples):
    texts = []
    for msgs in examples["messages"]:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

if is_main:
    print(f"Train samples: {len(train_dataset)}")
    print("=== First formatted example (truncated) ===")
    print(train_dataset[0]["text"][:500])

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
num_gpus = torch.cuda.device_count()
# Effective batch = num_gpus * per_device_batch * grad_accum
# Target effective batch ~16
grad_accum = max(1, 16 // (num_gpus * 1))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=3,
        warmup_ratio=0.03,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=OUTPUT_DIR,
        report_to="none",
        fp16=True,
        max_seq_length=2048,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    ),
)

if is_main:
    print(f"Effective batch size: {num_gpus * 1 * grad_accum}")
    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Training...")

trainer.train()

# ---------------------------------------------------------------------------
# Save (main process only)
# ---------------------------------------------------------------------------
if is_main:
    lora_dir = os.path.join(OUTPUT_DIR, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"LoRA adapters saved to {lora_dir}")

    if HF_TOKEN:
        model.push_to_hub("daresearch/sp500-exec-classifier-gemma4-lora", token=HF_TOKEN)
        tokenizer.push_to_hub("daresearch/sp500-exec-classifier-gemma4-lora", token=HF_TOKEN)
        print("Pushed to HuggingFace.")

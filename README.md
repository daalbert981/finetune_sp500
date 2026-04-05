# SP500 Executive Role Classification — Fine-Tuned gpt-oss-20b

Fine-tuned OpenAI's [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) to classify executive ranks and roles from SP500 proxy filings.

## What This Does

Given an executive's name, company, year, and official role title, the model outputs binary classifications across 18 labels:

**Ranks (12):** VP, SVP, EVP, SEVP, Director, Senior Director, MD, SMD, SE, VC, SVC, President

**Roles (6):** Board, CEO, CXO, Primary (value chain), Support (value chain), BU (business unit)

Example input:
```
In 2020 the company 'apple inc' had an executive with the name jane doe,
whose official role title was: 'senior vice president, human resources'.
```

Example output:
```
<rank>vp=0;svp=1;evp=0;sevp=0;dir=0;sdir=0;md=0;smd=0;se=0;vc=0;svc=0;president=0</rank>
<role>board=0;ceo=0;cxo=0;primary=0;support=1;bu=0</role>
```

## Model

- **Base model:** [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) — 21B total parameters, 3.6B active per token (Mixture-of-Experts)
- **Fine-tuning method:** QLoRA (4-bit quantization + Low-Rank Adaptation) via [Unsloth](https://github.com/unslothai/unsloth)
- **Training data:** 4,977 labeled examples in chat format (system/user/assistant)
- **LoRA config:** rank=128, alpha=32, targeting attention + MLP layers
- **Training:** 3 epochs, batch size 16, learning rate 2e-4, cosine schedule, A100 80GB GPU

## HuggingFace Artifacts

| Repo | Format | Purpose |
|------|--------|---------|
| [daresearch/sp500-exec-classifier-gguf](https://huggingface.co/daresearch/sp500-exec-classifier-gguf) | GGUF (MXFP4) | Fast inference via llama.cpp / Ollama |
| [daresearch/sp500-exec-classifier-lora](https://huggingface.co/daresearch/sp500-exec-classifier-lora) | LoRA adapters | Lightweight fine-tuning weights (~250MB) |
| [daresearch/sp500-exec-classifier](https://huggingface.co/daresearch/sp500-exec-classifier) | Merged MXFP4 | Full merged model (has known router weight issues — use GGUF instead) |

**For inference, use the GGUF version.** The merged safetensors version has MoE router weight mapping issues with current transformers/Unsloth versions.

## Repository Structure

```
finetune_sp500.ipynb              # Training notebook (Colab + Unsloth)
eval_holdout.ipynb                # Evaluation notebook (Colab + llama.cpp)
Training Datasets/
  full_data_n_4977.jsonl          # Training data (4,977 examples, chat format)
  holdout_labeling_09072024_template.csv  # Holdout set (2,010 examples)
```

## How to Run

### Training (finetune_sp500.ipynb)

1. Open in Colab: **File → Open → GitHub → `daalbert981/finetune_sp500` → `finetune_sp500.ipynb`**
2. Set runtime to **A100 GPU**
3. Add your HuggingFace token as a Colab secret named `HF_TOKEN`
4. Run all cells

The notebook installs Unsloth from source (required for gpt-oss), loads the base model in 4-bit, trains with SFTTrainer, and pushes the model to HuggingFace in both LoRA adapter and GGUF formats.

**Training time:** ~100 minutes on A100 80GB for 3 epochs.

### Evaluation (eval_holdout.ipynb)

1. Open in Colab: **File → Open → GitHub → `daalbert981/finetune_sp500` → `eval_holdout.ipynb`**
2. Set runtime to **any GPU** (T4 sufficient, A100 faster)
3. Run all cells

The notebook installs llama-cpp-python with CUDA support (~5-10 min compile), downloads the GGUF from HuggingFace, and runs predictions on all 2,010 holdout examples.

**Inference time:** ~42 minutes for 2,010 examples on A100 (~1.2s per example, 150 tokens/second).

## Architecture Notes

### Why Unsloth for Training

gpt-oss-20b has a Mixture-of-Experts (MoE) architecture with specific weight naming for the router layers (`model.layers.X.mlp.router.weight`). The HuggingFace `transformers` library (as of 4.56.2) expects a different naming convention (`router.linear.weight`). Unsloth patches this mismatch during model loading, which is why training must go through Unsloth.

### Why llama.cpp for Inference

Unsloth's inference path is extremely slow for gpt-oss (~0.4s per token, or 33s per example on A100). This is because Unsloth uses eager-mode Python execution with training-oriented attention patches. By contrast, llama.cpp uses compiled CUDA kernels with proper KV-caching, achieving ~150 tokens/second — a **60x speedup**.

The GGUF export (`model.save_pretrained_gguf()`) merges the LoRA adapter into the base weights and converts to a format llama.cpp reads directly. The resulting file is fully self-contained.

### Known Issues

- **Unsloth eval bug:** Batched evaluation during training crashes due to a 2D vs 4D attention mask mismatch in Unsloth's gpt-oss patches. Training eval is disabled; use the holdout evaluation instead.
- **Merged model router weights:** `save_pretrained_merged()` with both `merged_16bit` and `mxfp4` methods produces models with mismatched MoE router weight names. The GGUF export is the only reliable merged format.
- **vLLM LoRA:** vLLM does not yet support LoRA adapters for gpt-oss models.
- **Reasoning overhead:** gpt-oss is a reasoning model. The chat template injects `Reasoning: medium` by default. The fine-tuning data contained no reasoning tokens, so the model learned to skip reasoning. At inference, use `reasoning_effort="none"` if using the Unsloth/transformers path.

## Data Format

### Training Data (JSONL)

Each line is a JSON object with OpenAI chat format:
```json
{
  "messages": [
    {"role": "system", "content": "This assistant is trained to code executive ranks and roles..."},
    {"role": "user", "content": "In 2015 the company 'cms' had an executive with the name david mengebier, whose official role title was: 'senior vice president, cms energy and consumers energy'."},
    {"role": "assistant", "content": "<rank>vp=0;svp=1;...</rank>\n<role>board=0;ceo=0;...</role>"}
  ]
}
```

### Holdout Data (CSV)

Flat CSV with columns: `year`, `company`, `role`, `full_name`, `vp`, `svp`, `evp`, `sevp`, `dir`, `sdir`, `md`, `smd`, `se`, `vc`, `svc`, `board`, `ceo`, `cxo`, `primary`, `support`, `bu`, `president`.

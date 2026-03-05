# Model Architecture — Abu Hashim / QalamAI

## Overview

Abu Hashim is built on top of open-source, Arabic-capable large language models. The default base model is **Aya-23-8B** by CohereForAI, a multilingual model with strong Arabic language understanding and generation capabilities.

## Base Model

| Property           | Value                        |
|--------------------|------------------------------|
| Model Name         | CohereForAI/aya-23-8B        |
| Parameters         | 8 Billion                    |
| Architecture       | Transformer (Causal LM)      |
| Context Length     | 2048 tokens (configurable)   |
| Precision          | bfloat16                     |
| Quantization       | 4-bit (NF4 via bitsandbytes) |

## Supported Base Models

| Key            | Model                              | Parameters | Recommended |
|----------------|-------------------------------------|-----------|-------------|
| `aya-23-8b`    | CohereForAI/aya-23-8B              | 8B        | Yes         |
| `aya-23-35b`   | CohereForAI/aya-23-35B             | 35B       | No          |
| `jais-13b`     | inception-mbzuai/jais-13b-chat     | 13B       | No          |
| `qwen2-7b`     | Qwen/Qwen2-7B-Instruct            | 7B        | No          |

## Fine-Tuning Strategy

Abu Hashim uses **LoRA (Low-Rank Adaptation)** with **QLoRA** (Quantized LoRA) for efficient fine-tuning on consumer and cloud GPUs.

### LoRA Configuration

| Parameter         | Default Value |
|-------------------|---------------|
| Rank (r)          | 64            |
| Alpha             | 128           |
| Dropout           | 0.05          |
| Bias              | none          |
| Task Type         | CAUSAL_LM     |
| Target Modules    | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Quantization (QLoRA)

| Parameter                | Value    |
|--------------------------|----------|
| Load in 4-bit            | Yes      |
| Compute Dtype            | bfloat16 |
| Quantization Type        | NF4      |
| Double Quantization      | Yes      |

## Generation Parameters

| Parameter            | Default |
|----------------------|---------|
| Max New Tokens       | 512     |
| Temperature          | 0.7     |
| Top-p                | 0.9     |
| Top-k                | 50      |
| Repetition Penalty   | 1.1     |

## System Prompt

The model uses a built-in Arabic system prompt:

> أنت أبو هاشم، مساعد ذكاء اصطناعي عربي متقدم. أجب بدقة ووضوح باللغة العربية.

Translation: "You are Abu Hashim, an advanced Arabic AI assistant. Answer accurately and clearly in Arabic."

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│           User Input (Arabic)           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Safety Filter (Input)          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│        Tokenizer (Aya-23 / Base)        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   Base Model (4-bit Quantized)          │
│   + LoRA Adapter (Fine-tuned Weights)   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Safety Filter (Output)          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│        Generated Response (Arabic)      │
└─────────────────────────────────────────┘
```

## Directory Structure

```
model_base/
├── __init__.py
├── config.py            # Model configuration and supported models
├── download_model.py    # Script to download models from HuggingFace
└── weights/             # Downloaded model weights (gitignored)

model_finetune/
├── __init__.py
├── checkpoints/         # Training checkpoints
└── logs/                # Training logs
```

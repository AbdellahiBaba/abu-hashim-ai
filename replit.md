# QalamAI - Abu Hashim AI Model

## Overview
A fully independent, production-ready AI model framework called "Abu Hashim" built entirely using open-source LLMs, self-hosted components, and reproducible training pipelines. No dependency on proprietary LLM APIs.

## Project Structure
```
/model_base         - Base model configuration and download scripts
/model_finetune     - Fine-tuned LoRA adapter weights
/model_inference    - Inference optimization utilities
/dataset_raw        - Raw training data (JSONL, CSV, TXT, Parquet)
/dataset_processed  - Processed datasets in HuggingFace format
/training_scripts   - Data pipeline, training, and self-learning scripts
/evaluation         - Evaluation metrics, benchmarks, and report generation
/api_server         - FastAPI inference server with admin dashboard
/docs               - Full project documentation
/branding           - QalamAI brand assets (colors, typography, themes)
/qalam_bridge       - QalamAI data collection bridge (import, score, build, update)
/learning_buffer    - Self-learning data buffer for incremental training
```

## Tech Stack
- **Language**: Python 3.11
- **Web Framework**: FastAPI + Uvicorn
- **ML Framework**: PyTorch, HuggingFace Transformers, PEFT (LoRA/QLoRA)
- **Quantization**: BitsAndBytes (4-bit/8-bit)
- **Templating**: Jinja2
- **Branding**: QalamAI identity (#0A1A2F Dark Navy Blue, #D4AF37 Gold)

## Key Components
1. **Dataset Pipeline** (`training_scripts/data_pipeline.py`) - Ingests raw data, cleans Arabic text, removes PII, formats into instruction pairs
2. **Training Pipeline** (`training_scripts/train.py`) - LoRA/QLoRA fine-tuning with gradient checkpointing and mixed precision
3. **Incremental Training** (`training_scripts/incremental_train.py`) - Append new data and continue training
4. **Inference Server** (`api_server/main.py`) - FastAPI server with streaming, safety filters, temperature controls
5. **Evaluation Suite** (`evaluation/evaluate.py`) - Arabic fluency, style consistency, quality metrics
6. **Self-Learning** (`training_scripts/self_learning.py`) - Validate, clean, and merge new learning data
7. **Admin Dashboard** (`api_server/templates/dashboard.html`) - Web UI for model management
8. **QalamAI Bridge** (`qalam_bridge/`) - Imports QalamAI.net GPT-5.2 interaction logs, scores quality, builds training datasets, and supports continuous learning. Includes live webhook receiver (`/api/qalam-webhook`) for real-time data from QalamAI.net. See [docs/qalam_bridge.md](docs/qalam_bridge.md)

## Running
The application starts via `python main.py` which launches the FastAPI server on port 5000.

## Base Model
Recommended: CohereForAI/aya-23-8B (Arabic-capable, fine-tuning friendly)
Alternatives: core42/jais-13b-chat, Qwen/Qwen2-7B-Instruct

## Environment Variables
- `QALAM_MODEL_PATH` - Path to base model (optional, uses config default)
- `SESSION_SECRET` - Session secret for the application
- `WEBHOOK_SECRET` - Shared secret for QalamAI.net webhook authentication

## Notes
- ML libraries (torch, transformers, etc.) are listed in requirements.txt but not installed in Replit due to size
- The server runs in demo mode when ML libraries are unavailable
- For actual model training and inference, deploy to GPU infrastructure

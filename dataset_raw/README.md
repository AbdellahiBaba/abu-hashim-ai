# Dataset Raw — Expected Data Formats

This directory holds raw, unprocessed data files for training the Abu Hashim model.

## Supported Input Formats

### 1. JSONL (JSON Lines) — Preferred
Each line is a JSON object:
```json
{"input": "user prompt or question", "output": "model response or completion"}
```

### 2. CSV
Columns: `input`, `output` (and optionally `category`, `source`)
```csv
input,output,category,source
"ما هو الذكاء الاصطناعي؟","الذكاء الاصطناعي هو...","science","article"
```

### 3. Plain Text (.txt)
One document per file. Used for unsupervised pre-training data such as novels, articles, or essays. The pipeline will split these into chunks automatically.

### 4. Parquet
HuggingFace-compatible Parquet files with `input` and `output` columns.

## Expected Data Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `conversation` | Chat-style prompt/response pairs | Historical Abu Hashim outputs |
| `writing` | Creative or professional writing | Novels, articles, scripts |
| `academic` | Academic or technical content | Research summaries, explanations |
| `correction` | User corrections to model outputs | Before/after pairs |
| `system_prompt` | System prompt definitions | Persona instructions, behavior rules |
| `style_example` | Style reference samples | Tone, voice, formatting examples |

## File Naming Convention

Files should follow this pattern:
```
<category>_<source>_<date>.<ext>
```
Example: `conversation_history_20240301.jsonl`

## Data Quality Requirements

- Text must be UTF-8 encoded
- Arabic text should use standard Unicode (not legacy encodings)
- Minimum 10 characters per input/output field
- No duplicate entries within a single file
- No binary or non-text content embedded in text fields

## Directory Structure

```
dataset_raw/
├── README.md              (this file)
├── conversations/         (chat history exports)
├── writing_samples/       (novels, articles, scripts)
├── corrections/           (user correction pairs)
├── system_prompts/        (persona definitions)
└── style_examples/        (style reference texts)
```

Create subdirectories as needed. The data pipeline (`training_scripts/data_pipeline.py`) will recursively scan this directory for supported file types.

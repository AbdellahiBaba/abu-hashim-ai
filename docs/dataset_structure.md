# Dataset Structure — Abu Hashim / QalamAI

## Overview

The dataset pipeline processes raw text data into structured instruction-response pairs suitable for fine-tuning the Abu Hashim model. Data flows from `dataset_raw/` through the processing pipeline into `dataset_processed/`.

## Directory Layout

```
dataset_raw/
├── README.md                 # Format specifications
├── conversations/            # Chat history exports
├── writing_samples/          # Novels, articles, scripts
├── corrections/              # User correction pairs
├── system_prompts/           # Persona definitions
└── style_examples/           # Style reference texts

dataset_processed/
├── __init__.py
├── train.jsonl               # Training split (generated)
├── eval.jsonl                # Evaluation split (generated)
└── metadata.json             # Processing metadata (generated)
```

## Supported Input Formats

### JSONL (Preferred)

Each line is a JSON object with `input` and `output` fields:

```json
{"input": "ما هو الذكاء الاصطناعي؟", "output": "الذكاء الاصطناعي هو فرع من علوم الحاسوب..."}
```

### CSV

Columns: `input`, `output`, and optionally `category`, `source`:

```csv
input,output,category,source
"ما هو الذكاء الاصطناعي؟","الذكاء الاصطناعي هو...","science","article"
```

### Plain Text (.txt)

One document per file. The pipeline splits these into chunks automatically for unsupervised pre-training data.

### Parquet

HuggingFace-compatible Parquet files with `input` and `output` columns.

## Data Categories

| Category        | Description                          | Examples                        |
|-----------------|--------------------------------------|---------------------------------|
| `conversation`  | Chat-style prompt/response pairs     | Historical Abu Hashim outputs   |
| `writing`       | Creative or professional writing     | Novels, articles, scripts       |
| `academic`      | Academic or technical content        | Research summaries, explanations|
| `correction`    | User corrections to model outputs    | Before/after pairs              |
| `system_prompt` | System prompt definitions            | Persona instructions            |
| `style_example` | Style reference samples              | Tone, voice, formatting         |

## File Naming Convention

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

## Processing Pipeline

The data pipeline (`training_scripts/data_pipeline.py`) performs the following steps:

1. **Ingestion** — Recursively scans `dataset_raw/` for supported file types
2. **Text Cleaning** — Arabic-specific normalization via `training_scripts/text_cleaner.py`:
   - Remove diacritics (tashkeel)
   - Remove tatweel (kashida)
   - Normalize alef variants
   - Normalize Arabic punctuation
   - Remove URLs and emails
   - Collapse excessive whitespace
   - Limit repeated characters
3. **PII Removal** — Strip personal/sensitive information via `training_scripts/pii_remover.py`
4. **Formatting** — Convert raw data into instruction pairs via `training_scripts/data_formatter.py`
5. **Splitting** — Divide into training and evaluation sets (default 90/10 split)
6. **Output** — Write processed data to `dataset_processed/`

## Arabic Text Cleaning Details

The `TextCleaner` class handles:

| Operation               | Description                                      |
|------------------------|--------------------------------------------------|
| Diacritics removal     | Strips tashkeel marks (U+064B–U+065F, U+0670)   |
| Tatweel removal        | Removes kashida character (U+0640)               |
| Alef normalization     | Unifies أ, إ, آ → ا                              |
| Punctuation mapping    | Converts Arabic punctuation to Latin equivalents  |
| Repeated char limit    | Limits consecutive repeated characters to 2       |
| URL/email removal      | Strips URLs and email addresses                   |
| Whitespace collapse    | Normalizes spaces and newlines                    |
| Unicode normalization  | Applies NFKC normalization                       |

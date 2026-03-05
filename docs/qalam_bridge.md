# QalamAI Bridge — Data Collection & Learning Pipeline

## Overview

The QalamAI Bridge (`qalam_bridge/`) enables the Abu Hashim model to learn from QalamAI.net's GPT-5.2 interaction logs. It collects, cleans, anonymizes, scores, and structures prompt/response interactions into high-quality training data — without ever calling GPT for inference.

---

## 1. Exporting QalamAI.net Logs

### Expected Formats

The bridge accepts three export formats from QalamAI.net:

| Format | Extension | Description |
|--------|-----------|-------------|
| JSON   | `.json`   | Single object or array of records |
| JSONL  | `.jsonl`  | One JSON object per line |
| CSV    | `.csv`    | Comma-separated with header row |

### Expected Fields

Each record should contain some or all of the following fields (aliases are supported):

| Field | Aliases | Required | Description |
|-------|---------|----------|-------------|
| input | `prompt`, `user_prompt`, `question`, `user`, `user_message` | Yes | User's prompt |
| output | `response`, `gpt_response`, `answer`, `assistant`, `assistant_message`, `completion` | Yes | Model response |
| category | `type`, `genre`, `task_type` | No | Content category (novel, article, script, academic, poetry, essay, dialogue, summary, translation, general) |
| correction | `corrected`, `edited_response`, `human_correction` | No | Human-corrected version (replaces output if present) |
| timestamp | `created_at`, `date`, `time`, `datetime` | No | Record timestamp |
| writing_sample | `sample`, `text`, `content` | No | Standalone writing sample (used as output with a default Arabic prompt if no input/output exists) |

Additional metadata fields (`source`, `model`, `session_id`, `language`, `tags`) are preserved if present.

### JSON Structure Variants

The JSON importer supports multiple structures:

```json
// Array of records
[{"input": "...", "output": "..."}]

// Object with data key
{"data": [{"input": "...", "output": "..."}]}

// Object with records key
{"records": [{"input": "...", "output": "..."}]}

// Object with interactions key
{"interactions": [{"input": "...", "output": "..."}]}

// Single record
{"input": "...", "output": "..."}
```

---

## 2. Importing Data

### CLI Import

Import a single file:

```bash
python -m qalam_bridge.importer path/to/export.json
```

Import an entire directory:

```bash
python -m qalam_bridge.importer path/to/exports/
```

Optional arguments:

```bash
python -m qalam_bridge.importer path/to/export.json \
  --raw-dir dataset_raw/qalam_exports/ \
  --processed-dir dataset_processed/qalam_processed/
```

### API Import

Upload a file via the REST API:

```bash
curl -X POST http://localhost:5000/api/qalam-import \
  -F "file=@export.json"
```

### Programmatic Import

```python
from qalam_bridge.importer import QalamImporter

importer = QalamImporter()
records = importer.import_file("path/to/export.json")
stats = importer.get_stats()
```

### What Happens During Import

1. **Raw copy** — The original file is saved to `dataset_raw/qalam_exports/` (timestamped if a file with the same name exists)
2. **Parsing** — Records are extracted from the file based on format
3. **Field resolution** — Field aliases are mapped to the unified schema
4. **Arabic normalization** — Text is cleaned using `training_scripts.text_cleaner` (diacritics normalization, whitespace cleanup)
5. **PII removal** — Personal information is detected and redacted using `training_scripts.pii_remover`
6. **Validation** — Records with input < 5 chars or output < 10 chars are rejected
7. **Deduplication** — SHA-256 hash of input+output prevents duplicates
8. **Output** — Processed records are saved to `dataset_processed/qalam_processed/` in JSONL format

### Unified Record Format

Every processed record follows this schema:

```json
{
  "input": "cleaned user prompt",
  "output": "cleaned model response",
  "category": "article",
  "quality": null,
  "metadata": {
    "timestamp": "2025-01-15T10:30:00Z",
    "source": "qalamAI"
  },
  "hash": "sha256hex..."
}
```

---

## 3. Cleaning & Scoring Pipeline

### Quality Scoring

The quality scorer (`qalam_bridge/quality_scorer.py`) evaluates each record on five dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Length | 0.20 | Output length relative to category expectations |
| Coherence | 0.25 | Sentence structure, vocabulary diversity, absence of repetition |
| Formatting | 0.15 | Paragraph structure, line formatting |
| Arabic Ratio | 0.25 | Proportion of Arabic characters in the output |
| Completeness | 0.15 | Proper endings, sufficient length, non-empty input |

### Category-Aware Expectations

Each category has tailored length and structure expectations:

| Category | Min Length | Ideal Length | Max Length | Min Sentences | Expects Paragraphs |
|----------|-----------|-------------|-----------|---------------|-------------------|
| novel | 200 | 1000 | 50000 | 3 | Yes |
| article | 150 | 800 | 10000 | 2 | Yes |
| script | 100 | 500 | 20000 | 2 | No |
| academic | 200 | 1200 | 15000 | 3 | Yes |
| default | 50 | 400 | 20000 | 1 | No |

### Quality Flags

The scorer detects and flags problematic records:

- `empty_output` — No output text
- `repeated_phrases` — Same phrase repeated 3+ times
- `low_vocabulary_diversity` — Unique word ratio below 30%
- `incomplete_response` — Output ends with continuation markers (e.g., "...", "و", "ثم")
- `empty_input` — No input prompt
- `too_short` — Output below minimum length for category

### Filtering Thresholds

| Threshold | Default | Description |
|-----------|---------|-------------|
| Quality threshold | 0.6 | Minimum score to include in training data |
| Priority threshold | 0.85 | Score for high-priority training examples |

### Quality Distribution Buckets

- **Excellent** (0.85–1.0) — Flagged as priority training examples
- **Good** (0.7–0.85) — Included in training
- **Acceptable** (0.6–0.7) — Included in training
- **Poor** (0.4–0.6) — Rejected
- **Very Poor** (0–0.4) — Rejected

---

## 4. Building the Training Dataset

### CLI

```bash
python -m qalam_bridge.dataset_builder
```

Options:

```bash
python -m qalam_bridge.dataset_builder \
  --processed-dir dataset_processed/qalam_processed/ \
  --output-dir dataset_processed/ \
  --quality-threshold 0.6 \
  --train-ratio 0.9 \
  --seed 42
```

### API

```bash
curl -X POST http://localhost:5000/api/qalam-build-dataset
```

### Programmatic

```python
from qalam_bridge.dataset_builder import build_dataset

result = build_dataset(
    quality_threshold=0.6,
    train_ratio=0.9,
)
print(result["stats"])
print(result["paths"])
```

### Build Process

1. **Load** — Reads all `.jsonl` files from `dataset_processed/qalam_processed/`
2. **Score & Filter** — Applies quality scoring, keeps records above threshold
3. **Format** — Converts to instruction-pair format using `training_scripts.data_formatter`
4. **Split** — Randomly splits into train/eval sets (default 90/10, seeded for reproducibility)
5. **Save** — Writes `train.jsonl` and `eval.jsonl` to `dataset_processed/`

### Output Statistics

The build process tracks and reports:
- Total records loaded, accepted, rejected, priority
- Average quality score
- Quality distribution (excellent/good/acceptable/poor)
- Category breakdown (novel/article/script/academic/etc.)
- Train and eval set sizes

---

## 5. Incremental Training with New Data

### Continuous Learning Pipeline

The update pipeline (`qalam_bridge/update_dataset.py`) processes new exports incrementally:

```bash
python -m qalam_bridge.update_dataset
```

Options:

```bash
python -m qalam_bridge.update_dataset \
  --buffer-dir learning_buffer/ \
  --quality-threshold 0.6 \
  --priority-threshold 0.85
```

### How It Works

1. **Scan** — Checks `learning_buffer/` for new `.json`, `.csv`, or `.jsonl` files
2. **Import** — Cleans and normalizes using the importer pipeline
3. **Score** — Applies quality scoring and filtering
4. **Deduplicate** — Compares hashes against the main dataset to avoid duplicates
5. **Append** — Adds high-quality records to `dataset_processed/qalam_training_data.jsonl`
6. **Log** — Records update statistics to `dataset_processed/qalam_update_log.jsonl`

### Triggering Model Retraining

After updating the dataset, trigger incremental training:

```bash
python -m training_scripts.update_model
```

The `update_model` module can optionally run the dataset update before starting training, creating a seamless pipeline from new data to improved model.

### Full Workflow

```bash
# 1. Place new QalamAI exports in learning_buffer/
cp new_export.jsonl learning_buffer/

# 2. Process and append to dataset
python -m qalam_bridge.update_dataset

# 3. Rebuild train/eval splits
python -m qalam_bridge.dataset_builder

# 4. Run incremental training
python -m training_scripts.incremental_train

# 5. Evaluate the updated model
python -m evaluation.evaluate
```

---

## 6. Evaluating Improvements After Training

After training with new QalamAI data, evaluate the model using the evaluation suite:

```bash
python -m evaluation.evaluate
```

The evaluation framework (`evaluation/`) measures:

- **Arabic fluency** — Grammar, coherence, and natural language quality
- **Style consistency** — Adherence to expected writing style per category
- **Quality metrics** — Output length, completeness, relevance
- **Benchmarks** — Comparison against baseline performance

Generate an HTML report:

```bash
python -m evaluation.report_generator
```

### Monitoring Dataset Quality

Check current dataset statistics via the API:

```bash
curl http://localhost:5000/api/qalam-stats
```

This returns:
- Total records in the dataset
- Quality distribution (excellent/good/acceptable/poor)
- Category breakdown
- Dataset file path and existence status

---

## 7. Privacy & Safety Guarantees

### PII Protection

- All imported data passes through `training_scripts.pii_remover` before storage
- Detected PII types: email addresses, phone numbers, personal names, URLs, ID numbers
- PII is redacted (replaced with placeholder tokens) — never stored in raw form
- PII detection count is tracked in import statistics

### Data Isolation

- Raw exports are stored in `dataset_raw/qalam_exports/` (for audit purposes)
- Processed (anonymized) data is stored separately in `dataset_processed/qalam_processed/`
- No original user data is included in training datasets

### No External API Calls

- The bridge only processes exported log files — it never calls GPT or any external API
- All cleaning, scoring, and formatting happens locally
- The model learns from historical interactions, not live queries

### Content Safety

- The API server applies `api_server/safety_filters.py` to all generated content
- Bilingual (Arabic/English) content safety filtering is active during inference
- Training data quality scoring helps exclude low-quality or potentially harmful content

### Deduplication

- SHA-256 hashing prevents the same input/output pair from being trained on multiple times
- Deduplication is enforced at both import time and dataset update time

---

## 8. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/qalam-import` | Upload and import a QalamAI export file |
| GET | `/api/qalam-stats` | Get import statistics and dataset info |
| POST | `/api/qalam-build-dataset` | Build train/eval datasets from imported data |

### POST /api/qalam-import

**Request**: Multipart form with `file` field containing a `.json`, `.csv`, or `.jsonl` file.

**Response**:
```json
{
  "status": "success",
  "records_imported": 150,
  "stats": {
    "files_imported": 1,
    "records_read": 200,
    "records_accepted": 150,
    "records_rejected": 30,
    "duplicates_skipped": 20,
    "pii_detections": 5
  }
}
```

### GET /api/qalam-stats

**Response**:
```json
{
  "dataset_path": "dataset_processed/qalam_training_data.jsonl",
  "exists": true,
  "total_records": 500,
  "categories": {"article": 200, "novel": 150, "general": 150},
  "quality_distribution": {
    "excellent": 100,
    "good": 200,
    "acceptable": 150,
    "poor": 50
  }
}
```

### POST /api/qalam-build-dataset

**Response**:
```json
{
  "status": "success",
  "stats": {
    "total_records": 500,
    "accepted_records": 450,
    "rejected_records": 50,
    "train_records": 405,
    "eval_records": 45
  },
  "paths": {
    "train": "dataset_processed/train.jsonl",
    "eval": "dataset_processed/eval.jsonl"
  }
}
```

### POST /api/qalam-webhook

Live webhook endpoint for real-time data from QalamAI.net.

**Headers** (required):
- `X-Webhook-Secret`: Shared secret matching the `WEBHOOK_SECRET` environment variable

**Request body**:
```json
{
  "input": "user prompt text",
  "output": "AI response text",
  "category": "article",
  "correction": null,
  "timestamp": "2026-03-05T12:00:00Z",
  "metadata": {
    "session_id": "abc123",
    "language": "ar",
    "word_count": 150
  }
}
```

**Response** (200):
```json
{
  "status": "accepted",
  "quality": 0.782,
  "hash": "a1b2c3d4..."
}
```

Status values: `accepted` (stored for training), `rejected` (failed quality/validation), `duplicate` (already exists).

**Error responses**:
- `401`: Invalid or missing `X-Webhook-Secret` header
- `400`: Invalid JSON or non-object payload
- `500`: Internal processing error

### GET /api/qalam-webhook-stats

Returns webhook-specific counters.

**Response**:
```json
{
  "total_received": 150,
  "accepted": 120,
  "rejected": 15,
  "duplicates": 10,
  "errors": 5,
  "last_received": "2026-03-05T14:30:00.000000Z"
}
```

---

## 9. Live Webhook Integration

QalamAI.net is configured to send real-time interaction data to Abu Hashim via webhooks. Each time a user completes an interaction on QalamAI.net, a POST request is sent to `/api/qalam-webhook` with the interaction data.

### How It Works

1. User interacts with QalamAI.net (sends prompt, receives response)
2. QalamAI.net fires a webhook POST to Abu Hashim's `/api/qalam-webhook`
3. The request includes `X-Webhook-Secret` header for authentication
4. Abu Hashim verifies the secret, cleans the data (Arabic normalization, PII removal), checks for duplicates, scores quality, and stores accepted records
5. Data accumulates in `dataset_processed/qalam_processed/webhook_live.jsonl`
6. When ready, trigger "Build Training Dataset" to compile all data into train/eval splits

### Configuration

**On QalamAI.net** (environment variables):
- `TRAINING_WEBHOOK_URL`: The full URL to the webhook endpoint
- `WEBHOOK_SECRET`: Shared secret for authentication

**On Abu Hashim** (environment variables):
- `WEBHOOK_SECRET`: Must match the value set on QalamAI.net

### Monitoring

The QalamAI Bridge dashboard page shows live webhook status:
- Connection status (Waiting / Connected)
- Total received, accepted, rejected counts
- Last received timestamp

---

## 10. Module Structure

```
qalam_bridge/
├── __init__.py           # Package init with convenience imports
├── importer.py           # Core ingestion — JSON/CSV/JSONL parsing, cleaning, PII removal
├── quality_scorer.py     # Quality scoring — length, coherence, formatting, Arabic ratio
├── dataset_builder.py    # Dataset building — scoring, splitting, formatting for training
└── update_dataset.py     # Continuous learning — incremental updates, deduplication, logging
```

### Dependencies on Existing Modules

| Module | Used By | Purpose |
|--------|---------|---------|
| `training_scripts.text_cleaner` | importer, quality_scorer | Arabic text normalization |
| `training_scripts.pii_remover` | importer | PII detection and redaction |
| `training_scripts.data_formatter` | dataset_builder | Instruction-pair formatting |

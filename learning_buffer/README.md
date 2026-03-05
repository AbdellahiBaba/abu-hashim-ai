# Learning Buffer

The learning buffer is a staging area for new training data collected from user interactions, corrections, and curated sources before it is validated and merged into the main training dataset.

## Purpose

- Accumulate new conversational examples and instruction-response pairs over time
- Store user feedback signals (thumbs up/down, corrections) for quality filtering
- Provide a controlled pipeline for incremental model improvement

## Directory Structure

```
learning_buffer/
  pending/         # New raw entries awaiting validation
  validated/       # Entries that passed quality checks
  rejected/        # Entries that failed validation (kept for audit)
  metadata.jsonl   # Log of all entries with timestamps and status
```

## Data Format

Each entry is stored as a JSON object with the following schema:

```json
{
  "id": "uuid-v4",
  "timestamp": "ISO-8601",
  "source": "user_feedback | curated | synthetic",
  "instruction": "User prompt / question in Arabic",
  "response": "Model response in Arabic",
  "corrected_response": null,
  "feedback_score": 1,
  "status": "pending | validated | rejected",
  "metadata": {}
}
```

## Workflow

1. **Collection** - New entries are written to `pending/` as individual `.json` files.
2. **Validation** - `training_scripts/self_learning.py` reads pending entries, applies quality checks (language detection, PII removal, deduplication, min-length), and moves them to `validated/` or `rejected/`.
3. **Merging** - Validated entries are formatted into the training data schema and appended to `dataset_processed/`.
4. **Training** - An admin triggers `training_scripts/update_model.py` to launch incremental fine-tuning on the newly merged data.

## Quality Gates

- Minimum response length: 20 characters
- Arabic language detection confidence >= 0.8
- No PII detected (emails, phone numbers, national IDs)
- Deduplication against existing training data (cosine similarity < 0.95)
- Feedback score > 0 (net positive user signal)

## Safety

- All data is processed locally; nothing is sent to external APIs.
- PII removal runs before any data is persisted to validated storage.
- Rejected entries are kept for manual review but never used for training.

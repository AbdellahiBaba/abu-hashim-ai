# Update Model — Abu Hashim / QalamAI

## Overview

Abu Hashim includes a self-learning system that allows the model to improve over time using user feedback and new data. The update process is modular and admin-triggered — the model never updates itself automatically.

## Self-Learning Pipeline

### How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User        │     │  Learning    │     │  Validation  │     │  Incremental │
│  Feedback    │────▶│  Buffer      │────▶│  & Cleaning  │────▶│  Training    │
│  (API)       │     │  (Storage)   │     │  (Auto)      │     │  (Admin)     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Feedback Collection**: Users submit ratings and corrections through the `/feedback` API endpoint
2. **Learning Buffer**: Feedback is stored in `learning_buffer/` as structured JSONL files
3. **Validation & Cleaning**: `training_scripts/self_learning.py` validates, cleans, and deduplicates the data
4. **Model Update**: An admin triggers `training_scripts/update_model.py` to run incremental training

### Learning Buffer

The `learning_buffer/` directory stores incoming feedback data:

```
learning_buffer/
├── README.md              # Documentation
├── feedback_raw.jsonl     # Raw user feedback (generated)
├── feedback_clean.jsonl   # Cleaned and validated feedback (generated)
└── approved.jsonl         # Admin-approved training data (generated)
```

## Running an Update

### Step 1: Review Collected Feedback

```bash
python -m training_scripts.self_learning --review
```

This shows a summary of collected feedback: total entries, average ratings, categories.

### Step 2: Validate and Clean Data

```bash
python -m training_scripts.self_learning --validate
```

This processes raw feedback through:
- Text cleaning (Arabic normalization)
- Quality filtering (minimum rating threshold)
- Deduplication
- PII removal

Output is written to `learning_buffer/feedback_clean.jsonl`.

### Step 3: Merge with Training Data

```bash
python -m training_scripts.self_learning --merge
```

Merges validated feedback with existing training data in `dataset_processed/`.

### Step 4: Trigger Model Update

```bash
python -m training_scripts.update_model
```

This function:
1. Loads the latest model checkpoint from `model_finetune/checkpoints/`
2. Runs incremental training on the new data
3. Evaluates the updated model against benchmarks
4. Saves the new checkpoint if performance improves

## Safety Considerations

- Updates are always admin-triggered, never automatic
- All feedback data goes through validation and cleaning before training
- PII is removed from all training data
- The model is evaluated after each update to ensure quality does not degrade
- Previous checkpoints are preserved for rollback capability

## Rollback

If an update degrades model quality, revert to the previous checkpoint:

```bash
python -m training_scripts.update_model --rollback
```

This restores the model to the last known good checkpoint.

## Best Practices

1. **Review before updating**: Always review feedback data quality before triggering an update
2. **Backup checkpoints**: Keep at least 3 recent checkpoints (configured via `save_total_limit`)
3. **Evaluate after update**: Run the full evaluation suite after each update
4. **Small batches**: Update with smaller batches of high-quality data rather than large noisy batches
5. **Monitor metrics**: Track Arabic fluency, style consistency, and quality scores across updates

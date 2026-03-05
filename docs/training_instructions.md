# Training Instructions — Abu Hashim / QalamAI

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Dependencies installed: `pip install -r requirements.txt`
- Base model downloaded (see [Model Architecture](model_architecture.md))

## Step 1: Download the Base Model

```bash
python -m model_base.download_model --model aya-23-8b
```

List all available models:

```bash
python -m model_base.download_model --list
```

Specify a custom cache directory:

```bash
python -m model_base.download_model --model aya-23-8b --cache-dir /path/to/weights
```

## Step 2: Prepare Training Data

1. Place raw data files in `dataset_raw/` following the expected formats (see [Dataset Structure](dataset_structure.md))
2. Run the data processing pipeline:

```bash
python -m training_scripts.data_pipeline
```

This will output processed training data to `dataset_processed/`.

## Step 3: Configure Training

Training is configured via `training_scripts/train_config.py`. You can use preset configurations or customize parameters.

### Preset Configurations

| Preset        | Epochs | Batch Size | Seq Length | Use Case              |
|---------------|--------|------------|------------|-----------------------|
| `quick_test`  | 1      | 2          | 512        | Quick validation runs |
| `standard`    | 3      | 4          | 2048       | Normal training       |
| `high_quality`| 5      | 2          | 4096       | Maximum quality       |

### Key Hyperparameters

| Parameter                    | Default    | Description                           |
|------------------------------|------------|---------------------------------------|
| `num_train_epochs`           | 3          | Number of training epochs             |
| `per_device_train_batch_size`| 4          | Batch size per GPU                    |
| `gradient_accumulation_steps`| 4          | Gradient accumulation steps           |
| `learning_rate`              | 2e-4       | Learning rate                         |
| `weight_decay`               | 0.01       | Weight decay                          |
| `warmup_ratio`               | 0.03       | Warmup ratio                          |
| `lr_scheduler_type`          | cosine     | Learning rate scheduler               |
| `max_seq_length`             | 2048       | Maximum sequence length               |
| `gradient_checkpointing`     | True       | Enable gradient checkpointing         |
| `optim`                      | paged_adamw_32bit | Optimizer                       |
| `bf16`                       | True       | Use bfloat16 precision                |

### LoRA Parameters

| Parameter      | Default | Description              |
|----------------|---------|--------------------------|
| `lora_r`       | 64      | LoRA rank                |
| `lora_alpha`   | 128     | LoRA alpha               |
| `lora_dropout` | 0.05    | LoRA dropout             |
| `lora_bias`    | none    | LoRA bias                |

## Step 4: Run Training

```bash
python -m training_scripts.train
```

Use a preset:

```bash
python -m training_scripts.train --preset standard
```

Resume from a checkpoint:

```bash
python -m training_scripts.train --resume-from model_finetune/checkpoints/checkpoint-500
```

## Step 5: Monitor Training

- Checkpoints are saved to `model_finetune/checkpoints/` every 100 steps (configurable)
- Training logs are saved to `model_finetune/logs/`
- Best model is automatically selected based on evaluation loss

## Step 6: Evaluate the Model

After training, run the evaluation suite:

```bash
python -m evaluation.evaluate
```

See [Evaluation Metrics](model_architecture.md) for details on Arabic fluency, style consistency, and quality scoring.

## Incremental Training

For adding new data without full retraining:

```bash
python -m training_scripts.incremental_train --new-data path/to/new_data.jsonl
```

This loads the latest checkpoint and continues training on new data.

## Memory Optimization

The training pipeline includes several memory-saving features:

- **4-bit Quantization (QLoRA)**: Reduces model memory footprint by ~75%
- **Gradient Checkpointing**: Trades compute for memory
- **Paged AdamW**: Uses CPU memory for optimizer states when GPU memory is full
- **Gradient Accumulation**: Simulates larger batch sizes without additional memory

### Estimated VRAM Requirements

| Configuration  | Model      | VRAM Required |
|----------------|------------|---------------|
| Quick Test     | Aya-23-8B  | ~8 GB         |
| Standard       | Aya-23-8B  | ~12 GB        |
| High Quality   | Aya-23-8B  | ~16 GB        |
| Standard       | Aya-23-35B | ~24 GB        |

## Troubleshooting

| Issue                          | Solution                                          |
|--------------------------------|---------------------------------------------------|
| CUDA Out of Memory             | Reduce batch size or sequence length               |
| Slow training                  | Enable gradient checkpointing, reduce eval frequency|
| Loss not decreasing            | Check data quality, lower learning rate            |
| NaN loss                       | Lower learning rate, check for corrupted data      |
| Model not loading              | Verify model download completed successfully       |

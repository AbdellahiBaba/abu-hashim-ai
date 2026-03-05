import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "model_finetune", "checkpoints"
    )
    logging_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "model_finetune", "logs"
    )

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 16

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    max_seq_length: int = 2048

    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"

    quantization_enabled: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    report_to: str = "none"
    dataloader_num_workers: int = 4
    seed: int = 42
    optim: str = "paged_adamw_32bit"

    dataset_path: Optional[str] = None
    val_split_ratio: float = 0.1

    resume_from_checkpoint: Optional[str] = None
    merged_output_dir: Optional[str] = None


def get_training_config(**overrides) -> TrainingConfig:
    config = TrainingConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


PRESET_CONFIGS = {
    "quick_test": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 512,
        "save_steps": 50,
        "eval_steps": 50,
        "logging_steps": 5,
    },
    "standard": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
    },
    "high_quality": {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "learning_rate": 1e-4,
        "lora_r": 128,
        "lora_alpha": 256,
        "warmup_ratio": 0.05,
    },
}


def get_preset_config(preset_name: str) -> TrainingConfig:
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(PRESET_CONFIGS.keys())}"
        )
    return get_training_config(**PRESET_CONFIGS[preset_name])

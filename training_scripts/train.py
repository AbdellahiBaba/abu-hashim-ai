import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_base.config import ModelConfig, get_default_config
from training_scripts.train_config import TrainingConfig, get_training_config, get_preset_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("abu_hashim.train")


def build_quantization_config(config: TrainingConfig):
    if not config.quantization_enabled:
        return None

    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.bfloat16)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def load_base_model(model_config: ModelConfig, train_config: TrainingConfig):
    logger.info(f"Loading model: {model_config.model_name}")

    quant_config = build_quantization_config(train_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
        torch_dtype=getattr(torch, model_config.dtype, torch.bfloat16),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_name or model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def apply_lora(model, train_config: TrainingConfig):
    logger.info("Applying LoRA adapter")

    if train_config.quantization_enabled:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_config.gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=train_config.lora_r,
        lora_alpha=train_config.lora_alpha,
        lora_dropout=train_config.lora_dropout,
        target_modules=train_config.lora_target_modules,
        bias=train_config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def format_instruction(sample, tokenizer, model_config: ModelConfig, max_length: int):
    system_prompt = model_config.system_prompt
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text:
        prompt = f"### System:\n{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        prompt = f"### System:\n{system_prompt}\n\n### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_dataset(train_config: TrainingConfig, tokenizer, model_config: ModelConfig):
    dataset_path = train_config.dataset_path
    if not dataset_path:
        processed_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset_processed"
        )
        candidates = list(Path(processed_dir).glob("*.jsonl")) + list(
            Path(processed_dir).glob("*.json")
        )
        if candidates:
            dataset_path = str(candidates[0])
        else:
            raise FileNotFoundError(
                f"No dataset found in {processed_dir}. "
                "Run the data pipeline first or specify --dataset_path."
            )

    logger.info(f"Loading dataset from: {dataset_path}")

    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    elif dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")

    if isinstance(dataset, DatasetDict):
        if "train" in dataset and "validation" in dataset:
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"]
        else:
            split_key = list(dataset.keys())[0]
            split = dataset[split_key].train_test_split(
                test_size=train_config.val_split_ratio, seed=train_config.seed
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
    else:
        split = dataset.train_test_split(
            test_size=train_config.val_split_ratio, seed=train_config.seed
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    def tokenize_fn(sample):
        return format_instruction(
            sample, tokenizer, model_config, train_config.max_seq_length
        )

    train_dataset = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_fn, remove_columns=eval_dataset.column_names)

    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    return train_dataset, eval_dataset


def build_training_args(train_config: TrainingConfig) -> TrainingArguments:
    os.makedirs(train_config.output_dir, exist_ok=True)
    os.makedirs(train_config.logging_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=train_config.output_dir,
        logging_dir=train_config.logging_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        max_grad_norm=train_config.max_grad_norm,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        save_total_limit=train_config.save_total_limit,
        evaluation_strategy=train_config.evaluation_strategy,
        save_strategy=train_config.save_strategy,
        load_best_model_at_end=train_config.load_best_model_at_end,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        report_to=train_config.report_to,
        dataloader_num_workers=train_config.dataloader_num_workers,
        seed=train_config.seed,
        optim=train_config.optim,
        remove_unused_columns=False,
    )


def train(
    model_config: ModelConfig = None,
    train_config: TrainingConfig = None,
):
    if model_config is None:
        model_config = get_default_config()
    if train_config is None:
        train_config = get_training_config()

    logger.info("=" * 60)
    logger.info("Abu Hashim - QalamAI Training Pipeline")
    logger.info("=" * 60)

    model, tokenizer = load_base_model(model_config, train_config)
    model = apply_lora(model, train_config)

    if train_config.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset, eval_dataset = prepare_dataset(train_config, tokenizer, model_config)

    training_args = build_training_args(train_config)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    if train_config.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {train_config.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=train_config.resume_from_checkpoint)
    else:
        trainer.train()

    logger.info("Saving final adapter...")
    final_adapter_dir = os.path.join(train_config.output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    config_save_path = os.path.join(train_config.output_dir, "training_config.json")
    with open(config_save_path, "w") as f:
        json.dump(vars(train_config), f, indent=2, ensure_ascii=False)

    logger.info(f"Training complete. Adapter saved to: {final_adapter_dir}")
    return final_adapter_dir


def merge_and_save(adapter_dir: str, output_dir: str = None, model_config: ModelConfig = None):
    if model_config is None:
        model_config = get_default_config()
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "model_finetune", "merged"
        )

    logger.info(f"Merging adapter from {adapter_dir} into base model")

    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        device_map="auto",
        trust_remote_code=model_config.trust_remote_code,
        cache_dir=model_config.cache_dir,
        torch_dtype=getattr(torch, model_config.dtype, torch.bfloat16),
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Merged model saved to: {output_dir}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Abu Hashim LoRA/QLoRA Training")
    parser.add_argument("--preset", type=str, default=None, help="Training preset: quick_test, standard, high_quality")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--no_quantization", action="store_true", help="Disable quantization")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base model after training")
    parser.add_argument("--merge_only", type=str, default=None, help="Only merge an existing adapter (path)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.merge_only:
        merge_and_save(args.merge_only)
        return

    if args.preset:
        train_config = get_preset_config(args.preset)
    else:
        train_config = get_training_config()

    if args.dataset_path:
        train_config.dataset_path = args.dataset_path
    if args.output_dir:
        train_config.output_dir = args.output_dir
    if args.num_epochs:
        train_config.num_train_epochs = args.num_epochs
    if args.batch_size:
        train_config.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        train_config.learning_rate = args.learning_rate
    if args.max_seq_length:
        train_config.max_seq_length = args.max_seq_length
    if args.lora_r:
        train_config.lora_r = args.lora_r
    if args.no_quantization:
        train_config.quantization_enabled = False
    if args.resume_from:
        train_config.resume_from_checkpoint = args.resume_from

    adapter_dir = train(train_config=train_config)

    if args.merge:
        merge_and_save(adapter_dir)


if __name__ == "__main__":
    main()

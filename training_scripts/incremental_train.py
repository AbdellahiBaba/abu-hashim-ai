import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_base.config import ModelConfig, get_default_config
from training_scripts.train_config import TrainingConfig, get_training_config
from training_scripts.train import (
    build_quantization_config,
    format_instruction,
    build_training_args,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("abu_hashim.incremental_train")

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model_finetune",
    "training_history.json",
)


def load_training_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"sessions": [], "total_samples_trained": 0}


def save_training_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def find_latest_adapter(base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "model_finetune",
            "checkpoints",
        )

    final_adapter = os.path.join(base_dir, "final_adapter")
    if os.path.exists(final_adapter):
        return final_adapter

    checkpoint_dirs = sorted(
        Path(base_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
        reverse=True,
    )
    if checkpoint_dirs:
        return str(checkpoint_dirs[0])

    return None


def load_new_data(data_path, tokenizer, model_config, max_length, val_split=0.1, seed=42):
    logger.info(f"Loading new data from: {data_path}")

    if data_path.endswith(".jsonl") or data_path.endswith(".json"):
        dataset = load_dataset("json", data_files=data_path, split="train")
    elif os.path.isdir(data_path):
        from datasets import load_from_disk
        dataset = load_from_disk(data_path)
        if hasattr(dataset, "keys"):
            dataset = dataset[list(dataset.keys())[0]]
    else:
        dataset = load_dataset(data_path, split="train")

    def tokenize_fn(sample):
        return format_instruction(sample, tokenizer, model_config, max_length)

    dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    split = dataset.train_test_split(test_size=val_split, seed=seed)
    logger.info(
        f"New data - Train: {len(split['train'])}, Eval: {len(split['test'])}"
    )
    return split["train"], split["test"]


def incremental_train(
    new_data_path: str,
    adapter_path: str = None,
    model_config: ModelConfig = None,
    train_config: TrainingConfig = None,
    session_tag: str = None,
):
    if model_config is None:
        model_config = get_default_config()
    if train_config is None:
        train_config = get_training_config()

    if adapter_path is None:
        adapter_path = find_latest_adapter()

    if session_tag is None:
        session_tag = datetime.now().strftime("inc_%Y%m%d_%H%M%S")

    session_output_dir = os.path.join(train_config.output_dir, session_tag)
    train_config.output_dir = session_output_dir

    logger.info("=" * 60)
    logger.info("Abu Hashim - Incremental Training")
    logger.info(f"Session: {session_tag}")
    logger.info("=" * 60)

    quant_config = build_quantization_config(train_config)

    base_model = AutoModelForCausalLM.from_pretrained(
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
        base_model.config.pad_token_id = base_model.config.eos_token_id

    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Loading existing adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=True
        )
    else:
        logger.info("No existing adapter found, creating new LoRA adapter")
        if train_config.quantization_enabled:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=train_config.gradient_checkpointing,
            )
        lora_config = LoraConfig(
            r=train_config.lora_r,
            lora_alpha=train_config.lora_alpha,
            lora_dropout=train_config.lora_dropout,
            target_modules=train_config.lora_target_modules,
            bias=train_config.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)

    if train_config.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset, eval_dataset = load_new_data(
        new_data_path,
        tokenizer,
        model_config,
        train_config.max_seq_length,
        val_split=train_config.val_split_ratio,
        seed=train_config.seed,
    )

    train_config.learning_rate = train_config.learning_rate * 0.5
    train_config.warmup_ratio = 0.05

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

    logger.info("Starting incremental training...")
    trainer.train()

    final_adapter_dir = os.path.join(session_output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    history = load_training_history()
    history["sessions"].append({
        "tag": session_tag,
        "timestamp": datetime.now().isoformat(),
        "data_path": new_data_path,
        "adapter_path": adapter_path,
        "output_dir": final_adapter_dir,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "learning_rate": train_config.learning_rate,
        "epochs": train_config.num_train_epochs,
    })
    history["total_samples_trained"] += len(train_dataset)
    save_training_history(history)

    logger.info(f"Incremental training complete. Adapter saved to: {final_adapter_dir}")
    return final_adapter_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Abu Hashim - Incremental Training"
    )
    parser.add_argument(
        "--new_data", type=str, required=True,
        help="Path to new training data (JSON/JSONL)",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to existing adapter to continue from",
    )
    parser.add_argument(
        "--session_tag", type=str, default=None,
        help="Tag for this training session",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Learning rate",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_config = get_training_config()
    if args.num_epochs:
        train_config.num_train_epochs = args.num_epochs
    if args.learning_rate:
        train_config.learning_rate = args.learning_rate

    incremental_train(
        new_data_path=args.new_data,
        adapter_path=args.adapter_path,
        train_config=train_config,
        session_tag=args.session_tag,
    )


if __name__ == "__main__":
    main()

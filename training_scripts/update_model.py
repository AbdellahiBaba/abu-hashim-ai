import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PROCESSED_DIR = BASE_DIR / "dataset_processed"
MODEL_FINETUNE_DIR = BASE_DIR / "model_finetune"
LEARNING_BUFFER_DIR = BASE_DIR / "learning_buffer"
MERGED_DATA_FILE = DATASET_PROCESSED_DIR / "merged_training_data.jsonl"
UPDATE_LOG_FILE = MODEL_FINETUNE_DIR / "update_log.jsonl"


def count_merged_entries() -> int:
    if not MERGED_DATA_FILE.exists():
        return 0
    with open(MERGED_DATA_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def update_model(
    min_new_entries: int = 10,
    learning_rate: Optional[float] = None,
    num_epochs: Optional[int] = None,
    dry_run: bool = False,
    update_qalam_dataset: bool = False,
    qalam_buffer_dir: Optional[str] = None,
    qalam_quality_threshold: float = 0.6,
) -> dict:
    print("=" * 60)
    print("Abu Hashim Model Update")
    print("=" * 60)

    qalam_result = None
    if update_qalam_dataset:
        print("\n[Phase 0] Updating QalamAI dataset...")
        try:
            from qalam_bridge.update_dataset import update_dataset
            qalam_result = update_dataset(
                buffer_dir=qalam_buffer_dir,
                quality_threshold=qalam_quality_threshold,
            )
            print(f"  QalamAI dataset update: {qalam_result['status']}")
            print(f"  Records accepted: {qalam_result['stats']['records_accepted']}")
        except Exception as e:
            print(f"  Warning: QalamAI dataset update failed: {e}")
            qalam_result = {"status": "error", "reason": str(e)}

    from training_scripts.self_learning import run_self_learning_cycle
    print("\n[Phase 1] Running self-learning cycle...")
    learning_result = run_self_learning_cycle()

    entry_count = count_merged_entries()
    print(f"\nTotal merged training entries: {entry_count}")

    if entry_count < min_new_entries:
        msg = f"Not enough entries for training ({entry_count} < {min_new_entries}). Skipping."
        print(msg)
        return {
            "status": "skipped",
            "reason": msg,
            "entry_count": entry_count,
            "learning_result": learning_result,
            "qalam_result": qalam_result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    if dry_run:
        msg = "Dry run mode - skipping actual training."
        print(f"\n{msg}")
        return {
            "status": "dry_run",
            "reason": msg,
            "entry_count": entry_count,
            "learning_result": learning_result,
            "qalam_result": qalam_result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    print("\n[Phase 2] Preparing incremental training...")

    try:
        from model_base.config import get_default_config
        config = get_default_config()
    except ImportError:
        config = None
        print("Warning: Could not load model config, using defaults.")

    train_config = {
        "data_path": str(MERGED_DATA_FILE),
        "output_dir": str(MODEL_FINETUNE_DIR / "checkpoints" / datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
        "learning_rate": learning_rate or 1e-5,
        "num_epochs": num_epochs or 1,
        "entry_count": entry_count,
        "base_model": config.model_name if config else "CohereForAI/aya-23-8B",
    }

    print(f"Training config: {json.dumps(train_config, indent=2)}")

    print("\n[Phase 3] Launching incremental training...")
    training_result = _run_incremental_training(train_config)

    update_record = {
        "status": training_result.get("status", "completed"),
        "train_config": train_config,
        "learning_result": learning_result,
        "qalam_result": qalam_result,
        "training_result": training_result,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _log_update(update_record)

    print(f"\nModel update complete: {update_record['status']}")
    return update_record


def _run_incremental_training(train_config: dict) -> dict:
    try:
        from training_scripts.incremental_train import run_incremental_training
        result = run_incremental_training(
            data_path=train_config["data_path"],
            output_dir=train_config["output_dir"],
            learning_rate=train_config["learning_rate"],
            num_epochs=train_config["num_epochs"],
        )
        return {"status": "completed", "details": result}
    except ImportError:
        print("Warning: incremental_train module not available. Training skipped.")
        return {"status": "skipped", "reason": "incremental_train module not available"}
    except Exception as e:
        print(f"Error during training: {e}")
        return {"status": "error", "reason": str(e)}


def _log_update(record: dict):
    MODEL_FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    with open(UPDATE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def get_update_history() -> list:
    if not UPDATE_LOG_FILE.exists():
        return []
    history = []
    with open(UPDATE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Abu Hashim Model Update")
    parser.add_argument("--min-entries", type=int, default=10, help="Minimum entries required for training")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--dry-run", action="store_true", help="Run validation only, skip training")
    parser.add_argument("--update-qalam", action="store_true", help="Update QalamAI dataset before training")
    parser.add_argument("--qalam-buffer-dir", default=None, help="Directory with new QalamAI exports")
    parser.add_argument("--qalam-quality-threshold", type=float, default=0.6, help="Quality threshold for QalamAI data")
    args = parser.parse_args()

    result = update_model(
        min_new_entries=args.min_entries,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        dry_run=args.dry_run,
        update_qalam_dataset=args.update_qalam,
        qalam_buffer_dir=args.qalam_buffer_dir,
        qalam_quality_threshold=args.qalam_quality_threshold,
    )
    print(json.dumps(result, indent=2, default=str))

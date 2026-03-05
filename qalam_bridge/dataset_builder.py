"""Dataset builder and splitter for QalamAI processed data."""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from qalam_bridge.quality_scorer import QualityScorer
from training_scripts.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("dataset_processed/qalam_processed")
OUTPUT_DIR = Path("dataset_processed")


class DatasetBuilder:
    def __init__(
        self,
        processed_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        quality_threshold: float = 0.6,
        priority_threshold: float = 0.85,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_ratio = train_ratio
        self.seed = seed

        self.scorer = QualityScorer(
            threshold=quality_threshold,
            priority_threshold=priority_threshold,
        )
        self.formatter = DataFormatter()

        self.stats: Dict[str, Any] = {
            "total_records": 0,
            "scored_records": 0,
            "accepted_records": 0,
            "rejected_records": 0,
            "priority_records": 0,
            "train_records": 0,
            "eval_records": 0,
            "quality_distribution": {},
            "category_breakdown": {},
        }

    def load_processed_records(self) -> List[Dict]:
        records = []
        for path in sorted(self.processed_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON in %s", path)
            except OSError as e:
                logger.error("Error reading %s: %s", path, e)

        self.stats["total_records"] = len(records)
        logger.info("Loaded %d records from %s", len(records), self.processed_dir)
        return records

    def score_and_filter(self, records: List[Dict]) -> List[Dict]:
        result = self.scorer.filter_records(records)

        self.stats["scored_records"] = result["stats"]["total"]
        self.stats["accepted_records"] = result["stats"]["accepted"]
        self.stats["rejected_records"] = result["stats"]["rejected"]
        self.stats["priority_records"] = result["stats"]["priority"]
        self.stats["avg_quality"] = result["stats"]["avg_quality"]

        accepted = result["accepted"]

        self.stats["quality_distribution"] = self.scorer.get_quality_distribution(accepted)

        category_counts: Dict[str, int] = {}
        for rec in accepted:
            cat = rec.get("category", "general")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        self.stats["category_breakdown"] = category_counts

        logger.info(
            "Quality filtering: %d accepted, %d rejected, %d priority",
            len(accepted),
            result["stats"]["rejected"],
            result["stats"]["priority"],
        )
        return accepted

    def format_records(self, records: List[Dict]) -> List[Dict]:
        formatted = []
        for record in records:
            result = self.formatter.format_pair(
                record.get("input", ""),
                record.get("output", ""),
            )
            if result:
                result["category"] = record.get("category", "general")
                result["quality"] = record.get("quality", 0.0)
                formatted.append(result)
        return formatted

    def split_dataset(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        rng = random.Random(self.seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * self.train_ratio)
        train_set = shuffled[:split_idx]
        eval_set = shuffled[split_idx:]

        self.stats["train_records"] = len(train_set)
        self.stats["eval_records"] = len(eval_set)

        logger.info(
            "Split dataset: %d train, %d eval (ratio %.2f)",
            len(train_set),
            len(eval_set),
            self.train_ratio,
        )
        return {"train": train_set, "eval": eval_set}

    def save_datasets(self, splits: Dict[str, List[Dict]]) -> Dict[str, str]:
        paths = {}
        for split_name, records in splits.items():
            filename = f"{split_name}.jsonl"
            output_path = self.output_dir / filename
            with open(output_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            paths[split_name] = str(output_path)
            logger.info("Saved %d records to %s", len(records), output_path)
        return paths

    def build(self) -> Dict[str, Any]:
        records = self.load_processed_records()
        if not records:
            logger.warning("No records found in %s", self.processed_dir)
            return {"stats": self.stats, "paths": {}}

        accepted = self.score_and_filter(records)
        if not accepted:
            logger.warning("No records passed quality filtering")
            return {"stats": self.stats, "paths": {}}

        formatted = self.format_records(accepted)
        if not formatted:
            logger.warning("No records survived formatting")
            return {"stats": self.stats, "paths": {}}

        splits = self.split_dataset(formatted)
        paths = self.save_datasets(splits)

        return {"stats": self.stats, "paths": paths}

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)


def build_dataset(
    processed_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    quality_threshold: float = 0.6,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Dict[str, Any]:
    builder = DatasetBuilder(
        processed_dir=processed_dir,
        output_dir=output_dir,
        quality_threshold=quality_threshold,
        train_ratio=train_ratio,
        seed=seed,
    )
    return builder.build()


def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset from processed QalamAI data"
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory with processed QalamAI data (default: dataset_processed/qalam_processed/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save train/eval splits (default: dataset_processed/)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.6,
        help="Minimum quality score to include (default: 0.6)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train/eval split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = build_dataset(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        quality_threshold=args.quality_threshold,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    stats = result["stats"]
    paths = result["paths"]

    print("\nDataset build complete!")
    print(f"  Total records loaded:  {stats['total_records']}")
    print(f"  Accepted (quality):    {stats['accepted_records']}")
    print(f"  Rejected (quality):    {stats['rejected_records']}")
    print(f"  Priority examples:     {stats['priority_records']}")
    print(f"  Train set size:        {stats['train_records']}")
    print(f"  Eval set size:         {stats['eval_records']}")

    if stats.get("quality_distribution"):
        print("\n  Quality distribution:")
        for bucket, count in stats["quality_distribution"].items():
            print(f"    {bucket}: {count}")

    if stats.get("category_breakdown"):
        print("\n  Category breakdown:")
        for cat, count in stats["category_breakdown"].items():
            print(f"    {cat}: {count}")

    if paths:
        print("\n  Output files:")
        for name, path in paths.items():
            print(f"    {name}: {path}")


if __name__ == "__main__":
    main()

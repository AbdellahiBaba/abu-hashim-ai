"""Continuous learning pipeline — monitors and processes new QalamAI exports."""

import json
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from qalam_bridge.importer import QalamImporter, PROCESSED_DIR
from qalam_bridge.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LEARNING_BUFFER_DIR = BASE_DIR / "learning_buffer"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset_processed"
MAIN_DATASET_FILE = DATASET_PROCESSED_DIR / "qalam_training_data.jsonl"
UPDATE_LOG_FILE = DATASET_PROCESSED_DIR / "qalam_update_log.jsonl"

SUPPORTED_EXTENSIONS = {".json", ".csv", ".jsonl"}


def _content_hash(record: dict) -> str:
    content = (record.get("input", "") + record.get("output", "")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _load_existing_hashes(dataset_path: Path) -> set:
    hashes = set()
    if not dataset_path.exists():
        return hashes
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                h = rec.get("hash") or _content_hash(rec)
                hashes.add(h)
            except json.JSONDecodeError:
                pass
    return hashes


def _find_new_exports(buffer_dir: Path) -> list:
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        for p in sorted(buffer_dir.glob(f"*{ext}")):
            if p.name.startswith("."):
                continue
            files.append(p)
    return files


class DatasetUpdater:
    def __init__(
        self,
        buffer_dir: Optional[str] = None,
        quality_threshold: float = 0.6,
        priority_threshold: float = 0.85,
    ):
        self.buffer_dir = Path(buffer_dir) if buffer_dir else LEARNING_BUFFER_DIR
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        DATASET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        self.importer = QalamImporter()
        self.scorer = QualityScorer(
            threshold=quality_threshold,
            priority_threshold=priority_threshold,
        )
        self.quality_threshold = quality_threshold

        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "records_imported": 0,
            "records_scored": 0,
            "records_accepted": 0,
            "records_rejected": 0,
            "duplicates_removed": 0,
            "errors": 0,
        }

    def run(self) -> dict:
        logger.info("Starting dataset update from %s", self.buffer_dir)

        export_files = _find_new_exports(self.buffer_dir)
        self.stats["files_found"] = len(export_files)

        if not export_files:
            logger.info("No new export files found in %s", self.buffer_dir)
            return self._build_result("no_files")

        all_imported = []
        for file_path in export_files:
            try:
                records = self.importer.import_file(str(file_path))
                all_imported.extend(records)
                self.stats["files_processed"] += 1
                logger.info("Imported %d records from %s", len(records), file_path.name)
            except Exception as e:
                logger.error("Error importing %s: %s", file_path, e)
                self.stats["errors"] += 1

        self.stats["records_imported"] = len(all_imported)

        if not all_imported:
            logger.info("No records imported from export files")
            return self._build_result("no_records")

        filter_result = self.scorer.filter_records(all_imported)
        accepted = filter_result["accepted"]
        rejected = filter_result["rejected"]

        self.stats["records_scored"] = len(all_imported)
        self.stats["records_rejected"] = len(rejected)

        existing_hashes = _load_existing_hashes(MAIN_DATASET_FILE)
        deduplicated = []
        for record in accepted:
            h = record.get("hash") or _content_hash(record)
            if h in existing_hashes:
                self.stats["duplicates_removed"] += 1
                continue
            existing_hashes.add(h)
            deduplicated.append(record)

        self.stats["records_accepted"] = len(deduplicated)

        if deduplicated:
            self._append_to_dataset(deduplicated)

        self._log_update()

        result = self._build_result("completed")
        logger.info("Dataset update complete: %s", json.dumps(self.stats))
        return result

    def _append_to_dataset(self, records: list):
        with open(MAIN_DATASET_FILE, "a", encoding="utf-8") as f:
            for record in records:
                clean_record = {
                    "input": record["input"],
                    "output": record["output"],
                    "category": record.get("category", "general"),
                    "quality": record.get("quality"),
                    "hash": record.get("hash") or _content_hash(record),
                    "metadata": record.get("metadata", {}),
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
        logger.info("Appended %d records to %s", len(records), MAIN_DATASET_FILE)

    def _log_update(self):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": dict(self.stats),
        }
        with open(UPDATE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _build_result(self, status: str) -> dict:
        return {
            "status": status,
            "stats": dict(self.stats),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(MAIN_DATASET_FILE),
        }

    def get_stats(self) -> dict:
        return dict(self.stats)


def update_dataset(
    buffer_dir: Optional[str] = None,
    quality_threshold: float = 0.6,
    priority_threshold: float = 0.85,
) -> dict:
    updater = DatasetUpdater(
        buffer_dir=buffer_dir,
        quality_threshold=quality_threshold,
        priority_threshold=priority_threshold,
    )
    return updater.run()


def get_dataset_info() -> dict:
    info = {
        "dataset_path": str(MAIN_DATASET_FILE),
        "exists": MAIN_DATASET_FILE.exists(),
        "total_records": 0,
        "categories": {},
        "quality_distribution": {
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "poor": 0,
        },
    }

    if not MAIN_DATASET_FILE.exists():
        return info

    with open(MAIN_DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                info["total_records"] += 1

                cat = rec.get("category", "general")
                info["categories"][cat] = info["categories"].get(cat, 0) + 1

                q = rec.get("quality") or 0.0
                if q >= 0.85:
                    info["quality_distribution"]["excellent"] += 1
                elif q >= 0.7:
                    info["quality_distribution"]["good"] += 1
                elif q >= 0.6:
                    info["quality_distribution"]["acceptable"] += 1
                else:
                    info["quality_distribution"]["poor"] += 1
            except json.JSONDecodeError:
                pass

    return info


def get_update_history() -> list:
    if not UPDATE_LOG_FILE.exists():
        return []
    history = []
    with open(UPDATE_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return history


def main():
    parser = argparse.ArgumentParser(
        description="Update Abu Hashim dataset with new QalamAI exports"
    )
    parser.add_argument(
        "--buffer-dir",
        default=None,
        help="Directory to scan for new exports (default: learning_buffer/)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.6,
        help="Minimum quality score to accept (default: 0.6)",
    )
    parser.add_argument(
        "--priority-threshold",
        type=float,
        default=0.85,
        help="Quality score for priority flagging (default: 0.85)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    result = update_dataset(
        buffer_dir=args.buffer_dir,
        quality_threshold=args.quality_threshold,
        priority_threshold=args.priority_threshold,
    )

    print(f"\nDataset update complete!")
    print(f"  Status:            {result['status']}")
    stats = result["stats"]
    print(f"  Files found:       {stats['files_found']}")
    print(f"  Files processed:   {stats['files_processed']}")
    print(f"  Records imported:  {stats['records_imported']}")
    print(f"  Records accepted:  {stats['records_accepted']}")
    print(f"  Records rejected:  {stats['records_rejected']}")
    print(f"  Duplicates removed:{stats['duplicates_removed']}")
    print(f"  Errors:            {stats['errors']}")
    print(f"  Dataset path:      {result['dataset_path']}")


if __name__ == "__main__":
    main()

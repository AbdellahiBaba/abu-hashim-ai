import json
import csv
import hashlib
import logging
import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from training_scripts.text_cleaner import TextCleaner
from training_scripts.pii_remover import PIIRemover

logger = logging.getLogger(__name__)

RAW_EXPORT_DIR = Path("dataset_raw/qalam_exports")
PROCESSED_DIR = Path("dataset_processed/qalam_processed")

KNOWN_CATEGORIES = {
    "novel", "article", "script", "academic", "poetry",
    "essay", "dialogue", "summary", "translation", "general",
}

FIELD_ALIASES = {
    "input": ["input", "prompt", "user_prompt", "question", "user", "user_message"],
    "output": ["output", "response", "gpt_response", "answer", "assistant", "assistant_message", "completion"],
    "category": ["category", "type", "genre", "task_type"],
    "correction": ["correction", "corrected", "edited_response", "human_correction"],
    "timestamp": ["timestamp", "created_at", "date", "time", "datetime"],
    "writing_sample": ["writing_sample", "sample", "text", "content"],
}


def _resolve_field(record: dict, field_name: str) -> Optional[str]:
    for alias in FIELD_ALIASES.get(field_name, [field_name]):
        value = record.get(alias)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _record_hash(record: dict) -> str:
    content = (record.get("input", "") + record.get("output", "")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


class QalamImporter:
    def __init__(
        self,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_EXPORT_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = TextCleaner()
        self.pii_remover = PIIRemover()

        self.stats = {
            "files_imported": 0,
            "records_read": 0,
            "records_accepted": 0,
            "records_rejected": 0,
            "pii_detections": 0,
            "duplicates_skipped": 0,
        }
        self._seen_hashes: set = set()
        self._load_existing_hashes()

    def _load_existing_hashes(self):
        for path in self.processed_dir.glob("*.jsonl"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            h = rec.get("hash") or _record_hash(rec)
                            self._seen_hashes.add(h)
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

    def import_file(self, file_path: str) -> list:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in {".json", ".csv", ".jsonl"}:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .json, .csv, .jsonl")

        self._save_raw_copy(path)

        raw_records = self._load_file(path, ext)
        self.stats["records_read"] += len(raw_records)

        processed = []
        for raw in raw_records:
            record = self._normalize_record(raw)
            if record is None:
                self.stats["records_rejected"] += 1
                continue

            rec_hash = _record_hash(record)
            if rec_hash in self._seen_hashes:
                self.stats["duplicates_skipped"] += 1
                continue

            record["hash"] = rec_hash
            self._seen_hashes.add(rec_hash)
            processed.append(record)
            self.stats["records_accepted"] += 1

        self.stats["files_imported"] += 1

        if processed:
            self._save_processed(processed, path.stem)

        logger.info(
            "Imported %s: %d accepted, %d rejected, %d duplicates",
            path.name,
            len(processed),
            self.stats["records_rejected"],
            self.stats["duplicates_skipped"],
        )
        return processed

    def import_directory(self, dir_path: str) -> list:
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_records = []
        for ext in ("*.json", "*.csv", "*.jsonl"):
            for file_path in sorted(directory.glob(ext)):
                if file_path.name.startswith("."):
                    continue
                try:
                    records = self.import_file(str(file_path))
                    all_records.extend(records)
                except Exception as e:
                    logger.error("Error importing %s: %s", file_path, e)

        return all_records

    def _save_raw_copy(self, source_path: Path):
        dest = self.raw_dir / source_path.name
        if dest.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dest = self.raw_dir / f"{stem}_{ts}{suffix}"
        shutil.copy2(str(source_path), str(dest))
        logger.info("Saved raw copy to %s", dest)

    def _load_file(self, path: Path, ext: str) -> list:
        if ext == ".json":
            return self._load_json(path)
        elif ext == ".jsonl":
            return self._load_jsonl(path)
        elif ext == ".csv":
            return self._load_csv(path)
        return []

    def _load_json(self, path: Path) -> list:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "records" in data and isinstance(data["records"], list):
                return data["records"]
            if "interactions" in data and isinstance(data["interactions"], list):
                return data["interactions"]
            return [data]
        return []

    def _load_jsonl(self, path: Path) -> list:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON at %s:%d", path, line_num)
        return records

    def _load_csv(self, path: Path) -> list:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    def _normalize_record(self, raw: dict) -> Optional[dict]:
        input_text = _resolve_field(raw, "input")
        output_text = _resolve_field(raw, "output")

        correction = _resolve_field(raw, "correction")
        if correction:
            output_text = correction

        writing_sample = _resolve_field(raw, "writing_sample")
        if not input_text and not output_text and writing_sample:
            input_text = "اكتب نصاً إبداعياً"
            output_text = writing_sample

        if not input_text or not output_text:
            return None

        has_pii = (
            self.pii_remover.has_pii(input_text)
            or self.pii_remover.has_pii(output_text)
        )
        if has_pii:
            self.stats["pii_detections"] += 1

        input_text = self.cleaner.clean(input_text)
        output_text = self.cleaner.clean(output_text)
        input_text = self.pii_remover.remove_pii(input_text)
        output_text = self.pii_remover.remove_pii(output_text)

        if len(input_text) < 5 or len(output_text) < 10:
            return None

        category = _resolve_field(raw, "category") or "general"
        category = category.lower().strip()
        if category not in KNOWN_CATEGORIES:
            category = "general"

        timestamp = _resolve_field(raw, "timestamp")

        metadata = {}
        for key in ("source", "model", "session_id", "language", "tags"):
            val = raw.get(key)
            if val is not None:
                metadata[key] = val
        if timestamp:
            metadata["timestamp"] = timestamp

        return {
            "input": input_text,
            "output": output_text,
            "category": category,
            "quality": None,
            "metadata": metadata,
        }

    def _save_processed(self, records: list, source_name: str):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}_{ts}.jsonl"
        output_path = self.processed_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info("Saved %d processed records to %s", len(records), output_path)

    def import_single_record(self, raw: dict, quality_threshold: float = 0.6) -> dict:
        from qalam_bridge.quality_scorer import QualityScorer

        self.stats["records_read"] += 1

        record = self._normalize_record(raw)
        if record is None:
            self.stats["records_rejected"] += 1
            return {"status": "rejected", "quality": None, "hash": None}

        rec_hash = _record_hash(record)
        if rec_hash in self._seen_hashes:
            self.stats["duplicates_skipped"] += 1
            return {"status": "duplicate", "quality": None, "hash": rec_hash}

        scorer = QualityScorer(threshold=quality_threshold)
        record = scorer.score_record(record)
        quality = record.get("quality", 0.0)

        if quality < quality_threshold:
            self.stats["records_rejected"] += 1
            logger.info("Webhook record rejected: hash=%s quality=%.3f (threshold=%.2f)", rec_hash, quality, quality_threshold)
            return {"status": "rejected", "quality": quality, "hash": rec_hash}

        record["hash"] = rec_hash
        self._seen_hashes.add(rec_hash)

        webhook_file = self.processed_dir / "webhook_live.jsonl"
        with open(webhook_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.stats["records_accepted"] += 1
        logger.info("Webhook record accepted: hash=%s quality=%.3f", rec_hash, quality)

        return {
            "status": "accepted",
            "quality": quality,
            "hash": rec_hash,
        }

    def get_stats(self) -> dict:
        return dict(self.stats)


def import_qalam_file(file_path: str, **kwargs) -> list:
    importer = QalamImporter(**kwargs)
    return importer.import_file(file_path)


def import_qalam_directory(dir_path: str, **kwargs) -> list:
    importer = QalamImporter(**kwargs)
    return importer.import_directory(dir_path)


def main():
    parser = argparse.ArgumentParser(
        description="Import QalamAI.net exports into Abu Hashim training pipeline"
    )
    parser.add_argument(
        "path",
        help="Path to a QalamAI export file (.json, .csv, .jsonl) or directory",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Directory to store raw exports (default: dataset_raw/qalam_exports/)",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        help="Directory to store processed data (default: dataset_processed/qalam_processed/)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    importer = QalamImporter(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
    )

    path = Path(args.path)
    if path.is_dir():
        records = importer.import_directory(str(path))
    else:
        records = importer.import_file(str(path))

    stats = importer.get_stats()
    print(f"\nImport complete!")
    print(f"  Files imported:     {stats['files_imported']}")
    print(f"  Records read:       {stats['records_read']}")
    print(f"  Records accepted:   {stats['records_accepted']}")
    print(f"  Records rejected:   {stats['records_rejected']}")
    print(f"  Duplicates skipped: {stats['duplicates_skipped']}")
    print(f"  PII detections:     {stats['pii_detections']}")
    print(f"  Output records:     {len(records)}")


if __name__ == "__main__":
    main()

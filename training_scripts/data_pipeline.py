import json
import csv
import logging
from pathlib import Path
from typing import Optional

from training_scripts.text_cleaner import TextCleaner
from training_scripts.data_formatter import DataFormatter
from training_scripts.pii_remover import PIIRemover

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("dataset_raw")
PROCESSED_DATA_DIR = Path("dataset_processed")

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".txt", ".parquet"}


class DataPipeline:
    def __init__(
        self,
        raw_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        min_quality_score: float = 0.3,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = TextCleaner()
        self.pii_remover = PIIRemover()
        self.formatter = DataFormatter(
            text_cleaner=self.cleaner,
            pii_remover=self.pii_remover,
        )
        self.min_quality_score = min_quality_score

        self.stats = {
            "files_processed": 0,
            "records_read": 0,
            "records_accepted": 0,
            "records_rejected": 0,
            "pii_detections": 0,
        }

    def run(self) -> dict:
        logger.info("Starting data pipeline from %s", self.raw_dir)

        if not self.raw_dir.exists():
            logger.warning("Raw data directory does not exist: %s", self.raw_dir)
            return self.stats

        all_records = []
        for file_path in sorted(self.raw_dir.rglob("*")):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if file_path.name.startswith("."):
                continue

            try:
                records = self._load_file(file_path)
                self.stats["files_processed"] += 1
                self.stats["records_read"] += len(records)

                for record in records:
                    if self._quality_check(record):
                        all_records.append(record)
                        self.stats["records_accepted"] += 1
                    else:
                        self.stats["records_rejected"] += 1

            except Exception as e:
                logger.error("Error processing %s: %s", file_path, e)

        if all_records:
            output_path = self.processed_dir / "training_data.jsonl"
            count = self.formatter.save_jsonl(all_records, str(output_path))
            logger.info("Saved %d records to %s", count, output_path)

        logger.info("Pipeline complete: %s", self.stats)
        return self.stats

    def _load_file(self, file_path: Path) -> list:
        ext = file_path.suffix.lower()
        logger.info("Loading %s", file_path)

        if ext == ".jsonl":
            return self._load_jsonl(file_path)
        elif ext == ".json":
            return self._load_json(file_path)
        elif ext == ".csv":
            return self._load_csv(file_path)
        elif ext == ".txt":
            return self._load_txt(file_path)
        elif ext == ".parquet":
            return self._load_parquet(file_path)
        return []

    def _load_jsonl(self, file_path: Path) -> list:
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    formatted = self._format_record(data)
                    if formatted:
                        records.append(formatted)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON at %s:%d", file_path, line_num)
        return records

    def _load_json(self, file_path: Path) -> list:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            records = []
            for item in data:
                formatted = self._format_record(item)
                if formatted:
                    records.append(formatted)
            return records
        elif isinstance(data, dict):
            formatted = self._format_record(data)
            return [formatted] if formatted else []
        return []

    def _load_csv(self, file_path: Path) -> list:
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                formatted = self._format_record(dict(row))
                if formatted:
                    records.append(formatted)
        return records

    def _load_txt(self, file_path: Path) -> list:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.formatter.format_plain_text(text)

    def _load_parquet(self, file_path: Path) -> list:
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            records = []
            for _, row in df.iterrows():
                formatted = self._format_record(row.to_dict())
                if formatted:
                    records.append(formatted)
            return records
        except ImportError:
            logger.error("pandas/pyarrow not installed; cannot load parquet files")
            return []

    def _format_record(self, data: dict) -> Optional[dict]:
        input_text = data.get("input", data.get("prompt", data.get("question", "")))
        output_text = data.get("output", data.get("response", data.get("answer", data.get("completion", ""))))
        system_prompt = data.get("system", data.get("system_prompt", None))

        if not input_text or not output_text:
            return None

        if self.pii_remover.has_pii(str(input_text)) or self.pii_remover.has_pii(str(output_text)):
            self.stats["pii_detections"] += 1

        return self.formatter.format_pair(
            str(input_text), str(output_text), system_prompt
        )

    def _quality_check(self, record: dict) -> bool:
        text = record.get("text", "")
        if not text:
            return False

        if len(text) < 20:
            return False

        unique_chars = len(set(text))
        if unique_chars < 5:
            return False

        return True

    def add_file(self, file_path: str) -> list:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return self._load_file(path)

    def get_stats(self) -> dict:
        return dict(self.stats)


def run_pipeline(raw_dir: Optional[str] = None, processed_dir: Optional[str] = None) -> dict:
    pipeline = DataPipeline(raw_dir=raw_dir, processed_dir=processed_dir)
    return pipeline.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    stats = run_pipeline()
    print(f"Pipeline finished: {stats}")

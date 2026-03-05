import os
import json
import re
import uuid
import shutil
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LEARNING_BUFFER_DIR = BASE_DIR / "learning_buffer"
PENDING_DIR = LEARNING_BUFFER_DIR / "pending"
VALIDATED_DIR = LEARNING_BUFFER_DIR / "validated"
REJECTED_DIR = LEARNING_BUFFER_DIR / "rejected"
METADATA_FILE = LEARNING_BUFFER_DIR / "metadata.jsonl"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset_processed"


@dataclass
class LearningEntry:
    instruction: str
    response: str
    source: str = "user_feedback"
    corrected_response: Optional[str] = None
    feedback_score: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "pending"
    metadata: dict = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)


PII_PATTERNS = [
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    re.compile(r'\b(?:\+?966|05)\d{8,9}\b'),
    re.compile(r'\b(?:\+?20|01)\d{9,10}\b'),
    re.compile(r'\b(?:\+?971|05)\d{8,9}\b'),
    re.compile(r'\b\d{10,12}\b'),
]

ARABIC_CHAR_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

MIN_RESPONSE_LENGTH = 20
MIN_ARABIC_RATIO = 0.3
MAX_SIMILARITY_THRESHOLD = 0.95
MIN_FEEDBACK_SCORE = 0


def ensure_directories():
    for d in [PENDING_DIR, VALIDATED_DIR, REJECTED_DIR, DATASET_PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def add_entry(instruction: str, response: str, source: str = "user_feedback",
              corrected_response: Optional[str] = None, feedback_score: int = 1,
              metadata: Optional[dict] = None) -> LearningEntry:
    ensure_directories()
    entry = LearningEntry(
        instruction=instruction,
        response=response,
        source=source,
        corrected_response=corrected_response,
        feedback_score=feedback_score,
        metadata=metadata or {},
    )
    entry_path = PENDING_DIR / f"{entry.id}.json"
    with open(entry_path, "w", encoding="utf-8") as f:
        json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
    _append_metadata(entry)
    return entry


def _append_metadata(entry: LearningEntry):
    with open(METADATA_FILE, "a", encoding="utf-8") as f:
        record = {"id": entry.id, "timestamp": entry.timestamp, "status": entry.status, "source": entry.source}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def check_pii(text: str) -> List[str]:
    found = []
    for pattern in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            found.extend(matches)
    return found


def remove_pii(text: str) -> str:
    cleaned = text
    for pattern in PII_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def check_arabic_content(text: str) -> float:
    if not text:
        return 0.0
    arabic_chars = len(ARABIC_CHAR_PATTERN.findall(text))
    total_chars = len(text.replace(" ", "").replace("\n", ""))
    if total_chars == 0:
        return 0.0
    return arabic_chars / total_chars


def compute_text_hash(text: str) -> str:
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_existing_hashes() -> set:
    hashes = set()
    for json_file in VALIDATED_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            combined = data.get("instruction", "") + " " + data.get("response", "")
            hashes.add(compute_text_hash(combined))
        except (json.JSONDecodeError, KeyError):
            continue
    merged_file = DATASET_PROCESSED_DIR / "merged_training_data.jsonl"
    if merged_file.exists():
        with open(merged_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    combined = data.get("instruction", "") + " " + data.get("output", "")
                    hashes.add(compute_text_hash(combined))
                except (json.JSONDecodeError, KeyError):
                    continue
    return hashes


def validate_entry(entry: LearningEntry, existing_hashes: set) -> tuple:
    reasons = []
    effective_response = entry.corrected_response or entry.response

    if len(effective_response.strip()) < MIN_RESPONSE_LENGTH:
        reasons.append(f"Response too short ({len(effective_response.strip())} < {MIN_RESPONSE_LENGTH})")

    arabic_ratio = check_arabic_content(effective_response)
    if arabic_ratio < MIN_ARABIC_RATIO:
        reasons.append(f"Insufficient Arabic content (ratio={arabic_ratio:.2f} < {MIN_ARABIC_RATIO})")

    pii_found = check_pii(entry.instruction + " " + effective_response)
    if pii_found:
        reasons.append(f"PII detected: {len(pii_found)} pattern(s) found")

    if entry.feedback_score <= MIN_FEEDBACK_SCORE:
        reasons.append(f"Low feedback score ({entry.feedback_score} <= {MIN_FEEDBACK_SCORE})")

    combined = entry.instruction + " " + effective_response
    text_hash = compute_text_hash(combined)
    if text_hash in existing_hashes:
        reasons.append("Duplicate entry detected")

    is_valid = len(reasons) == 0
    return is_valid, reasons, text_hash


def validate_pending_entries() -> dict:
    ensure_directories()
    existing_hashes = load_existing_hashes()

    stats = {"total": 0, "validated": 0, "rejected": 0, "errors": 0}
    pending_files = list(PENDING_DIR.glob("*.json"))
    stats["total"] = len(pending_files)

    for entry_path in pending_files:
        try:
            with open(entry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry = LearningEntry(**data)
        except (json.JSONDecodeError, TypeError) as e:
            stats["errors"] += 1
            print(f"Error loading {entry_path.name}: {e}")
            continue

        is_valid, reasons, text_hash = validate_entry(entry, existing_hashes)

        if is_valid:
            entry.status = "validated"
            effective_response = entry.corrected_response or entry.response
            entry.response = remove_pii(effective_response)
            entry.instruction = remove_pii(entry.instruction)
            entry.corrected_response = None

            dest = VALIDATED_DIR / entry_path.name
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
            entry_path.unlink()
            existing_hashes.add(text_hash)
            stats["validated"] += 1
        else:
            entry.status = "rejected"
            entry.rejection_reasons = reasons

            dest = REJECTED_DIR / entry_path.name
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
            entry_path.unlink()
            stats["rejected"] += 1

    print(f"Validation complete: {stats}")
    return stats


def merge_validated_to_dataset() -> int:
    ensure_directories()
    merged_file = DATASET_PROCESSED_DIR / "merged_training_data.jsonl"
    validated_files = list(VALIDATED_DIR.glob("*.json"))

    if not validated_files:
        print("No validated entries to merge.")
        return 0

    count = 0
    with open(merged_file, "a", encoding="utf-8") as out:
        for vf in validated_files:
            try:
                with open(vf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                training_record = {
                    "instruction": data["instruction"],
                    "input": "",
                    "output": data["response"],
                    "source": data.get("source", "user_feedback"),
                    "added_at": datetime.utcnow().isoformat(),
                }
                out.write(json.dumps(training_record, ensure_ascii=False) + "\n")
                count += 1

                archive_dir = VALIDATED_DIR / "merged"
                archive_dir.mkdir(exist_ok=True)
                shutil.move(str(vf), str(archive_dir / vf.name))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error merging {vf.name}: {e}")
                continue

    print(f"Merged {count} entries into training dataset.")
    return count


def run_self_learning_cycle() -> dict:
    print("=" * 60)
    print("Starting self-learning cycle...")
    print("=" * 60)

    print("\n[Step 1] Validating pending entries...")
    validation_stats = validate_pending_entries()

    print("\n[Step 2] Merging validated entries to dataset...")
    merged_count = merge_validated_to_dataset()

    result = {
        "validation": validation_stats,
        "merged_count": merged_count,
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"\nSelf-learning cycle complete: {json.dumps(result, indent=2)}")
    return result


if __name__ == "__main__":
    run_self_learning_cycle()

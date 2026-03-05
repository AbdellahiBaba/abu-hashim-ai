"""Quality scoring and filtering for QalamAI imported data."""

import re
from typing import Any, Dict, List, Optional

from training_scripts.text_cleaner import TextCleaner


CATEGORY_EXPECTATIONS = {
    "novel": {
        "min_output_length": 200,
        "ideal_output_length": 1000,
        "max_output_length": 50000,
        "min_sentences": 3,
        "expects_paragraphs": True,
    },
    "article": {
        "min_output_length": 150,
        "ideal_output_length": 800,
        "max_output_length": 10000,
        "min_sentences": 2,
        "expects_paragraphs": True,
    },
    "script": {
        "min_output_length": 100,
        "ideal_output_length": 500,
        "max_output_length": 20000,
        "min_sentences": 2,
        "expects_paragraphs": False,
    },
    "academic": {
        "min_output_length": 200,
        "ideal_output_length": 1200,
        "max_output_length": 15000,
        "min_sentences": 3,
        "expects_paragraphs": True,
    },
    "default": {
        "min_output_length": 50,
        "ideal_output_length": 400,
        "max_output_length": 20000,
        "min_sentences": 1,
        "expects_paragraphs": False,
    },
}

SENTENCE_END_PATTERN = re.compile(r"[.!?؟。\n]")
REPEATED_PHRASE_PATTERN = re.compile(r"(.{10,}?)\1{2,}")
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")


class QualityScorer:
    def __init__(
        self,
        threshold: float = 0.6,
        priority_threshold: float = 0.85,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.threshold = threshold
        self.priority_threshold = priority_threshold
        self.weights = weights or {
            "length": 0.20,
            "coherence": 0.25,
            "formatting": 0.15,
            "arabic_ratio": 0.25,
            "completeness": 0.15,
        }
        self._cleaner = TextCleaner()

    def score_record(self, record: Dict) -> Dict:
        input_text = record.get("input", "") or ""
        output_text = record.get("output", "") or ""
        category = record.get("category", "default") or "default"

        if not output_text.strip():
            record["quality"] = 0.0
            record["quality_details"] = {
                "length": 0.0,
                "coherence": 0.0,
                "formatting": 0.0,
                "arabic_ratio": 0.0,
                "completeness": 0.0,
                "flags": ["empty_output"],
            }
            return record

        expectations = CATEGORY_EXPECTATIONS.get(
            category, CATEGORY_EXPECTATIONS["default"]
        )

        flags = []

        length_score = self._score_length(output_text, expectations)
        coherence_score = self._score_coherence(output_text, expectations, flags)
        formatting_score = self._score_formatting(output_text, expectations)
        arabic_score = self._score_arabic_ratio(output_text, input_text)
        completeness_score = self._score_completeness(
            input_text, output_text, expectations, flags
        )

        composite = (
            self.weights["length"] * length_score
            + self.weights["coherence"] * coherence_score
            + self.weights["formatting"] * formatting_score
            + self.weights["arabic_ratio"] * arabic_score
            + self.weights["completeness"] * completeness_score
        )

        composite = round(max(0.0, min(1.0, composite)), 4)

        record["quality"] = composite
        record["quality_details"] = {
            "length": round(length_score, 4),
            "coherence": round(coherence_score, 4),
            "formatting": round(formatting_score, 4),
            "arabic_ratio": round(arabic_score, 4),
            "completeness": round(completeness_score, 4),
            "flags": flags,
        }

        if composite >= self.priority_threshold:
            record["quality_details"]["priority"] = True

        return record

    def _score_length(self, output: str, expectations: Dict) -> float:
        length = len(output.strip())
        min_len = expectations["min_output_length"]
        ideal_len = expectations["ideal_output_length"]
        max_len = expectations["max_output_length"]

        if length < min_len // 2:
            return 0.0
        if length < min_len:
            return 0.3 * (length / min_len)
        if length <= ideal_len:
            return 0.7 + 0.3 * ((length - min_len) / max(1, ideal_len - min_len))
        if length <= max_len:
            overshoot = (length - ideal_len) / max(1, max_len - ideal_len)
            return max(0.6, 1.0 - 0.4 * overshoot)
        return 0.3

    def _score_coherence(
        self, output: str, expectations: Dict, flags: List[str]
    ) -> float:
        score = 1.0

        sentences = [
            s.strip()
            for s in SENTENCE_END_PATTERN.split(output)
            if s.strip()
        ]
        num_sentences = len(sentences)
        min_sentences = expectations["min_sentences"]

        if num_sentences < min_sentences:
            score *= 0.5

        if REPEATED_PHRASE_PATTERN.search(output):
            flags.append("repeated_phrases")
            score *= 0.3

        words = output.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                flags.append("low_vocabulary_diversity")
                score *= 0.4

        if num_sentences >= 2:
            lengths = [len(s.split()) for s in sentences if s]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                if avg_len < 3:
                    score *= 0.6
                elif avg_len > 100:
                    score *= 0.7

        return max(0.0, min(1.0, score))

    def _score_formatting(self, output: str, expectations: Dict) -> float:
        score = 0.8

        paragraphs = PARAGRAPH_PATTERN.split(output)
        if expectations["expects_paragraphs"]:
            if len(paragraphs) >= 2:
                score = 1.0
            else:
                score = 0.5
        else:
            if output.strip():
                score = 0.9

        lines = output.split("\n")
        non_empty = [l for l in lines if l.strip()]
        if non_empty:
            avg_line_len = sum(len(l) for l in non_empty) / len(non_empty)
            if avg_line_len < 5:
                score *= 0.6

        return max(0.0, min(1.0, score))

    def _score_arabic_ratio(self, output: str, input_text: str) -> float:
        ratio = self._cleaner.get_arabic_ratio(output)
        if ratio >= 0.8:
            return 1.0
        if ratio >= 0.5:
            return 0.6 + 0.4 * ((ratio - 0.5) / 0.3)
        if ratio >= 0.2:
            return 0.3 + 0.3 * ((ratio - 0.2) / 0.3)

        input_ratio = self._cleaner.get_arabic_ratio(input_text)
        if input_ratio < 0.2:
            return max(0.5, ratio * 2)

        return max(0.0, ratio)

    def _score_completeness(
        self,
        input_text: str,
        output: str,
        expectations: Dict,
        flags: List[str],
    ) -> float:
        score = 1.0

        stripped = output.rstrip()
        if stripped and stripped[-1] not in ".!?؟\n\"'»)":
            incomplete_markers = ["...", "…", "و", "ثم", "لكن", "أو"]
            last_word = stripped.split()[-1] if stripped.split() else ""
            if last_word in incomplete_markers or stripped.endswith("...") or stripped.endswith("…"):
                flags.append("incomplete_response")
                score *= 0.5

        if not input_text.strip():
            flags.append("empty_input")
            score *= 0.6

        if len(output.strip()) < expectations["min_output_length"]:
            flags.append("too_short")
            score *= 0.5

        return max(0.0, min(1.0, score))

    def filter_records(self, records: List[Dict]) -> Dict[str, Any]:
        scored = [self.score_record(r) for r in records]

        accepted = []
        rejected = []
        priority = []

        for record in scored:
            quality = record.get("quality", 0.0)
            if quality >= self.priority_threshold:
                priority.append(record)
                accepted.append(record)
            elif quality >= self.threshold:
                accepted.append(record)
            else:
                rejected.append(record)

        return {
            "accepted": accepted,
            "rejected": rejected,
            "priority": priority,
            "stats": {
                "total": len(scored),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "priority": len(priority),
                "avg_quality": round(
                    sum(r.get("quality", 0) for r in scored) / max(1, len(scored)), 4
                ),
            },
        }

    def get_quality_distribution(self, records: List[Dict]) -> Dict[str, int]:
        buckets = {
            "excellent (0.85-1.0)": 0,
            "good (0.7-0.85)": 0,
            "acceptable (0.6-0.7)": 0,
            "poor (0.4-0.6)": 0,
            "very_poor (0-0.4)": 0,
        }
        for record in records:
            q = record.get("quality", 0.0)
            if q >= 0.85:
                buckets["excellent (0.85-1.0)"] += 1
            elif q >= 0.7:
                buckets["good (0.7-0.85)"] += 1
            elif q >= 0.6:
                buckets["acceptable (0.6-0.7)"] += 1
            elif q >= 0.4:
                buckets["poor (0.4-0.6)"] += 1
            else:
                buckets["very_poor (0-0.4)"] += 1
        return buckets


def score_records(
    records: List[Dict],
    threshold: float = 0.6,
    priority_threshold: float = 0.85,
) -> Dict[str, Any]:
    scorer = QualityScorer(
        threshold=threshold, priority_threshold=priority_threshold
    )
    return scorer.filter_records(records)

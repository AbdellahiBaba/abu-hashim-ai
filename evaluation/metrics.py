import re
import math
from collections import Counter
from typing import Dict, List, Optional


class ArabicFluencyMetric:
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
    DIACRITICS_PATTERN = re.compile(r'[\u064B-\u065F\u0670]')

    def compute(self, text: str) -> Dict[str, float]:
        if not text.strip():
            return {
                "arabic_ratio": 0.0,
                "avg_word_length": 0.0,
                "diacritics_ratio": 0.0,
                "fluency_score": 0.0,
            }

        chars = [c for c in text if not c.isspace()]
        arabic_chars = self.ARABIC_PATTERN.findall(text)
        arabic_ratio = len(arabic_chars) / max(len(chars), 1)

        words = text.split()
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)

        diacritics = self.DIACRITICS_PATTERN.findall(text)
        diacritics_ratio = len(diacritics) / max(len(arabic_chars), 1)

        word_length_score = min(avg_word_length / 6.0, 1.0)
        fluency_score = (arabic_ratio * 0.5) + (word_length_score * 0.3) + (min(diacritics_ratio * 5, 1.0) * 0.2)

        return {
            "arabic_ratio": round(arabic_ratio, 4),
            "avg_word_length": round(avg_word_length, 4),
            "diacritics_ratio": round(diacritics_ratio, 4),
            "fluency_score": round(fluency_score, 4),
        }


class StyleConsistencyMetric:
    def compute(self, texts: List[str]) -> Dict[str, float]:
        if not texts or len(texts) < 2:
            return {
                "vocabulary_consistency": 0.0,
                "length_consistency": 0.0,
                "style_score": 0.0,
            }

        word_sets = [set(t.split()) for t in texts]
        lengths = [len(t.split()) for t in texts]

        pairwise_overlaps = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                intersection = word_sets[i] & word_sets[j]
                if union:
                    pairwise_overlaps.append(len(intersection) / len(union))
        vocab_consistency = sum(pairwise_overlaps) / max(len(pairwise_overlaps), 1)

        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / max(mean_len, 1)
        length_consistency = max(0.0, 1.0 - cv)

        style_score = (vocab_consistency * 0.6) + (length_consistency * 0.4)

        return {
            "vocabulary_consistency": round(vocab_consistency, 4),
            "length_consistency": round(length_consistency, 4),
            "style_score": round(style_score, 4),
        }


class QualityMetric:
    def __init__(self):
        self.fluency_metric = ArabicFluencyMetric()

    def compute(self, generated: str, reference: Optional[str] = None) -> Dict[str, float]:
        if not generated.strip():
            return {
                "fluency": 0.0,
                "length_score": 0.0,
                "repetition_penalty": 0.0,
                "bleu_1": 0.0,
                "quality_score": 0.0,
            }

        fluency_result = self.fluency_metric.compute(generated)
        fluency = fluency_result["fluency_score"]

        words = generated.split()
        length_score = min(len(words) / 50.0, 1.0)

        word_counts = Counter(words)
        if words:
            max_freq = max(word_counts.values())
            repetition_penalty = 1.0 - (max_freq / len(words))
        else:
            repetition_penalty = 0.0

        bleu_1 = 0.0
        if reference and reference.strip():
            ref_tokens = set(reference.split())
            gen_tokens = set(generated.split())
            if gen_tokens:
                bleu_1 = len(ref_tokens & gen_tokens) / len(gen_tokens)

        if reference and reference.strip():
            quality_score = (fluency * 0.3) + (length_score * 0.2) + (repetition_penalty * 0.2) + (bleu_1 * 0.3)
        else:
            quality_score = (fluency * 0.4) + (length_score * 0.3) + (repetition_penalty * 0.3)

        return {
            "fluency": round(fluency, 4),
            "length_score": round(length_score, 4),
            "repetition_penalty": round(repetition_penalty, 4),
            "bleu_1": round(bleu_1, 4),
            "quality_score": round(quality_score, 4),
        }


class PerplexityMetric:
    def compute_from_logprobs(self, log_probs: List[float]) -> Dict[str, float]:
        if not log_probs:
            return {"perplexity": float("inf"), "avg_log_prob": 0.0}

        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-avg_log_prob)

        return {
            "perplexity": round(perplexity, 4),
            "avg_log_prob": round(avg_log_prob, 4),
        }

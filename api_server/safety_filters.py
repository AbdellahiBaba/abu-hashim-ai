import re
from typing import Tuple, List, Optional


BLOCKED_PATTERNS_AR = [
    r"كيف\s+(تصنع|أصنع|نصنع)\s+(قنبلة|متفجرات|سلاح)",
    r"طريقة\s+(صنع|تصنيع)\s+(قنبلة|متفجرات|سلاح|سم)",
    r"كيفية\s+(اختراق|قرصنة|تهكير)",
    r"طريقة\s+(انتحار|قتل\s+النفس)",
]

BLOCKED_PATTERNS_EN = [
    r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)",
    r"instructions\s+for\s+(making|building)\s+(a\s+)?(bomb|weapon|poison)",
    r"how\s+to\s+hack\s+into",
    r"how\s+to\s+(commit\s+)?suicide",
    r"how\s+to\s+(kill|murder|harm)\s+(someone|a\s+person|people)",
]

PROFANITY_PATTERNS = [
    r"(?i)\b(fuck|shit|damn|bitch|ass|bastard)\b",
]

SENSITIVE_TOPICS = [
    "terrorism", "extremism", "radicalization",
    "إرهاب", "تطرف", "تكفير",
]


class SafetyFilter:
    def __init__(self, enabled: bool = True, strict_mode: bool = False):
        self.enabled = enabled
        self.strict_mode = strict_mode
        self._compile_patterns()

    def _compile_patterns(self):
        all_patterns = BLOCKED_PATTERNS_AR + BLOCKED_PATTERNS_EN
        if self.strict_mode:
            all_patterns += PROFANITY_PATTERNS
        self.compiled_patterns = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in all_patterns
        ]

    def check_input(self, text: str) -> Tuple[bool, Optional[str]]:
        if not self.enabled:
            return True, None

        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return False, "Input contains content that violates safety guidelines."

        text_lower = text.lower()
        for topic in SENSITIVE_TOPICS:
            if topic in text_lower:
                if self.strict_mode:
                    return False, "Input touches on sensitive topics."

        return True, None

    def check_output(self, text: str) -> Tuple[bool, Optional[str]]:
        if not self.enabled:
            return True, None

        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return False, "Generated content violates safety guidelines."

        return True, None

    def filter_output(self, text: str) -> str:
        if not self.enabled:
            return text

        is_safe, reason = self.check_output(text)
        if not is_safe:
            return "عذراً، لا أستطيع تقديم هذا المحتوى لأنه يخالف إرشادات السلامة. (Sorry, I cannot provide this content as it violates safety guidelines.)"

        return text

    def sanitize_for_logging(self, text: str) -> str:
        sanitized = text[:500] if len(text) > 500 else text
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
        return sanitized


safety_filter = SafetyFilter(enabled=True, strict_mode=False)

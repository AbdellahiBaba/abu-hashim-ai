import re
from typing import Optional


PHONE_PATTERNS = [
    re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"),
    re.compile(r"\b\d{10,14}\b"),
]

EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

IP_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

NATIONAL_ID_PATTERN = re.compile(r"\b[12]\d{9}\b")

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
)

DATE_OF_BIRTH_PATTERN = re.compile(
    r"\b(?:born|مواليد|تاريخ الميلاد)\s*:?\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b",
    re.IGNORECASE,
)

REPLACEMENT_MAP = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ip": "[IP_ADDRESS]",
    "ssn": "[SSN]",
    "credit_card": "[CREDIT_CARD]",
    "national_id": "[NATIONAL_ID]",
    "url": "[URL]",
    "dob": "[DATE_OF_BIRTH]",
}


class PIIRemover:
    def __init__(
        self,
        remove_emails: bool = True,
        remove_phones: bool = True,
        remove_ips: bool = True,
        remove_ssns: bool = True,
        remove_credit_cards: bool = True,
        remove_national_ids: bool = True,
        remove_urls: bool = False,
        remove_dobs: bool = True,
        custom_patterns: Optional[dict] = None,
    ):
        self.rules = []

        if remove_emails:
            self.rules.append(("email", EMAIL_PATTERN))
        if remove_phones:
            for pattern in PHONE_PATTERNS:
                self.rules.append(("phone", pattern))
        if remove_ips:
            self.rules.append(("ip", IP_PATTERN))
        if remove_ssns:
            self.rules.append(("ssn", SSN_PATTERN))
        if remove_credit_cards:
            self.rules.append(("credit_card", CREDIT_CARD_PATTERN))
        if remove_national_ids:
            self.rules.append(("national_id", NATIONAL_ID_PATTERN))
        if remove_urls:
            self.rules.append(("url", URL_PATTERN))
        if remove_dobs:
            self.rules.append(("dob", DATE_OF_BIRTH_PATTERN))

        if custom_patterns:
            for label, pattern_str in custom_patterns.items():
                compiled = re.compile(pattern_str)
                self.rules.append((label, compiled))
                if label not in REPLACEMENT_MAP:
                    REPLACEMENT_MAP[label] = f"[{label.upper()}]"

    def remove_pii(self, text: Optional[str]) -> str:
        if not text:
            return ""

        for label, pattern in self.rules:
            replacement = REPLACEMENT_MAP.get(label, "[REDACTED]")
            text = pattern.sub(replacement, text)

        return text

    def detect_pii(self, text: str) -> list:
        findings = []
        if not text:
            return findings

        for label, pattern in self.rules:
            for match in pattern.finditer(text):
                findings.append({
                    "type": label,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                })

        return findings

    def has_pii(self, text: str) -> bool:
        if not text:
            return False
        for _, pattern in self.rules:
            if pattern.search(text):
                return True
        return False


def remove_pii(text: str, **kwargs) -> str:
    remover = PIIRemover(**kwargs)
    return remover.remove_pii(text)

import re
import unicodedata
from typing import Optional


ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0656-\u065F\u0670]")

TATWEEL = "\u0640"

ARABIC_PUNCTUATION_MAP = {
    "٫": ".",
    "٬": ",",
    "؛": ";",
    "؟": "?",
}

ALEF_VARIANTS = {
    "\u0622": "\u0627",  # ALEF WITH MADDA -> ALEF
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE -> ALEF
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW -> ALEF
}

REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
)

EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)

EXTRA_WHITESPACE = re.compile(r"[ \t]+")
EXTRA_NEWLINES = re.compile(r"\n{3,}")


class TextCleaner:
    def __init__(
        self,
        remove_diacritics: bool = True,
        remove_tatweel: bool = True,
        normalize_alef: bool = True,
        normalize_punctuation: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        collapse_whitespace: bool = True,
        max_repeated_chars: int = 2,
    ):
        self.remove_diacritics = remove_diacritics
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_punctuation = normalize_punctuation
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.collapse_whitespace = collapse_whitespace
        self.max_repeated_chars = max_repeated_chars

    def clean(self, text: Optional[str]) -> str:
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)

        if self.remove_urls:
            text = URL_PATTERN.sub("", text)

        if self.remove_emails:
            text = EMAIL_PATTERN.sub("", text)

        if self.remove_diacritics:
            text = ARABIC_DIACRITICS.sub("", text)

        if self.remove_tatweel:
            text = text.replace(TATWEEL, "")

        if self.normalize_alef:
            for variant, replacement in ALEF_VARIANTS.items():
                text = text.replace(variant, replacement)

        if self.normalize_punctuation:
            for arabic_punct, latin_punct in ARABIC_PUNCTUATION_MAP.items():
                text = text.replace(arabic_punct, latin_punct)

        if self.max_repeated_chars > 0:
            text = REPEATED_CHAR_PATTERN.sub(
                r"\1" * self.max_repeated_chars, text
            )

        text = self._remove_control_chars(text)

        if self.collapse_whitespace:
            text = EXTRA_WHITESPACE.sub(" ", text)
            text = EXTRA_NEWLINES.sub("\n\n", text)

        return text.strip()

    def _remove_control_chars(self, text: str) -> str:
        return "".join(
            ch for ch in text
            if ch in ("\n", "\r", "\t")
            or not unicodedata.category(ch).startswith("C")
        )

    def is_arabic(self, text: str) -> bool:
        if not text:
            return False
        arabic_chars = sum(
            1 for ch in text if "\u0600" <= ch <= "\u06FF"
        )
        total_alpha = sum(1 for ch in text if ch.isalpha())
        if total_alpha == 0:
            return False
        return (arabic_chars / total_alpha) > 0.5

    def get_arabic_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        arabic_chars = sum(
            1 for ch in text if "\u0600" <= ch <= "\u06FF"
        )
        total_alpha = sum(1 for ch in text if ch.isalpha())
        if total_alpha == 0:
            return 0.0
        return arabic_chars / total_alpha


def clean_text(text: str, **kwargs) -> str:
    cleaner = TextCleaner(**kwargs)
    return cleaner.clean(text)

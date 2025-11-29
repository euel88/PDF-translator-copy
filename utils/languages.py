"""
Language utilities for PDFMathTranslate.
"""
from typing import Dict, List, Tuple

# Comprehensive language code mapping
LANGUAGE_CODES: Dict[str, str] = {
    # Major languages
    "en": "English",
    "ko": "한국어 (Korean)",
    "zh": "简体中文 (Simplified Chinese)",
    "zh-TW": "繁體中文 (Traditional Chinese)",
    "ja": "日本語 (Japanese)",

    # European languages
    "fr": "Français (French)",
    "de": "Deutsch (German)",
    "es": "Español (Spanish)",
    "it": "Italiano (Italian)",
    "pt": "Português (Portuguese)",
    "nl": "Nederlands (Dutch)",
    "pl": "Polski (Polish)",
    "ru": "Русский (Russian)",
    "uk": "Українська (Ukrainian)",
    "cs": "Čeština (Czech)",
    "sv": "Svenska (Swedish)",
    "da": "Dansk (Danish)",
    "fi": "Suomi (Finnish)",
    "no": "Norsk (Norwegian)",
    "el": "Ελληνικά (Greek)",
    "hu": "Magyar (Hungarian)",
    "ro": "Română (Romanian)",
    "bg": "Български (Bulgarian)",
    "sk": "Slovenčina (Slovak)",
    "hr": "Hrvatski (Croatian)",

    # Asian languages
    "vi": "Tiếng Việt (Vietnamese)",
    "th": "ไทย (Thai)",
    "id": "Bahasa Indonesia (Indonesian)",
    "ms": "Bahasa Melayu (Malay)",
    "tl": "Tagalog (Filipino)",

    # Middle Eastern languages
    "ar": "العربية (Arabic)",
    "he": "עברית (Hebrew)",
    "fa": "فارسی (Persian)",
    "tr": "Türkçe (Turkish)",

    # South Asian languages
    "hi": "हिन्दी (Hindi)",
    "bn": "বাংলা (Bengali)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "ur": "اردو (Urdu)",
}

# Default source languages (commonly used)
DEFAULT_SOURCE_LANGS = ["en", "zh", "ja", "ko", "de", "fr", "es", "ru"]

# Default target languages
DEFAULT_TARGET_LANGS = ["ko", "en", "zh", "ja", "fr", "de", "es", "pt", "ru"]


def get_language_name(code: str) -> str:
    """Get the display name for a language code."""
    return LANGUAGE_CODES.get(code, code)


def get_language_code(name: str) -> str:
    """Get the language code from a display name."""
    for code, display_name in LANGUAGE_CODES.items():
        if display_name == name or code == name:
            return code
    return name


def get_language_pairs() -> List[Tuple[str, str]]:
    """Get list of (code, name) tuples for language selection."""
    return [(code, name) for code, name in LANGUAGE_CODES.items()]


def get_source_languages() -> Dict[str, str]:
    """Get dictionary of source languages for UI dropdown."""
    return {code: LANGUAGE_CODES[code] for code in DEFAULT_SOURCE_LANGS if code in LANGUAGE_CODES}


def get_target_languages() -> Dict[str, str]:
    """Get dictionary of target languages for UI dropdown."""
    return {code: LANGUAGE_CODES[code] for code in DEFAULT_TARGET_LANGS if code in LANGUAGE_CODES}


def get_all_languages() -> Dict[str, str]:
    """Get all available languages."""
    return LANGUAGE_CODES.copy()

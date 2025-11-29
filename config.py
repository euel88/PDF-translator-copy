"""
PDFMathTranslate - Central Configuration Module
Based on: https://github.com/PDFMathTranslate/PDFMathTranslate
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
from pathlib import Path
import os


class TranslationService(Enum):
    """Supported translation services."""
    GOOGLE = "google"
    OPENAI = "openai"
    DEEPL = "deepl"
    OLLAMA = "ollama"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


class OutputFormat(Enum):
    """Output PDF format options."""
    DUAL = "dual"       # Bilingual: original + translation
    MONO = "mono"       # Translation only
    BOTH = "both"       # Generate both versions


@dataclass
class TranslationConfig:
    """Configuration for PDF translation."""

    # Translation settings
    service: TranslationService = TranslationService.GOOGLE
    source_lang: str = "en"
    target_lang: str = "ko"

    # Output settings
    output_format: OutputFormat = OutputFormat.DUAL

    # API Keys (loaded from environment or user input)
    openai_api_key: Optional[str] = None
    deepl_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # OpenAI specific settings
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None

    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma2"

    # DeepSeek settings
    deepseek_api_key: Optional[str] = None
    deepseek_model: str = "deepseek-chat"

    # Processing settings
    thread_count: int = 4
    pages: Optional[str] = None  # e.g., "1-5,8,10-12"

    # Layout detection
    use_layout_detection: bool = True
    onnx_model_path: Optional[str] = None

    # Cache settings
    use_cache: bool = True
    cache_dir: str = ".cache/translations"

    # Custom prompt for LLM-based translation
    custom_prompt: Optional[str] = None

    def __post_init__(self):
        """Load API keys from environment variables if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.deepl_api_key is None:
            self.deepl_api_key = os.getenv("DEEPL_API_KEY")
        if self.azure_api_key is None:
            self.azure_api_key = os.getenv("AZURE_API_KEY")
        if self.gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.deepseek_api_key is None:
            self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    def to_pdf2zh_params(self) -> Dict[str, Any]:
        """Convert config to pdf2zh translation parameters."""
        params = {
            "lang_in": self.source_lang,
            "lang_out": self.target_lang,
            "service": self.service.value,
            "thread": self.thread_count,
        }

        if self.pages:
            params["pages"] = self.pages

        # Service-specific parameters
        if self.service == TranslationService.OPENAI:
            if self.openai_api_key:
                params["envs"] = {"OPENAI_API_KEY": self.openai_api_key}
            if self.openai_model:
                params["model"] = self.openai_model
            if self.openai_base_url:
                params["envs"] = params.get("envs", {})
                params["envs"]["OPENAI_BASE_URL"] = self.openai_base_url

        elif self.service == TranslationService.OLLAMA:
            params["model"] = self.ollama_model
            params["envs"] = {"OLLAMA_HOST": self.ollama_host}

        elif self.service == TranslationService.DEEPL:
            if self.deepl_api_key:
                params["envs"] = {"DEEPL_API_KEY": self.deepl_api_key}

        elif self.service == TranslationService.DEEPSEEK:
            if self.deepseek_api_key:
                params["envs"] = {"DEEPSEEK_API_KEY": self.deepseek_api_key}
            params["model"] = self.deepseek_model

        elif self.service == TranslationService.GEMINI:
            if self.gemini_api_key:
                params["envs"] = {"GEMINI_API_KEY": self.gemini_api_key}

        if self.custom_prompt:
            params["prompt"] = self.custom_prompt

        return params


@dataclass
class AppConfig:
    """Application-level configuration."""

    # App metadata
    app_name: str = "PDFMathTranslate"
    version: str = "2.0.0"

    # File handling
    max_file_size_mb: int = 200
    temp_dir: str = "/tmp/pdfmathtranslate"
    output_dir: str = "outputs"

    # UI settings
    page_title: str = "PDF Math Translator"
    page_icon: str = "📄"
    layout: str = "wide"

    # Feature flags
    enable_ocr: bool = False  # OCR support (requires additional deps)
    enable_cache: bool = True
    enable_progress: bool = True

    # Default translation config
    default_config: TranslationConfig = field(default_factory=TranslationConfig)

    def ensure_dirs(self):
        """Ensure required directories exist."""
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.default_config.use_cache:
            os.makedirs(self.default_config.cache_dir, exist_ok=True)


# Global app configuration instance
APP_CONFIG = AppConfig()


# Language configurations
LANGUAGES = {
    "en": "English",
    "ko": "한국어 (Korean)",
    "zh": "中文 (Chinese)",
    "zh-TW": "繁體中文 (Traditional Chinese)",
    "ja": "日本語 (Japanese)",
    "fr": "Français (French)",
    "de": "Deutsch (German)",
    "es": "Español (Spanish)",
    "it": "Italiano (Italian)",
    "pt": "Português (Portuguese)",
    "ru": "Русский (Russian)",
    "ar": "العربية (Arabic)",
    "hi": "हिन्दी (Hindi)",
    "vi": "Tiếng Việt (Vietnamese)",
    "th": "ไทย (Thai)",
}

# Translation service display names
SERVICE_NAMES = {
    TranslationService.GOOGLE: "Google Translate (Free)",
    TranslationService.OPENAI: "OpenAI GPT",
    TranslationService.DEEPL: "DeepL",
    TranslationService.OLLAMA: "Ollama (Local)",
    TranslationService.AZURE: "Azure Translator",
    TranslationService.DEEPSEEK: "DeepSeek",
    TranslationService.GEMINI: "Google Gemini",
}


def get_font_path(language: str) -> Optional[Path]:
    """Get font path for language."""
    font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"

    font_mapping = {
        "ko": "NanumGothic.ttf",
        "zh": "SourceHanSerifCN-Regular.ttf",
        "ja": "SourceHanSerifJP-Regular.ttf",
        "default": "GoNotoKurrent-Regular.ttf",
    }

    font_name = font_mapping.get(language, font_mapping["default"])
    font_path = font_dir / font_name

    return font_path if font_path.exists() else None

"""
설정 관리 모듈
"""
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    """앱 설정"""
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # 언어
    source_lang: str = "English"
    target_lang: str = "Korean"

    # OCR
    ocr_lang: str = "eng"
    ocr_confidence: float = 0.5

    # PDF
    render_dpi: int = 150

    # 폰트
    font_path: str = ""

    # 저장 경로
    _config_path: str = field(default="", repr=False)

    def __post_init__(self):
        if not self._config_path:
            self._config_path = str(Path.home() / ".pdf_translator" / "config.json")

    def save(self):
        """설정 저장"""
        path = Path(self._config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        del data["_config_path"]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "Config":
        """설정 로드"""
        path = Path.home() / ".pdf_translator" / "config.json"

        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data, _config_path=str(path))
            except Exception:
                pass

        return cls(_config_path=str(path))


# 지원 언어
LANGUAGES = {
    "English": "en",
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Vietnamese": "vi",
    "Thai": "th",
    "Arabic": "ar",
}

# Tesseract 언어 코드
TESSERACT_LANGS = {
    "English": "eng",
    "Korean": "kor",
    "Japanese": "jpn",
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Russian": "rus",
    "Portuguese": "por",
    "Italian": "ita",
    "Vietnamese": "vie",
    "Thai": "tha",
    "Arabic": "ara",
}

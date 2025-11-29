"""
설정 관리 모듈
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class AppConfig:
    """앱 설정"""
    # OpenAI 설정
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # OCR 설정
    ocr_lang: str = "en"  # 원본 언어
    ocr_confidence_threshold: float = 0.5

    # 번역 설정
    source_lang: str = "English"
    target_lang: str = "Korean"

    # PDF 처리 설정
    render_dpi: int = 150

    # 출력 설정
    output_dir: str = ""

    # 폰트 설정
    font_path: str = ""


class ConfigManager:
    """설정 관리자"""

    CONFIG_FILE = "config.json"

    def __init__(self):
        self.config_dir = Path.home() / ".pdf_translator_local"
        self.config_path = self.config_dir / self.CONFIG_FILE
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """설정 파일 로드"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return AppConfig(**data)
            except Exception:
                return AppConfig()
        return AppConfig()

    def save_config(self):
        """설정 파일 저장"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

    def get(self, key: str, default=None):
        """설정 값 가져오기"""
        return getattr(self.config, key, default)

    def set(self, key: str, value):
        """설정 값 설정"""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save_config()


# 언어 목록
LANGUAGES = {
    "Korean": "ko",
    "English": "en",
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
    "Indonesian": "id",
    "Arabic": "ar",
}

# OCR 언어 코드 (PaddleOCR)
OCR_LANGUAGES = {
    "English": "en",
    "Korean": "korean",
    "Japanese": "japan",
    "Chinese": "ch",
    "French": "french",
    "German": "german",
    "Russian": "ru",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Arabic": "ar",
}

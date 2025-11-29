"""
설정 관리 모듈 - PDFMathTranslate 구조 기반
"""
import os
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigManager:
    """설정 관리자 (싱글톤)"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._config_dir = Path.home() / ".config" / "PDFTranslator"
        self._config_file = self._config_dir / "config.json"
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """설정 로드"""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception:
                self._config = {}

    def _save(self):
        """설정 저장"""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._config_file, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 조회"""
        with self._lock:
            # 환경 변수 우선
            env_val = os.environ.get(key.upper())
            if env_val is not None:
                return env_val
            return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """설정 값 저장"""
        with self._lock:
            self._config[key] = value
            self._save()

    def delete(self, key: str):
        """설정 값 삭제"""
        with self._lock:
            if key in self._config:
                del self._config[key]
                self._save()

    def clear(self):
        """모든 설정 삭제"""
        with self._lock:
            self._config = {}
            self._save()

    # 번역기 설정
    def get_translator_env(self, name: str, key: str, default: str = "") -> str:
        """번역기 환경 변수 조회"""
        translators = self._config.get("translators", [])
        for t in translators:
            if t.get("name") == name:
                return t.get("envs", {}).get(key, default)
        return os.environ.get(key, default)

    def set_translator_env(self, name: str, key: str, value: str):
        """번역기 환경 변수 저장"""
        with self._lock:
            translators = self._config.setdefault("translators", [])
            for t in translators:
                if t.get("name") == name:
                    t.setdefault("envs", {})[key] = value
                    self._save()
                    return
            # 새 번역기 추가
            translators.append({"name": name, "envs": {key: value}})
            self._save()


# 전역 설정 인스턴스
config = ConfigManager()


# 지원 언어
LANGUAGES = {
    "English": "en",
    "Korean": "ko",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
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

# 번역 서비스
TRANSLATORS = [
    "openai",
    "google",
    "deepl",
    "ollama",
]

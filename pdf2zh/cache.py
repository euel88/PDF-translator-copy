"""
번역 캐시 모듈 - PDFMathTranslate 구조 기반
"""
import hashlib
import json
import threading
from pathlib import Path
from typing import Dict, Optional


class TranslationCache:
    """번역 캐시 (파일 기반)"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, cache_dir: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_dir: Optional[str] = None):
        if self._initialized:
            return
        self._initialized = True

        if cache_dir:
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = Path.home() / ".cache" / "PDFTranslator"

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, str] = {}

    def _make_key(self, text: str, source: str, target: str, service: str) -> str:
        """캐시 키 생성"""
        content = f"{text}|{source}|{target}|{service}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_file(self, key: str) -> Path:
        """캐시 파일 경로"""
        return self._cache_dir / f"{key[:2]}" / f"{key}.json"

    def get(self, text: str, source: str, target: str, service: str) -> Optional[str]:
        """캐시에서 번역 조회"""
        key = self._make_key(text, source, target, service)

        # 메모리 캐시 확인
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 파일 캐시 확인
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    translation = data.get("translation")
                    if translation:
                        self._memory_cache[key] = translation
                        return translation
            except Exception:
                pass

        return None

    def set(self, text: str, source: str, target: str, service: str, translation: str):
        """캐시에 번역 저장"""
        key = self._make_key(text, source, target, service)

        # 메모리 캐시
        self._memory_cache[key] = translation

        # 파일 캐시
        cache_file = self._get_cache_file(key)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({
                    "text": text,
                    "source": source,
                    "target": target,
                    "service": service,
                    "translation": translation,
                }, f, ensure_ascii=False)
        except Exception:
            pass

    def clear(self):
        """캐시 클리어"""
        with self._lock:
            self._memory_cache.clear()

            # 파일 캐시 삭제
            import shutil
            if self._cache_dir.exists():
                for item in self._cache_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)


# 전역 캐시 인스턴스
cache = TranslationCache()

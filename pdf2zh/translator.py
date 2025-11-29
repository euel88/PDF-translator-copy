"""
번역 모듈 - PDFMathTranslate 구조 기반
다중 번역 서비스 지원
"""
import re
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from pdf2zh.cache import cache
from pdf2zh.config import config


class BaseTranslator(ABC):
    """번역기 기본 클래스"""

    name: str = "base"

    # 수식 플레이스홀더
    FORMULA_PLACEHOLDER = "{{v{id}}}"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_cache = kwargs.get("use_cache", True)

    def get_formular_placeholder(self, id: int) -> str:
        """수식 플레이스홀더 생성"""
        return self.FORMULA_PLACEHOLDER.format(id=id)

    def translate(self, text: str) -> str:
        """단일 텍스트 번역"""
        if not text or not text.strip():
            return text

        # 캐시 확인
        if self.use_cache:
            cached = cache.get(text, self.source_lang, self.target_lang, self.name)
            if cached:
                return cached

        # 번역 수행
        result = self._translate(text)

        # 캐시 저장
        if self.use_cache and result:
            cache.set(text, self.source_lang, self.target_lang, self.name, result)

        return result

    def translate_batch(self, texts: List[str]) -> List[str]:
        """배치 번역"""
        results = []
        for text in texts:
            results.append(self.translate(text))
        return results

    @abstractmethod
    def _translate(self, text: str) -> str:
        """실제 번역 수행 (서브클래스에서 구현)"""
        pass


class OpenAITranslator(BaseTranslator):
    """OpenAI 번역기"""

    name = "openai"
    FORMULA_PLACEHOLDER = "{{{{v{id}}}}}"  # OpenAI용 이중 중괄호

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.api_key = kwargs.get("api_key") or config.get("OPENAI_API_KEY", "")
        self.model = kwargs.get("model", "gpt-4o-mini")
        self.base_url = kwargs.get("base_url") or config.get("OPENAI_BASE_URL")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {
                "api_key": self.api_key,
                "timeout": 120.0,  # 120초 타임아웃
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _get_prompt(self) -> str:
        return f"""You are a professional translator for technical documents.

Rules:
1. Translate from {self.source_lang} to {self.target_lang}
2. Keep formula placeholders like {{{{vN}}}} unchanged
3. Preserve numbers, units, and technical terms
4. Return ONLY the translation"""

    def _translate(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI] 번역 오류: {e}")
            return text

    def translate_batch(self, texts: List[str]) -> List[str]:
        """배치 번역 (최적화)"""
        if not texts:
            return []

        results = texts.copy()
        to_translate = []

        # 캐시 확인
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            if self.use_cache:
                cached = cache.get(text, self.source_lang, self.target_lang, self.name)
                if cached:
                    results[i] = cached
                    continue
            to_translate.append((i, text))

        if not to_translate:
            return results

        # 배치 요청
        numbered = "\n".join(f"[{idx+1}] {t}" for idx, (_, t) in enumerate(to_translate))
        prompt = f"""Translate each line. Keep [N] format and {{{{vN}}}} placeholders.

{numbered}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )

            result_text = response.choices[0].message.content.strip()
            translations = []

            for line in result_text.split("\n"):
                line = line.strip()
                if line:
                    cleaned = re.sub(r"^\[\d+\]\s*", "", line)
                    translations.append(cleaned)

            for idx, (orig_idx, orig_text) in enumerate(to_translate):
                if idx < len(translations):
                    results[orig_idx] = translations[idx]
                    if self.use_cache:
                        cache.set(
                            orig_text, self.source_lang, self.target_lang,
                            self.name, translations[idx]
                        )

        except Exception as e:
            print(f"[OpenAI] 배치 번역 오류: {e}")
            # 배치 실패 시 개별 번역으로 폴백
            print("[OpenAI] 개별 번역으로 재시도...")
            for orig_idx, orig_text in to_translate:
                try:
                    translated = self._translate(orig_text)
                    results[orig_idx] = translated
                    if self.use_cache:
                        cache.set(
                            orig_text, self.source_lang, self.target_lang,
                            self.name, translated
                        )
                except Exception as e2:
                    print(f"[OpenAI] 개별 번역 오류: {e2}")

        return results


class GoogleTranslator(BaseTranslator):
    """Google 번역기 (무료)"""

    name = "google"

    def _translate(self, text: str) -> str:
        try:
            from deep_translator import GoogleTranslator as GT
            translator = GT(source=self.source_lang, target=self.target_lang)
            return translator.translate(text)
        except Exception as e:
            print(f"[Google] 번역 오류: {e}")
            return text


class DeepLTranslator(BaseTranslator):
    """DeepL 번역기"""

    name = "deepl"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.api_key = kwargs.get("api_key") or config.get("DEEPL_API_KEY", "")

    def _translate(self, text: str) -> str:
        try:
            import deepl
            translator = deepl.Translator(self.api_key)
            result = translator.translate_text(
                text,
                source_lang=self.source_lang.upper(),
                target_lang=self.target_lang.upper(),
            )
            return result.text
        except Exception as e:
            print(f"[DeepL] 번역 오류: {e}")
            return text


class OllamaTranslator(BaseTranslator):
    """Ollama 로컬 번역기"""

    name = "ollama"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.model = kwargs.get("model", "llama3")
        self.host = kwargs.get("host") or config.get("OLLAMA_HOST", "http://localhost:11434")

    def _translate(self, text: str) -> str:
        try:
            import ollama
            prompt = f"Translate from {self.source_lang} to {self.target_lang}. Return only translation:\n{text}"
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[Ollama] 번역 오류: {e}")
            return text


# 번역기 팩토리
TRANSLATOR_CLASSES = {
    "openai": OpenAITranslator,
    "google": GoogleTranslator,
    "deepl": DeepLTranslator,
    "ollama": OllamaTranslator,
}


def create_translator(
    service: str,
    source_lang: str,
    target_lang: str,
    **kwargs
) -> BaseTranslator:
    """번역기 생성"""
    cls = TRANSLATOR_CLASSES.get(service.lower())
    if cls is None:
        raise ValueError(f"Unknown translator: {service}")
    return cls(source_lang, target_lang, **kwargs)

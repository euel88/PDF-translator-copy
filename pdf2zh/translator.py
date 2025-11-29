"""
번역 모듈 - PDFMathTranslate 구조 기반
다중 번역 서비스 지원, 멀티스레드 지원
"""
import re
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pdf2zh.cache import cache
from pdf2zh.config import config


class BaseTranslator(ABC):
    """번역기 기본 클래스 - 멀티스레드 지원"""

    name: str = "base"

    # 수식 플레이스홀더
    FORMULA_PLACEHOLDER = "{{v{id}}}"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_cache = kwargs.get("use_cache", True)
        self.num_threads = kwargs.get("num_threads", 1)
        self.custom_prompt = kwargs.get("custom_prompt", None)
        self._lock = threading.RLock()

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
            with self._lock:
                cache.set(text, self.source_lang, self.target_lang, self.name, result)

        return result

    def translate_batch(self, texts: List[str]) -> List[str]:
        """배치 번역 - 멀티스레드 지원"""
        if not texts:
            return []

        # 단일 스레드 또는 적은 수의 텍스트
        if self.num_threads <= 1 or len(texts) <= 3:
            return self._translate_batch_sequential(texts)

        # 멀티스레드 번역
        return self._translate_batch_parallel(texts)

    def _translate_batch_sequential(self, texts: List[str]) -> List[str]:
        """순차 배치 번역"""
        results = []
        for text in texts:
            results.append(self.translate(text))
        return results

    def _translate_batch_parallel(self, texts: List[str]) -> List[str]:
        """병렬 배치 번역"""
        results = [None] * len(texts)

        def translate_item(idx: int, text: str) -> tuple:
            try:
                return (idx, self.translate(text))
            except Exception as e:
                print(f"[{self.name}] 번역 오류 (인덱스 {idx}): {e}")
                return (idx, text)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(translate_item, i, text): i
                for i, text in enumerate(texts)
            }

            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    results[idx] = texts[idx]
                    print(f"[{self.name}] Future 오류 (인덱스 {idx}): {e}")

        return results

    def translate_with_progress(
        self,
        texts: List[str],
        callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """진행률 콜백과 함께 번역"""
        results = []
        total = len(texts)

        for i, text in enumerate(texts):
            results.append(self.translate(text))
            if callback:
                callback(i + 1, total)

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
        # 커스텀 프롬프트 지원
        if self.custom_prompt:
            return self.custom_prompt.format(
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )

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


class ClaudeTranslator(BaseTranslator):
    """Anthropic Claude 번역기"""

    name = "claude"
    FORMULA_PLACEHOLDER = "{{{{v{id}}}}}"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.api_key = kwargs.get("api_key") or config.get("ANTHROPIC_API_KEY", "")
        self.model = kwargs.get("model", "claude-sonnet-4-20250514")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def _get_prompt(self) -> str:
        # 커스텀 프롬프트 지원
        if self.custom_prompt:
            return self.custom_prompt.format(
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )

        return f"""You are a professional translator for technical documents.

Rules:
1. Translate from {self.source_lang} to {self.target_lang}
2. Keep formula placeholders like {{{{vN}}}} unchanged
3. Preserve numbers, units, and technical terms
4. Return ONLY the translation"""

    def _translate(self, text: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._get_prompt(),
                messages=[{"role": "user", "content": text}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            print(f"[Claude] 번역 오류: {e}")
            return text

    def translate_batch(self, texts: List[str]) -> List[str]:
        """배치 번역"""
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
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._get_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            result_text = message.content[0].text.strip()
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
            print(f"[Claude] 배치 번역 오류: {e}")
            # 개별 번역으로 폴백
            for orig_idx, orig_text in to_translate:
                try:
                    translated = self._translate(orig_text)
                    results[orig_idx] = translated
                    if self.use_cache:
                        cache.set(
                            orig_text, self.source_lang, self.target_lang,
                            self.name, translated
                        )
                except Exception:
                    pass

        return results


class XinferenceTranslator(BaseTranslator):
    """Xinference 로컬 LLM 번역기"""

    name = "xinference"
    FORMULA_PLACEHOLDER = "{{{{v{id}}}}}"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.model = kwargs.get("model", "qwen2-instruct")
        self.host = kwargs.get("host") or config.get("XINFERENCE_HOST", "http://localhost:9997")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="not-needed",
                base_url=f"{self.host}/v1",
            )
        return self._client

    def _get_prompt(self) -> str:
        # 커스텀 프롬프트 지원
        if self.custom_prompt:
            return self.custom_prompt.format(
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )

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
            print(f"[Xinference] 번역 오류: {e}")
            return text


class AzureTranslator(BaseTranslator):
    """Azure Translator API"""

    name = "azure"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.api_key = kwargs.get("api_key") or config.get("AZURE_TRANSLATOR_KEY", "")
        self.region = kwargs.get("region") or config.get("AZURE_TRANSLATOR_REGION", "global")
        self.endpoint = kwargs.get("endpoint") or config.get(
            "AZURE_TRANSLATOR_ENDPOINT",
            "https://api.cognitive.microsofttranslator.com"
        )

    def _translate(self, text: str) -> str:
        try:
            import requests
            import uuid

            path = '/translate'
            url = self.endpoint + path

            params = {
                'api-version': '3.0',
                'from': self._get_azure_lang(self.source_lang),
                'to': self._get_azure_lang(self.target_lang)
            }

            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.region,
                'Content-type': 'application/json',
                'X-ClientTraceId': str(uuid.uuid4())
            }

            body = [{'text': text}]
            response = requests.post(url, params=params, headers=headers, json=body)
            response.raise_for_status()

            result = response.json()
            return result[0]['translations'][0]['text']
        except Exception as e:
            print(f"[Azure] 번역 오류: {e}")
            return text

    def _get_azure_lang(self, lang: str) -> str:
        """언어 코드 변환"""
        lang_map = {
            "korean": "ko", "ko": "ko",
            "english": "en", "en": "en", "eng": "en",
            "japanese": "ja", "ja": "ja", "jp": "ja",
            "chinese": "zh-Hans", "zh": "zh-Hans", "zh-cn": "zh-Hans",
            "chinese-traditional": "zh-Hant", "zh-tw": "zh-Hant",
            "spanish": "es", "es": "es",
            "french": "fr", "fr": "fr",
            "german": "de", "de": "de",
            "russian": "ru", "ru": "ru",
            "portuguese": "pt", "pt": "pt",
            "italian": "it", "it": "it",
            "vietnamese": "vi", "vi": "vi",
            "thai": "th", "th": "th",
            "arabic": "ar", "ar": "ar",
        }
        return lang_map.get(lang.lower(), lang)


class PapagoTranslator(BaseTranslator):
    """Naver Papago 번역기"""

    name = "papago"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.client_id = kwargs.get("client_id") or config.get("PAPAGO_CLIENT_ID", "")
        self.client_secret = kwargs.get("client_secret") or config.get("PAPAGO_CLIENT_SECRET", "")

    def _translate(self, text: str) -> str:
        try:
            import requests

            url = "https://openapi.naver.com/v1/papago/n2mt"
            headers = {
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
            data = {
                "source": self._get_papago_lang(self.source_lang),
                "target": self._get_papago_lang(self.target_lang),
                "text": text
            }

            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()

            result = response.json()
            return result['message']['result']['translatedText']
        except Exception as e:
            print(f"[Papago] 번역 오류: {e}")
            return text

    def _get_papago_lang(self, lang: str) -> str:
        """언어 코드 변환"""
        lang_map = {
            "korean": "ko", "ko": "ko",
            "english": "en", "en": "en", "eng": "en",
            "japanese": "ja", "ja": "ja", "jp": "ja",
            "chinese": "zh-CN", "zh": "zh-CN", "zh-cn": "zh-CN",
            "chinese-traditional": "zh-TW", "zh-tw": "zh-TW",
            "spanish": "es", "es": "es",
            "french": "fr", "fr": "fr",
            "german": "de", "de": "de",
            "russian": "ru", "ru": "ru",
            "portuguese": "pt", "pt": "pt",
            "italian": "it", "it": "it",
            "vietnamese": "vi", "vi": "vi",
            "thai": "th", "th": "th",
        }
        return lang_map.get(lang.lower(), lang)


class GeminiTranslator(BaseTranslator):
    """Google Gemini 번역기"""

    name = "gemini"
    FORMULA_PLACEHOLDER = "{{{{v{id}}}}}"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.api_key = kwargs.get("api_key") or config.get("GOOGLE_API_KEY", "")
        self.model = kwargs.get("model", "gemini-1.5-flash")

    def _get_prompt(self) -> str:
        # 커스텀 프롬프트 지원
        if self.custom_prompt:
            return self.custom_prompt.format(
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )

        return f"""You are a professional translator for technical documents.

Rules:
1. Translate from {self.source_lang} to {self.target_lang}
2. Keep formula placeholders like {{{{vN}}}} unchanged
3. Preserve numbers, units, and technical terms
4. Return ONLY the translation"""

    def _translate(self, text: str) -> str:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            prompt = f"{self._get_prompt()}\n\nTranslate this text:\n{text}"
            response = model.generate_content(prompt)

            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] 번역 오류: {e}")
            return text


class LLMTranslator(BaseTranslator):
    """범용 OpenAI-호환 API 번역기 (vLLM, LM Studio, etc.)"""

    name = "llm"
    FORMULA_PLACEHOLDER = "{{{{v{id}}}}}"

    def __init__(self, source_lang: str, target_lang: str, **kwargs):
        super().__init__(source_lang, target_lang, **kwargs)
        self.model = kwargs.get("model", "")
        self.api_key = kwargs.get("api_key") or config.get("LLM_API_KEY", "not-needed")
        self.base_url = kwargs.get("base_url") or config.get("LLM_BASE_URL", "http://localhost:8000/v1")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def _get_prompt(self) -> str:
        # 커스텀 프롬프트 지원
        if self.custom_prompt:
            return self.custom_prompt.format(
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )

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
            print(f"[LLM] 번역 오류: {e}")
            return text


# 번역기 팩토리
TRANSLATOR_CLASSES = {
    "openai": OpenAITranslator,
    "google": GoogleTranslator,
    "deepl": DeepLTranslator,
    "ollama": OllamaTranslator,
    "claude": ClaudeTranslator,
    "xinference": XinferenceTranslator,
    "azure": AzureTranslator,
    "papago": PapagoTranslator,
    "gemini": GeminiTranslator,
    "llm": LLMTranslator,
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

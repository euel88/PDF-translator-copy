"""
번역 모듈 (OpenAI)
"""
from typing import List, Optional
import re


class Translator:
    """OpenAI 기반 번역기"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        source_lang: str = "English",
        target_lang: str = "Korean",
    ):
        self.api_key = api_key
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._client = None

    @property
    def client(self):
        """OpenAI 클라이언트 (지연 초기화)"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def translate(self, text: str) -> str:
        """단일 텍스트 번역"""
        if not text or not text.strip():
            return text

        prompt = f"""Translate the following text from {self.source_lang} to {self.target_lang}.
Return ONLY the translation, nothing else.

Text: {text}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate accurately and naturally."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def translate_batch(self, texts: List[str]) -> List[str]:
        """여러 텍스트 일괄 번역"""
        if not texts:
            return []

        # 빈 텍스트 필터링
        valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid:
            return texts.copy()

        # 번호 붙여서 하나로 합침
        numbered = "\n".join(f"[{i+1}] {t}" for i, (_, t) in enumerate(valid))

        prompt = f"""Translate each numbered line from {self.source_lang} to {self.target_lang}.
Keep the numbering format [N] at the start of each line.
Return ONLY the translations, nothing else.

{numbered}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        # 결과 파싱
        result_text = response.choices[0].message.content.strip()
        translations = []

        for line in result_text.split("\n"):
            line = line.strip()
            if line:
                # [숫자] 제거
                cleaned = re.sub(r"^\[\d+\]\s*", "", line)
                translations.append(cleaned)

        # 원본 리스트에 매핑
        result = texts.copy()
        for idx, (orig_idx, _) in enumerate(valid):
            if idx < len(translations):
                result[orig_idx] = translations[idx]

        return result

    def set_languages(self, source: str, target: str):
        """언어 설정 변경"""
        self.source_lang = source
        self.target_lang = target

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False

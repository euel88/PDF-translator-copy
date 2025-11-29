"""
OpenAI 기반 번역 모듈
"""
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """번역 결과"""
    original: str
    translated: str
    source_lang: str
    target_lang: str


class OpenAITranslator:
    """OpenAI API 기반 번역기"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        source_lang: str = "English",
        target_lang: str = "Korean"
    ):
        """
        초기화

        Args:
            api_key: OpenAI API 키
            model: 사용할 모델 (gpt-4o-mini, gpt-4o 등)
            source_lang: 원본 언어
            target_lang: 번역 대상 언어
        """
        self.api_key = api_key
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._client = None

    def _get_client(self):
        """OpenAI 클라이언트 가져오기 (지연 초기화)"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def translate(self, text: str) -> str:
        """
        텍스트 번역

        Args:
            text: 번역할 텍스트

        Returns:
            번역된 텍스트
        """
        if not text or not text.strip():
            return text

        client = self._get_client()

        system_prompt = f"""You are a professional translator.
Translate the following text from {self.source_lang} to {self.target_lang}.
Only return the translated text, nothing else.
Maintain the original formatting and line breaks where appropriate.
If the text contains technical terms, translate them accurately."""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,  # 일관된 번역을 위해 낮은 temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"번역 실패: {e}")

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        여러 텍스트 일괄 번역

        Args:
            texts: 번역할 텍스트 목록

        Returns:
            번역된 텍스트 목록
        """
        if not texts:
            return []

        # 빈 텍스트 필터링하고 인덱스 기록
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not non_empty:
            return texts

        client = self._get_client()

        # 일괄 번역을 위한 프롬프트
        system_prompt = f"""You are a professional translator.
Translate each line from {self.source_lang} to {self.target_lang}.
Return ONLY the translations, one per line, in the same order.
Maintain the exact number of lines as input.
Do not add numbering or any other formatting."""

        # 텍스트들을 번호와 함께 조합
        numbered_texts = "\n".join([
            f"[{i+1}] {t}" for i, (_, t) in enumerate(non_empty)
        ])

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate these {len(non_empty)} texts:\n\n{numbered_texts}"}
                ],
                temperature=0.3,
            )

            # 응답 파싱
            translated_text = response.choices[0].message.content.strip()
            translated_lines = []

            # [숫자] 패턴 제거하고 번역 텍스트만 추출
            import re
            for line in translated_text.split("\n"):
                line = line.strip()
                if line:
                    # [숫자] 패턴 제거
                    cleaned = re.sub(r'^\[\d+\]\s*', '', line)
                    translated_lines.append(cleaned)

            # 결과 매핑
            result = texts.copy()
            for idx, (original_idx, _) in enumerate(non_empty):
                if idx < len(translated_lines):
                    result[original_idx] = translated_lines[idx]

            return result

        except Exception as e:
            raise RuntimeError(f"일괄 번역 실패: {e}")

    def translate_with_context(
        self,
        text: str,
        context: Optional[str] = None
    ) -> str:
        """
        컨텍스트를 고려한 번역

        Args:
            text: 번역할 텍스트
            context: 추가 컨텍스트 (문서 제목, 주제 등)

        Returns:
            번역된 텍스트
        """
        if not text or not text.strip():
            return text

        client = self._get_client()

        system_prompt = f"""You are a professional translator.
Translate the following text from {self.source_lang} to {self.target_lang}.
Only return the translated text, nothing else."""

        if context:
            system_prompt += f"\n\nContext: {context}"

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"번역 실패: {e}")

    def set_languages(self, source: str, target: str):
        """언어 설정 변경"""
        self.source_lang = source
        self.target_lang = target

    def set_model(self, model: str):
        """모델 변경"""
        self.model = model

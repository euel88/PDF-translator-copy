"""
번역 모듈 (OpenAI) - 개선된 버전
수식 보존, 캐싱, 향상된 프롬프트 포함
"""
import re
import hashlib
from typing import List, Optional, Dict, Tuple


class TranslationCache:
    """번역 캐시 - 중복 번역 방지"""

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def _make_key(self, text: str, src: str, tgt: str, model: str) -> str:
        """캐시 키 생성"""
        content = f"{text}|{src}|{tgt}|{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, src: str, tgt: str, model: str) -> Optional[str]:
        """캐시에서 번역 조회"""
        key = self._make_key(text, src, tgt, model)
        return self._cache.get(key)

    def set(self, text: str, src: str, tgt: str, model: str, translation: str):
        """캐시에 번역 저장"""
        key = self._make_key(text, src, tgt, model)
        self._cache[key] = translation

    def clear(self):
        """캐시 클리어"""
        self._cache.clear()


class FormulaHandler:
    """수식 감지 및 플레이스홀더 처리"""

    # 수식 패턴들
    FORMULA_PATTERNS = [
        # LaTeX 인라인 수식
        r'\$[^$]+\$',
        # LaTeX 블록 수식
        r'\$\$[^$]+\$\$',
        # 괄호로 둘러싼 수식 표현
        r'\\[\(\[][^\\]*\\[\)\]]',
        # 일반적인 수학 표현 (숫자, 연산자, 변수)
        r'[A-Za-z]?\s*[=≈≠<>≤≥±×÷]\s*[\d\.\-\+]+',
        # 분수 표현
        r'\d+\s*/\s*\d+',
        # 지수/첨자 표현
        r'[A-Za-z]\d+',
        r'[A-Za-z]_\{?[^}]+\}?',
        r'[A-Za-z]\^[\d\{\}]+',
        # 그리스 문자 (유니코드)
        r'[α-ωΑ-Ω]+',
        # 수학 기호
        r'[∑∏∫∂∇√∞∝∀∃∈∉⊂⊃⊆⊇∪∩]+',
        # 괄호 수식
        r'\([^)]*[+\-*/=][^)]*\)',
    ]

    def __init__(self):
        self._formulas: List[str] = []
        self._placeholder_prefix = "{{FORMULA_"
        self._placeholder_suffix = "}}"

    def extract_formulas(self, text: str) -> Tuple[str, List[str]]:
        """
        텍스트에서 수식을 추출하고 플레이스홀더로 교체

        Returns:
            (플레이스홀더가 적용된 텍스트, 추출된 수식 리스트)
        """
        self._formulas = []
        result = text

        for pattern in self.FORMULA_PATTERNS:
            matches = list(re.finditer(pattern, result))
            # 역순으로 처리 (인덱스 유지)
            for match in reversed(matches):
                formula = match.group()
                if len(formula) > 1:  # 단일 문자 제외
                    idx = len(self._formulas)
                    placeholder = f"{self._placeholder_prefix}{idx}{self._placeholder_suffix}"
                    self._formulas.append(formula)
                    result = result[:match.start()] + placeholder + result[match.end():]

        return result, self._formulas

    def restore_formulas(self, text: str, formulas: List[str]) -> str:
        """플레이스홀더를 원본 수식으로 복원"""
        result = text
        for idx, formula in enumerate(formulas):
            placeholder = f"{self._placeholder_prefix}{idx}{self._placeholder_suffix}"
            result = result.replace(placeholder, formula)
        return result

    @staticmethod
    def is_formula_only(text: str) -> bool:
        """텍스트가 수식으로만 구성되어 있는지 확인"""
        # 숫자, 수학 기호, 공백으로만 구성
        cleaned = re.sub(r'[\d\s\.\,\+\-\*/\=\(\)\[\]\{\}<>≈≠≤≥±×÷α-ωΑ-Ω∑∏∫∂∇√∞∝∀∃∈∉⊂⊃⊆⊇∪∩]', '', text)
        return len(cleaned) < len(text) * 0.3  # 70% 이상이 수식 문자


class Translator:
    """OpenAI 기반 번역기 (개선된 버전)"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        source_lang: str = "English",
        target_lang: str = "Korean",
        use_cache: bool = True,
        preserve_formulas: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.use_cache = use_cache
        self.preserve_formulas = preserve_formulas

        self._client = None
        self._cache = TranslationCache() if use_cache else None
        self._formula_handler = FormulaHandler() if preserve_formulas else None

    @property
    def client(self):
        """OpenAI 클라이언트 (지연 초기화)"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        prompt = f"""You are a professional translator specializing in technical and scientific documents.

Rules:
1. Translate from {self.source_lang} to {self.target_lang}
2. Preserve all mathematical formulas and expressions exactly as-is
3. Keep placeholder markers like {{{{FORMULA_N}}}} unchanged
4. Maintain numbers, units, and technical terms accurately
5. Preserve proper nouns and abbreviations
6. Return ONLY the translation, no explanations
7. If text is purely numerical or formulaic, return it unchanged"""
        return prompt

    def translate(self, text: str) -> str:
        """단일 텍스트 번역"""
        if not text or not text.strip():
            return text

        # 수식만으로 구성된 경우 번역 스킵
        if self._formula_handler and FormulaHandler.is_formula_only(text):
            return text

        # 캐시 확인
        if self._cache:
            cached = self._cache.get(text, self.source_lang, self.target_lang, self.model)
            if cached:
                return cached

        # 수식 추출 및 플레이스홀더 적용
        formulas = []
        text_to_translate = text
        if self._formula_handler:
            text_to_translate, formulas = self._formula_handler.extract_formulas(text)

        # 번역 요청
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": text_to_translate},
            ],
            temperature=0,  # 일관성을 위해 0 사용
        )

        result = response.choices[0].message.content.strip()

        # 수식 복원
        if self._formula_handler and formulas:
            result = self._formula_handler.restore_formulas(result, formulas)

        # 캐시 저장
        if self._cache:
            self._cache.set(text, self.source_lang, self.target_lang, self.model, result)

        return result

    def translate_batch(self, texts: List[str]) -> List[str]:
        """여러 텍스트 일괄 번역 (개선된 버전)"""
        if not texts:
            return []

        results = texts.copy()
        to_translate = []  # (original_index, text, formulas)

        # 1단계: 전처리 - 캐시 확인, 수식 처리
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            # 수식만으로 구성된 경우 스킵
            if self._formula_handler and FormulaHandler.is_formula_only(text):
                continue

            # 캐시 확인
            if self._cache:
                cached = self._cache.get(text, self.source_lang, self.target_lang, self.model)
                if cached:
                    results[i] = cached
                    continue

            # 수식 추출
            formulas = []
            text_to_translate = text
            if self._formula_handler:
                text_to_translate, formulas = self._formula_handler.extract_formulas(text)

            to_translate.append((i, text_to_translate, text, formulas))

        # 번역할 게 없으면 반환
        if not to_translate:
            return results

        # 2단계: 배치 번역
        numbered = "\n".join(
            f"[{idx+1}] {item[1]}" for idx, item in enumerate(to_translate)
        )

        batch_prompt = f"""Translate each numbered line below.
Keep the [N] numbering format.
Preserve all {{{{FORMULA_N}}}} placeholders exactly.
Return only translations.

{numbered}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": batch_prompt},
            ],
            temperature=0,
        )

        # 3단계: 결과 파싱
        result_text = response.choices[0].message.content.strip()
        translations = []

        for line in result_text.split("\n"):
            line = line.strip()
            if line:
                cleaned = re.sub(r"^\[\d+\]\s*", "", line)
                translations.append(cleaned)

        # 4단계: 결과 매핑 및 수식 복원
        for idx, (orig_idx, _, orig_text, formulas) in enumerate(to_translate):
            if idx < len(translations):
                translation = translations[idx]

                # 수식 복원
                if self._formula_handler and formulas:
                    translation = self._formula_handler.restore_formulas(translation, formulas)

                results[orig_idx] = translation

                # 캐시 저장
                if self._cache:
                    self._cache.set(
                        orig_text, self.source_lang, self.target_lang,
                        self.model, translation
                    )

        return results

    def set_languages(self, source: str, target: str):
        """언어 설정 변경"""
        self.source_lang = source
        self.target_lang = target
        if self._cache:
            self._cache.clear()

    def clear_cache(self):
        """캐시 클리어"""
        if self._cache:
            self._cache.clear()

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

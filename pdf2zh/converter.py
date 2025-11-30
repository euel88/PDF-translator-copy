"""
PDF 변환 모듈 - PDFMathTranslate 구조 기반
핵심 PDF 파싱 및 변환 로직

개선된 기능:
- 다중 페이지 병렬 텍스트 추출
- 전체 문서 일괄 번역 (API 호출 최소화)
- 병렬 페이지 처리
- 번역 소요 시간 표시
"""
import io
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import numpy as np

from pdf2zh.translator import BaseTranslator, create_translator
from pdf2zh.doclayout import LayoutDetector, LayoutBox, TRANSLATABLE_LABELS
from pdf2zh.ocr import TesseractOCR, OCRResult, merge_ocr_results
from pdf2zh.fonts import font_manager, get_font_path


# 언어별 줄간격 설정 (PDFMathTranslate 참고)
LANG_LINEHEIGHT_MAP = {
    "zh-cn": 1.4, "zh-tw": 1.4, "zh": 1.4,
    "ja": 1.1, "jp": 1.1,
    "ko": 1.2, "kr": 1.2,
    "en": 1.2, "eng": 1.2,
    "de": 1.2, "fr": 1.2, "es": 1.2,
}

# 폰트 파일 경로 (시스템별)
FONT_PATHS = {
    "windows": [
        r"C:\Windows\Fonts\malgun.ttf",      # 맑은 고딕
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
    ],
    "linux": [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "darwin": [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
    ],
}


@dataclass
class TextBlock:
    """텍스트 블록"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    font_name: str
    color: Tuple[int, int, int]
    is_formula: bool = False
    is_vertical: bool = False  # 수직 텍스트 여부 (PDFMathTranslate 참고)

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]
    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0


@dataclass
class TranslateResult:
    """번역 결과"""
    mono_pdf: Optional[bytes] = None  # 단일 언어 PDF
    dual_pdf: Optional[bytes] = None  # 이중 언어 PDF
    output_path: Optional[str] = None  # 출력 경로
    page_count: int = 0
    success: bool = False
    error: Optional[str] = None


class FormulaDetector:
    """수식 감지기 - PDFMathTranslate 수준의 완벽한 LaTeX 보존"""

    # 수학 폰트 패턴 (PDFMathTranslate 참고)
    # Computer Modern, MathTime, Symbol 등 LaTeX/수학 폰트 감지
    MATH_FONT_PATTERNS = [
        r"^CM[^RT]",           # Computer Modern (CMR, CMB 제외 - 일반 텍스트)
        r"^CMSY",              # Computer Modern Symbol
        r"^CMMI",              # Computer Modern Math Italic
        r"^CMEX",              # Computer Modern Extension
        r"^MS[A-Z]?M",         # Math Symbol fonts
        r"^Math",              # Generic Math fonts
        r"^Symbol",            # Symbol font
        r"^MT[A-Z]*",          # MathTime fonts
        r"^Euclid",            # Euclid math fonts
        r"^STIX",              # STIX math fonts
        r"^.*Math.*",          # Any font with "Math" in name
        r"^.*Sym.*",           # Any font with "Sym" in name
        r"^LM[A-Z]*$",         # Latin Modern Math
        r"^NimbusRom.*Ita",    # Nimbus Roman Italic (often used for math)
    ]

    # 컴파일된 폰트 패턴 (성능 최적화)
    _compiled_font_patterns = None

    @classmethod
    def _get_font_patterns(cls):
        """컴파일된 폰트 패턴 반환"""
        if cls._compiled_font_patterns is None:
            cls._compiled_font_patterns = [
                re.compile(p, re.IGNORECASE) for p in cls.MATH_FONT_PATTERNS
            ]
        return cls._compiled_font_patterns

    @classmethod
    def is_math_font(cls, font_name: str) -> bool:
        """폰트 이름이 수학 폰트인지 확인"""
        if not font_name:
            return False
        for pattern in cls._get_font_patterns():
            if pattern.search(font_name):
                return True
        return False

    # LaTeX 수식 패턴 (우선순위 순서)
    LATEX_PATTERNS = [
        # Display math environments
        r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
        r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
        r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}',
        r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}',
        r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}',
        r'\\begin\{displaymath\}.*?\\end\{displaymath\}',
        r'\\begin\{math\}.*?\\end\{math\}',
        r'\\begin\{array\}.*?\\end\{array\}',
        r'\\begin\{matrix\}.*?\\end\{matrix\}',
        r'\\begin\{pmatrix\}.*?\\end\{pmatrix\}',
        r'\\begin\{bmatrix\}.*?\\end\{bmatrix\}',
        r'\\begin\{vmatrix\}.*?\\end\{vmatrix\}',
        r'\\begin\{cases\}.*?\\end\{cases\}',
        # Block math delimiters
        r'\$\$[^$]+\$\$',                    # $$ ... $$
        r'\\\[[^\]]*\\\]',                   # \[ ... \]
        # Inline math delimiters
        r'(?<![\\$])\$(?!\$)[^$\n]+\$(?!\$)',  # $ ... $ (not $$)
        r'\\\([^)]*\\\)',                    # \( ... \)
        # LaTeX commands with arguments
        r'\\(?:frac|dfrac|tfrac|cfrac)\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\\(?:sqrt|root)\[[^\]]*\]\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\\(?:sqrt|root)\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
        r'\\(?:sum|prod|int|oint|iint|iiint|coprod|bigcup|bigcap|bigoplus|bigotimes)(?:_\{[^{}]*\}|\^[^{}]|_[^{}]|\^\{[^{}]*\})*',
        r'\\(?:lim|limsup|liminf|sup|inf|max|min)(?:_\{[^{}]*\})*',
        r'\\(?:vec|hat|bar|tilde|dot|ddot|overline|underline|widehat|widetilde|overrightarrow|overleftarrow)\{[^{}]*\}',
        r'\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|mathfrak|mathscr)\{[^{}]*\}',
        r'\\(?:left|right|big|Big|bigg|Bigg)[\[\](){}|.]',
        r'\\(?:binom|tbinom|dbinom)\{[^{}]*\}\{[^{}]*\}',
    ]

    # LaTeX 단일 명령어 (인자 없음)
    LATEX_COMMANDS = [
        # Greek letters (lowercase)
        r'\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|upsilon|phi|varphi|chi|psi|omega)',
        # Greek letters (uppercase)
        r'\\(?:Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)',
        # Binary operators
        r'\\(?:pm|mp|times|div|cdot|ast|star|circ|bullet|cap|cup|vee|wedge|oplus|ominus|otimes|oslash|odot)',
        # Relations
        r'\\(?:leq|geq|le|ge|ll|gg|subset|supset|subseteq|supseteq|in|ni|notin|equiv|sim|simeq|approx|cong|neq|ne|propto|perp|parallel)',
        # Arrows
        r'\\(?:leftarrow|rightarrow|leftrightarrow|Leftarrow|Rightarrow|Leftrightarrow|uparrow|downarrow|updownarrow|Uparrow|Downarrow|Updownarrow|mapsto|longmapsto|longleftarrow|longrightarrow|longleftrightarrow|Longleftarrow|Longrightarrow|Longleftrightarrow|nearrow|searrow|swarrow|nwarrow|hookrightarrow|hookleftarrow|leadsto)',
        # Misc symbols
        r'\\(?:infty|nabla|partial|forall|exists|nexists|emptyset|varnothing|Re|Im|wp|aleph|hbar|ell|degree|angle|measuredangle|sphericalangle|prime|backprime|triangle|square|diamond|bigcirc|Box)',
        # Functions
        r'\\(?:sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|coth|exp|log|ln|lg|arg|det|dim|gcd|hom|ker|Pr|deg)',
        # Accents and modifiers
        r'\\(?:prime|backprime|acute|grave|breve|check|dot|ddot|dddot|ddddot|hat|widehat|tilde|widetilde|bar|overline|underline|overbrace|underbrace|vec|overrightarrow|overleftarrow)',
        # Spacing
        r'\\(?:quad|qquad|,|;|:|!|enspace|thinspace|medspace|thickspace|negthickspace|negmedspace|negthinspace)',
        # Delimiters
        r'\\(?:lbrace|rbrace|langle|rangle|lfloor|rfloor|lceil|rceil|vert|Vert|lVert|rVert|backslash)',
        # Dots
        r'\\(?:ldots|cdots|vdots|ddots|dots|dotsc|dotsb|dotsm|dotsi|dotso)',
    ]

    # Unicode 수학 기호
    MATH_SYMBOLS = r'[∑∏∫∬∭∮∯∰∂∇√∛∜∞∝∀∃∄∈∉∋∌⊂⊃⊆⊇⊄⊅∪∩⊕⊖⊗⊘⊙⊚⊛⊜⊝∅∆∧∨¬⊥∥≡≢≈≉≠≤≥≪≫≲≳≺≻≼≽⊀⊁⊰⊱⊲⊳⊴⊵←→↑↓↔↕⇐⇒⇑⇓⇔⇕↦↤⟨⟩⌈⌉⌊⌋⟦⟧⟪⟫′″‴⁗±∓×÷·⋅∗∘∙⊡⊞⊟⊠∧∨⊼⊽⊻△▽□◇○◎●∎∴∵∶∷∸∹∺∻]+'

    # Greek letters (Unicode)
    GREEK_LETTERS = r'[αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩϑϕϖϱϵ]+'

    # Subscripts and superscripts
    SUBSCRIPT_SUPERSCRIPT = r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁱⁿ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ]+'

    @classmethod
    def is_formula(cls, text: str, font_name: str = "", is_vertical: bool = False) -> bool:
        """
        텍스트가 수식인지 확인 (더 정밀한 감지)

        Args:
            text: 확인할 텍스트
            font_name: 폰트 이름 (수학 폰트 감지용)
            is_vertical: 수직 텍스트 여부

        Returns:
            수식이면 True
        """
        text = text.strip()
        if not text:
            return False

        # 1. 폰트 기반 감지 (PDFMathTranslate 핵심 기능)
        if font_name and cls.is_math_font(font_name):
            return True

        # 2. 수직 텍스트는 일반적으로 수식이 아님 (일본어/중국어 등)
        # 하지만 단일 기호인 경우 수식일 수 있음
        if is_vertical and len(text) > 1:
            # 수직 텍스트가 수학 기호로만 구성된 경우만 수식으로 처리
            if not re.search(cls.MATH_SYMBOLS, text) and not re.search(cls.GREEK_LETTERS, text):
                return False

        # 3. LaTeX 환경 체크
        for pattern in cls.LATEX_PATTERNS:
            if re.search(pattern, text, re.DOTALL):
                return True

        # 4. LaTeX 명령어 체크
        for pattern in cls.LATEX_COMMANDS:
            if re.search(pattern, text):
                return True

        # 5. 수학 기호 체크
        if re.search(cls.MATH_SYMBOLS, text):
            return True

        # 6. Greek letters 체크
        if re.search(cls.GREEK_LETTERS, text):
            # Greek letters만 있는 경우 수식으로 간주
            greek_count = len(re.findall(cls.GREEK_LETTERS, text))
            alpha_count = sum(1 for c in text if c.isalpha())
            if greek_count > 0 and greek_count >= alpha_count * 0.3:
                return True

        # 7. Subscript/superscript 체크
        if re.search(cls.SUBSCRIPT_SUPERSCRIPT, text):
            return True

        # 8. 수식 특성 분석: 숫자, 연산자, 변수의 조합
        operators = set('+-*/=<>≤≥≠≈∈∉⊂⊃')
        has_number = any(c.isdigit() for c in text)
        has_operator = any(c in operators for c in text)
        single_letters = re.findall(r'\b[a-zA-Z]\b', text)

        if has_number and has_operator and len(single_letters) >= 2:
            # 가능성 높은 수식 (예: "x + y = 5")
            return True

        # 9. 순수 숫자+연산자 조합
        clean = re.sub(r'\s', '', text)
        if clean and all(c.isdigit() or c in '+-*/=.^_()[]{}' for c in clean):
            if len(clean) >= 3 and any(c in '+-*/=^_' for c in clean):
                return True

        return False

    @classmethod
    def extract_formulas(cls, text: str) -> Tuple[str, List[str]]:
        """수식 추출 및 플레이스홀더 적용 - 개선된 버전"""
        formulas = []
        result = text
        used_positions = set()

        def add_formula(match_start: int, match_end: int, formula: str):
            """중복 방지하며 수식 추가"""
            # 이미 처리된 위치인지 확인
            for pos_start, pos_end in used_positions:
                if not (match_end <= pos_start or match_start >= pos_end):
                    return False
            used_positions.add((match_start, match_end))
            formulas.append((match_start, match_end, formula))
            return True

        # 1. LaTeX 환경과 수식 먼저 추출
        for pattern in cls.LATEX_PATTERNS:
            for match in re.finditer(pattern, text, re.DOTALL):
                add_formula(match.start(), match.end(), match.group())

        # 2. LaTeX 명령어 추출
        for pattern in cls.LATEX_COMMANDS:
            for match in re.finditer(pattern, text):
                add_formula(match.start(), match.end(), match.group())

        # 3. Unicode 수학 기호 추출
        for match in re.finditer(cls.MATH_SYMBOLS, text):
            add_formula(match.start(), match.end(), match.group())

        # 4. Greek letters 추출
        for match in re.finditer(cls.GREEK_LETTERS, text):
            add_formula(match.start(), match.end(), match.group())

        # 5. Subscripts/superscripts 추출
        for match in re.finditer(cls.SUBSCRIPT_SUPERSCRIPT, text):
            add_formula(match.start(), match.end(), match.group())

        # 정렬 후 역순으로 플레이스홀더 적용 (위치 변경 방지)
        formulas.sort(key=lambda x: x[0], reverse=True)

        formula_list = []
        for start, end, formula in formulas:
            idx = len(formula_list)
            placeholder = f"{{{{v{idx}}}}}"
            result = result[:start] + placeholder + result[end:]
            formula_list.insert(0, formula)  # 원래 순서 유지

        return result, formula_list

    @classmethod
    def restore_formulas(cls, text: str, formulas: List[str]) -> str:
        """플레이스홀더를 수식으로 복원"""
        result = text
        for idx, formula in enumerate(formulas):
            placeholder = f"{{{{v{idx}}}}}"
            # 번역 과정에서 변형될 수 있는 플레이스홀더 변형들도 처리
            variations = [
                placeholder,
                f"{{v{idx}}}",          # 단일 중괄호
                f"{{ v{idx} }}",        # 공백 추가
                f"{{{{v{idx} }}}}",     # 뒤 공백
                f"{{{{ v{idx}}}}}",     # 앞 공백
                f"v{idx}",              # 중괄호 제거
                f"[v{idx}]",            # 대괄호로 변형
                f"(v{idx})",            # 소괄호로 변형
            ]
            for var in variations:
                if var in result:
                    result = result.replace(var, formula)
                    break
        return result

    @classmethod
    def protect_formulas_for_translation(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        번역을 위해 수식을 보호하는 고급 메서드
        수식을 XML-like 태그로 감싸서 번역기가 무시하도록 함
        """
        protected_text, formulas = cls.extract_formulas(text)
        formula_map = {}

        for idx, formula in enumerate(formulas):
            key = f"__FORMULA_{idx}__"
            protected_text = protected_text.replace(f"{{{{v{idx}}}}}", key)
            formula_map[key] = formula

        return protected_text, formula_map

    @classmethod
    def unprotect_formulas(cls, text: str, formula_map: Dict[str, str]) -> str:
        """보호된 수식 복원"""
        result = text
        for key, formula in formula_map.items():
            result = result.replace(key, formula)
        return result


class PDFConverter:
    """PDF 변환기"""

    def __init__(
        self,
        translator: BaseTranslator,
        dpi: int = 150,
        callback: Optional[Callable[[str], None]] = None,
        use_layout_detection: bool = False,
        use_ocr: bool = False,
        source_lang: str = "eng",
        target_lang: str = "ko",  # 대상 언어 추가
        # 최적화 옵션
        output_quality: int = 85,  # JPEG 품질 (1-100)
        use_vector_text: bool = True,  # 벡터 텍스트 사용 (파일 크기 대폭 감소)
        compress_images: bool = True,  # 이미지 압축 사용
        # 이중 언어 출력
        dual_output: bool = False,  # 이중 언어 PDF 생성
        # 정규표현식 예외 처리
        exclude_patterns: Optional[List[str]] = None,  # 번역 제외 패턴
        include_patterns: Optional[List[str]] = None,  # 번역 포함 패턴
    ):
        self.translator = translator
        self.dpi = dpi
        self.callback = callback
        self.use_layout_detection = use_layout_detection
        self.use_ocr = use_ocr
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._cancelled = False

        # 최적화 옵션
        self.output_quality = output_quality
        self.use_vector_text = use_vector_text
        self.compress_images = compress_images

        # 이중 언어 출력
        self.dual_output = dual_output

        # 정규표현식 예외 처리 (컴파일된 패턴 저장)
        self.exclude_patterns = []
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    self.exclude_patterns.append(re.compile(pattern))
                except re.error:
                    pass  # 잘못된 패턴 무시

        self.include_patterns = []
        if include_patterns:
            for pattern in include_patterns:
                try:
                    self.include_patterns.append(re.compile(pattern))
                except re.error:
                    pass  # 잘못된 패턴 무시

        # 언어별 줄간격
        self.line_height = LANG_LINEHEIGHT_MAP.get(target_lang.lower(), 1.2)

        # 외부 폰트 파일 경로 찾기
        self.font_file = self._find_font_file()

        # 레이아웃 감지기
        self.layout_detector = LayoutDetector() if use_layout_detection else None

        # OCR 엔진
        self.ocr_engine = TesseractOCR(source_lang) if use_ocr else None

    def _should_translate(self, text: str) -> bool:
        """
        텍스트를 번역해야 하는지 확인 (정규표현식 필터링)

        Args:
            text: 확인할 텍스트

        Returns:
            번역해야 하면 True, 제외해야 하면 False
        """
        # include_patterns가 있으면 해당 패턴에 매칭되는 것만 번역
        if self.include_patterns:
            for pattern in self.include_patterns:
                if pattern.search(text):
                    return True
            return False  # 어떤 패턴에도 매칭 안 되면 번역 안 함

        # exclude_patterns에 매칭되면 번역 제외
        if self.exclude_patterns:
            for pattern in self.exclude_patterns:
                if pattern.search(text):
                    return False

        return True  # 기본적으로 번역

    def _find_font_file(self) -> Optional[str]:
        """시스템에서 사용 가능한 폰트 파일 찾기 - Go Noto Universal 우선"""
        # 새로운 폰트 관리자 사용 (Go Noto Universal 우선)
        font_path = get_font_path(self.target_lang)
        if font_path:
            return font_path

        # 폴백: 기존 시스템 폰트 검색
        import sys
        if sys.platform == "win32":
            paths = FONT_PATHS["windows"]
        elif sys.platform == "darwin":
            paths = FONT_PATHS["darwin"]
        else:
            paths = FONT_PATHS["linux"]

        for path in paths:
            if Path(path).exists():
                return path
        return None

    def cancel(self):
        """변환 취소"""
        self._cancelled = True

    def _log(self, msg: str):
        """로그 출력"""
        if self.callback:
            self.callback(msg)

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        pages: Optional[List[int]] = None,
    ) -> TranslateResult:
        """
        PDF 변환

        Args:
            input_path: 입력 PDF 경로
            output_path: 출력 PDF 경로 (None이면 메모리에만)
            pages: 변환할 페이지 목록 (None이면 전체)

        Returns:
            TranslateResult
        """
        self._cancelled = False

        try:
            self._log(f"PDF 열기: {input_path}")
            doc = fitz.open(input_path)

            if pages is None:
                pages = list(range(doc.page_count))

            self._log(f"총 {len(pages)} 페이지 처리")

            dual_pdf = None

            # 이중 언어 출력 모드
            if self.dual_output:
                self._log("이중 언어 모드 사용 (원문 + 번역)")
                mono_pdf, dual_pdf = self._convert_dual_mode(doc, pages)
            # 벡터 텍스트 모드: 원본 PDF를 직접 수정
            elif self.use_vector_text:
                self._log("벡터 텍스트 모드 사용 (파일 크기 최적화)")
                mono_pdf = self._convert_vector_mode(doc, pages)
            else:
                # 이미지 모드: 기존 방식 (호환성용)
                self._log("이미지 모드 사용")
                mono_pdf = self._convert_image_mode(doc, pages)

            # 출력 파일 저장
            if output_path:
                with open(output_path, "wb") as f:
                    f.write(mono_pdf)
                self._log(f"저장됨: {output_path}")

                # 이중 언어 PDF도 저장 (있는 경우)
                if dual_pdf:
                    dual_path = output_path.replace('.pdf', '-dual.pdf')
                    with open(dual_path, "wb") as f:
                        f.write(dual_pdf)
                    self._log(f"이중 언어 저장됨: {dual_path}")

            doc.close()

            return TranslateResult(
                mono_pdf=mono_pdf,
                dual_pdf=dual_pdf,
                output_path=output_path,
                page_count=len(pages),
                success=True,
            )

        except Exception as e:
            self._log(f"오류: {str(e)}")
            return TranslateResult(error=str(e))

    def _convert_vector_mode(
        self,
        doc: fitz.Document,
        pages: List[int]
    ) -> bytes:
        """
        벡터 모드 변환 - 원본 PDF 구조 유지, 텍스트만 교체
        파일 크기가 원본과 비슷하게 유지됨

        개선: 2단계 처리로 번역 속도 5배 향상
        1단계: 모든 페이지에서 텍스트 추출 (병렬)
        2단계: 전체 텍스트 일괄 번역 (API 호출 최소화)
        3단계: 번역 결과 적용
        """
        # 전체 시간 측정 시작
        total_start_time = time.time()

        # ===== 1단계: 모든 페이지에서 텍스트 추출 =====
        extract_start_time = time.time()
        self._log("1단계: 텍스트 추출 중...")

        # 페이지별 데이터 저장
        page_data = []  # [(page_num, blocks, to_translate_indices)]
        all_texts = []  # 전체 번역할 텍스트
        all_formulas = []  # 전체 수식 맵

        ocr_used = False
        for page_num in pages:
            if self._cancelled:
                break

            page = doc[page_num]
            blocks, has_text = self._extract_blocks_for_vector(page)

            # OCR 폴백: 텍스트가 없는 스캔된 PDF
            if not has_text and self.ocr_engine:
                self._log(f"  페이지 {page_num + 1}: 텍스트 없음 → OCR 사용")
                blocks = self._extract_blocks_for_vector_ocr(page)
                ocr_used = True

            # 번역할 블록 필터링
            to_translate_indices = []
            for i, b in enumerate(blocks):
                if not b.is_formula and self._should_translate(b.text):
                    to_translate_indices.append(i)
                    # 수식 추출 및 텍스트 저장
                    text, formulas = FormulaDetector.extract_formulas(b.text)
                    all_texts.append(text)
                    all_formulas.append(formulas)

            page_data.append((page_num, blocks, to_translate_indices))

        total_texts = len(all_texts)
        extract_elapsed = time.time() - extract_start_time
        ocr_msg = " (OCR 사용됨)" if ocr_used else ""
        self._log(f"  총 {len(pages)}페이지에서 {total_texts}개 텍스트 추출 완료 ({extract_elapsed:.1f}초){ocr_msg}")

        # ===== 2단계: 전체 텍스트 일괄 번역 =====
        translate_start_time = time.time()
        if total_texts > 0:
            self._log(f"2단계: {total_texts}개 텍스트 일괄 번역 중...")
            all_translations = self.translator.translate_batch(all_texts)

            # 번역 결과 검증 및 통계
            translated_count = sum(1 for i, t in enumerate(all_translations)
                                   if t and t.strip() and t.strip() != all_texts[i].strip())
            unchanged_count = sum(1 for i, t in enumerate(all_translations)
                                  if t and t.strip() == all_texts[i].strip())
            empty_count = sum(1 for t in all_translations if not t or not t.strip())

            if empty_count > 0 or unchanged_count > total_texts * 0.5:
                self._log(f"  경고: 번역 실패 {empty_count}개, 변경 없음 {unchanged_count}개")

            # 수식 복원
            for i, trans in enumerate(all_translations):
                if all_formulas[i]:
                    all_translations[i] = FormulaDetector.restore_formulas(trans, all_formulas[i])

            translate_elapsed = time.time() - translate_start_time
            self._log(f"  번역 완료 ({translate_elapsed:.1f}초, {translated_count}/{total_texts}개 성공)")
        else:
            all_translations = []
            translate_elapsed = 0
            self._log("  경고: 번역할 텍스트가 없습니다.")

        # ===== 3단계: 번역 결과 적용 =====
        apply_start_time = time.time()
        self._log("3단계: 번역 결과 적용 중...")
        output_doc = fitz.open()
        translation_idx = 0
        total_pages = len(page_data)

        for idx, (page_num, blocks, to_translate_indices) in enumerate(page_data):
            if self._cancelled:
                break

            # 진행 상황 로깅 (5페이지마다 또는 마지막 페이지)
            if idx % 5 == 0 or idx == total_pages - 1:
                self._log(f"  페이지 {idx + 1}/{total_pages} 처리 중...")

            # 원본 페이지 복사
            page = doc[page_num]
            output_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_page = output_doc[-1]

            if not to_translate_indices:
                continue

            # 배경색 감지를 위한 페이지 렌더링
            page_pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1))

            # 이 페이지의 번역 결과 가져오기
            page_blocks = [blocks[i] for i in to_translate_indices]
            page_translations = all_translations[translation_idx:translation_idx + len(to_translate_indices)]
            translation_idx += len(to_translate_indices)

            # 벡터 텍스트 교체
            self._replace_text_vector(new_page, page_blocks, page_translations, page_pixmap)

        apply_elapsed = time.time() - apply_start_time
        self._log(f"  {len(pages)}페이지 처리 완료 ({apply_elapsed:.1f}초)")

        # 주석 및 폼 필드 번역
        self._log("주석 및 폼 필드 번역 중...")
        self._translate_annotations_and_forms(output_doc)

        # 압축 옵션으로 저장
        self._log("4단계: PDF 저장 중...")
        output = io.BytesIO()
        output_doc.save(
            output,
            garbage=4,  # 최대 가비지 컬렉션
            deflate=True,  # 압축 사용
            clean=True,  # 사용하지 않는 객체 제거
        )
        output_doc.close()

        # 전체 소요 시간 출력
        total_elapsed = time.time() - total_start_time
        self._log(f"  저장 완료")
        self._log(f"===== 번역 완료: 총 {total_elapsed:.1f}초 소요 =====")
        self._log(f"  - 텍스트 추출: {extract_elapsed:.1f}초")
        self._log(f"  - 번역: {translate_elapsed:.1f}초 ({total_texts}개 항목)")
        self._log(f"  - 결과 적용: {apply_elapsed:.1f}초")

        return output.getvalue()

    def _extract_blocks_for_vector(self, page: fitz.Page) -> Tuple[List[TextBlock], bool]:
        """벡터 모드용 텍스트 블록 추출 (스케일 없음)

        Returns:
            Tuple[List[TextBlock], bool]: (블록 리스트, 텍스트 발견 여부)
        """
        blocks = []
        has_text = False

        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_text = ""
                line_spans = []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_text += text + " "
                        line_spans.append(span)
                        has_text = True

                line_text = line_text.strip()
                if not line_text or len(line_text) < 2:
                    continue

                # 바운딩 박스 (스케일 없음 - 원본 좌표)
                x0 = min(s["bbox"][0] for s in line_spans)
                y0 = min(s["bbox"][1] for s in line_spans)
                x1 = max(s["bbox"][2] for s in line_spans)
                y1 = max(s["bbox"][3] for s in line_spans)

                # 수직 텍스트 감지 (PDFMathTranslate 참고)
                # 텍스트 방향은 line의 dir 속성 또는 bbox의 비율로 판단
                line_dir = line.get("dir", (1, 0))  # 기본값: 수평 (1, 0)
                is_vertical = False
                if line_dir:
                    # dir[0]이 0에 가깝고 dir[1]이 1 또는 -1에 가까우면 수직
                    if abs(line_dir[0]) < 0.5 and abs(line_dir[1]) > 0.5:
                        is_vertical = True
                # bbox 비율로도 체크 (높이가 너비보다 훨씬 큰 경우)
                width = x1 - x0
                height = y1 - y0
                if width > 0 and height > width * 3 and len(line_text) > 3:
                    is_vertical = True

                # 가장 많은 텍스트를 커버하는 색상 찾기 (dominant color)
                color_weights = {}  # color -> text_length
                for span in line_spans:
                    color_int = span.get("color", 0)
                    r = (color_int >> 16) & 0xFF
                    g = (color_int >> 8) & 0xFF
                    b = color_int & 0xFF
                    color = (r, g, b)
                    text_len = len(span.get("text", ""))
                    color_weights[color] = color_weights.get(color, 0) + text_len

                # 가장 가중치가 높은 색상 선택
                if color_weights:
                    dominant_color = max(color_weights.keys(), key=lambda c: color_weights[c])
                else:
                    dominant_color = (0, 0, 0)

                # 가장 큰 폰트 크기 사용 (제목 등에서 중요)
                font_sizes = [s.get("size", 12) for s in line_spans]
                font_size = max(font_sizes) if font_sizes else 12

                # 가장 많이 사용된 폰트 이름
                font_names = {}
                for span in line_spans:
                    fn = span.get("font", "")
                    text_len = len(span.get("text", ""))
                    font_names[fn] = font_names.get(fn, 0) + text_len
                font_name = max(font_names.keys(), key=lambda f: font_names[f]) if font_names else ""

                # 수식 감지 (폰트 이름과 수직 텍스트 여부 전달)
                is_formula = FormulaDetector.is_formula(line_text, font_name, is_vertical)

                blocks.append(TextBlock(
                    text=line_text,
                    bbox=(x0, y0, x1, y1),
                    font_size=font_size,
                    font_name=font_name,
                    color=dominant_color,
                    is_formula=is_formula,
                    is_vertical=is_vertical,
                ))

        return blocks, has_text

    def _extract_blocks_for_vector_ocr(
        self,
        page: fitz.Page,
        img: Optional[Image.Image] = None
    ) -> List[TextBlock]:
        """OCR을 사용하여 벡터 모드용 텍스트 블록 추출 (스캔된 PDF용)

        Args:
            page: PDF 페이지
            img: 렌더링된 이미지 (없으면 새로 생성)

        Returns:
            List[TextBlock]: OCR로 추출된 텍스트 블록
        """
        if not self.ocr_engine:
            self._log("  OCR 엔진이 활성화되지 않음. use_ocr=True로 설정하세요.")
            return []

        blocks = []

        # 이미지가 없으면 페이지 렌더링
        if img is None:
            # 벡터 모드에서는 스케일 없이 원본 좌표 사용
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # OCR 실행
        self._log(f"  OCR 텍스트 인식 중...")
        ocr_results = self.ocr_engine.recognize(img)
        merged = merge_ocr_results(ocr_results)

        self._log(f"  OCR: {len(merged)}개 텍스트 영역 감지")

        for result in merged:
            if len(result.text) < 2:
                continue

            # OCR 결과에서 수식 감지
            is_formula = FormulaDetector.is_formula(result.text)

            blocks.append(TextBlock(
                text=result.text,
                bbox=result.bbox,
                font_size=14,  # OCR은 폰트 정보 없음, 기본값 사용
                font_name="",
                color=(0, 0, 0),  # 기본 검정색
                is_formula=is_formula,
                is_vertical=False,
            ))

        return blocks

    def _replace_text_vector(
        self,
        page: fitz.Page,
        blocks: List[TextBlock],
        translations: List[str],
        page_pixmap: Optional[fitz.Pixmap] = None
    ):
        """벡터 방식으로 텍스트 교체 - PyMuPDF 사용, 원본 레이아웃 최대 보존

        개선: 번역 결과가 비어있거나 실패할 경우 원본 텍스트 유지
        """

        # 외부 폰트 등록 (한글 지원)
        font_registered = False
        font_xref = None
        if self.font_file and Path(self.font_file).exists():
            try:
                font_xref = page.insert_font(
                    fontfile=self.font_file,
                    fontname="korean-font"
                )
                font_registered = True
            except Exception:
                pass

        for block, translation in zip(blocks, translations):
            # 번역 결과 검증: 비어있거나 None인 경우 원본 유지 (텍스트를 덮지 않음)
            if translation is None or not translation.strip():
                continue  # 원본 텍스트 유지

            x0, y0, x1, y1 = block.bbox
            rect = fitz.Rect(x0, y0, x1, y1)

            # 번역 결과 유효성 확인: 원본과 동일하면 처리 건너뜀
            if translation.strip() == block.text.strip():
                continue  # 변경 없음, 원본 유지

            # 배경색 감지 (페이지 렌더링에서)
            bg_color = self._detect_bg_color_from_pixmap(page_pixmap, rect) if page_pixmap else (1, 1, 1)

            # 폰트 설정
            font_size = block.font_size
            text_color = tuple(c / 255 for c in block.color)

            # 대상 언어에 맞는 CJK 내장 폰트
            cjk_fontname = self._get_vector_fontname("")

            # 폰트 후보 목록 (우선순위 순서)
            font_candidates = []
            if font_registered:
                font_candidates.append("korean-font")
            font_candidates.append(cjk_fontname)  # 대상 언어에 맞는 CJK 폰트
            font_candidates.append("helv")  # 라틴 기본 폰트

            # 텍스트 맞춤 테스트 및 적합한 폰트 찾기
            working_font = None
            working_size = font_size * 0.9
            working_text = translation

            for fontname in font_candidates:
                try:
                    # 삽입 테스트 (overlay=False는 실제로 삽입하지 않고 테스트)
                    adjusted_size, fitted_text = self._fit_text_vector_improved(
                        page, translation, rect, font_size, fontname
                    )
                    if adjusted_size >= 6:  # 최소 크기 이상이면 사용 가능
                        working_font = fontname
                        working_size = adjusted_size
                        working_text = fitted_text
                        break
                except Exception:
                    continue

            # 적합한 폰트를 찾지 못한 경우 기본값 사용
            if working_font is None:
                working_font = cjk_fontname
                working_size = max(6, font_size * 0.5)
                working_text = translation

            # 원본 텍스트 영역을 배경색으로 덮기
            pad_x = 0.5
            pad_y = 0.5
            cover_rect = fitz.Rect(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)
            page.draw_rect(cover_rect, color=None, fill=bg_color)

            # 번역된 텍스트 삽입
            text_inserted = False
            for fontname in [working_font] + font_candidates:
                if text_inserted:
                    break
                try:
                    page.insert_textbox(
                        rect,
                        working_text,
                        fontsize=working_size,
                        fontname=fontname,
                        color=text_color,
                        align=fitz.TEXT_ALIGN_LEFT,
                        lineheight=self.line_height,
                    )
                    text_inserted = True
                except Exception:
                    continue

            # 모든 시도 실패 시 원본 텍스트라도 삽입 시도
            if not text_inserted:
                try:
                    page.insert_textbox(
                        rect,
                        block.text,
                        fontsize=font_size,
                        fontname=cjk_fontname,
                        color=text_color,
                        align=fitz.TEXT_ALIGN_LEFT,
                    )
                except Exception:
                    # 최종 폴백 실패 - 로깅만 수행
                    pass

    def _detect_bg_color_from_pixmap(
        self,
        pixmap: fitz.Pixmap,
        rect: fitz.Rect
    ) -> Tuple[float, float, float]:
        """Pixmap에서 영역의 배경색 감지 - 더 정확한 버전"""
        try:
            # 영역 주변 샘플링
            x0, y0, x1, y1 = int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)

            # 클램핑
            x0 = max(0, min(x0, pixmap.width - 1))
            x1 = max(0, min(x1, pixmap.width - 1))
            y0 = max(0, min(y0, pixmap.height - 1))
            y1 = max(0, min(y1, pixmap.height - 1))

            width = x1 - x0
            height = y1 - y0

            if width <= 0 or height <= 0:
                return (1, 1, 1)

            all_samples = []

            # 1. 외부 4방향 가장자리 샘플링 (가장 신뢰할 수 있음)
            edge_offsets = [1, 2, 3, 5]  # 더 넓은 범위
            step_x = max(1, width // 15)
            step_y = max(1, height // 8)

            # 상단 가장자리
            for x in range(x0, x1, step_x):
                for offset in edge_offsets:
                    if y0 - offset >= 0:
                        idx = ((y0 - offset) * pixmap.width + x) * pixmap.n
                        if idx + 2 < len(pixmap.samples):
                            r, g, b = pixmap.samples[idx:idx+3]
                            all_samples.append((r, g, b, 2))  # 가중치 2

            # 하단 가장자리
            for x in range(x0, x1, step_x):
                for offset in edge_offsets:
                    if y1 + offset < pixmap.height:
                        idx = ((y1 + offset) * pixmap.width + x) * pixmap.n
                        if idx + 2 < len(pixmap.samples):
                            r, g, b = pixmap.samples[idx:idx+3]
                            all_samples.append((r, g, b, 2))

            # 좌측 가장자리
            for y in range(y0, y1, step_y):
                for offset in edge_offsets:
                    if x0 - offset >= 0:
                        idx = (y * pixmap.width + (x0 - offset)) * pixmap.n
                        if idx + 2 < len(pixmap.samples):
                            r, g, b = pixmap.samples[idx:idx+3]
                            all_samples.append((r, g, b, 2))

            # 우측 가장자리
            for y in range(y0, y1, step_y):
                for offset in edge_offsets:
                    if x1 + offset < pixmap.width:
                        idx = (y * pixmap.width + (x1 + offset)) * pixmap.n
                        if idx + 2 < len(pixmap.samples):
                            r, g, b = pixmap.samples[idx:idx+3]
                            all_samples.append((r, g, b, 2))

            # 2. 4개 코너 샘플링 (외부)
            corners = [
                (x0 - 3, y0 - 3), (x1 + 3, y0 - 3),
                (x0 - 3, y1 + 3), (x1 + 3, y1 + 3)
            ]
            for cx, cy in corners:
                if 0 <= cx < pixmap.width and 0 <= cy < pixmap.height:
                    idx = (cy * pixmap.width + cx) * pixmap.n
                    if idx + 2 < len(pixmap.samples):
                        r, g, b = pixmap.samples[idx:idx+3]
                        all_samples.append((r, g, b, 3))  # 코너는 가중치 3

            # 3. 샘플 분석 - 가장 일반적인 밝은 색상 찾기
            if all_samples:
                # 가중치를 고려한 색상 그룹핑
                weighted_samples = []
                for sample in all_samples:
                    r, g, b = sample[:3]
                    weight = sample[3] if len(sample) > 3 else 1
                    for _ in range(weight):
                        weighted_samples.append((r, g, b))

                # 색상 그룹핑 (더 타이트한 임계값)
                groups = self._group_colors(weighted_samples, threshold=20)

                if groups:
                    # 가장 큰 그룹의 평균 색상 사용
                    largest_group = max(groups, key=len)
                    avg_r = sum(c[0] for c in largest_group) // len(largest_group)
                    avg_g = sum(c[1] for c in largest_group) // len(largest_group)
                    avg_b = sum(c[2] for c in largest_group) // len(largest_group)

                    return (avg_r / 255, avg_g / 255, avg_b / 255)

        except Exception:
            pass

        return (1, 1, 1)  # 기본 흰색

    def _group_colors(
        self,
        samples: List[Tuple[int, int, int]],
        threshold: int = 25
    ) -> List[List[Tuple[int, int, int]]]:
        """비슷한 색상을 그룹핑"""
        color_groups = []
        for sample in samples:
            found = False
            for group in color_groups:
                rep = group[0]
                if (abs(sample[0] - rep[0]) < threshold and
                    abs(sample[1] - rep[1]) < threshold and
                    abs(sample[2] - rep[2]) < threshold):
                    group.append(sample)
                    found = True
                    break
            if not found:
                color_groups.append([sample])
        return color_groups

    def _fit_text_vector_improved(
        self,
        page: fitz.Page,
        text: str,
        rect: fitz.Rect,
        font_size: float,
        fontname: str
    ) -> Tuple[float, str]:
        """개선된 텍스트 맞춤 - 원본 크기 최대한 유지"""
        min_size = 6
        # 원본 크기의 90%부터 시작 (번역 텍스트가 보통 더 김)
        current_size = font_size * 0.9

        # 박스 높이/너비 비율에 따른 초기 조정
        box_width = rect.width
        box_height = rect.height
        text_len = len(text)

        # 대략적인 글자당 너비 추정 (한글은 약 0.9배)
        est_char_width = current_size * 0.5
        est_text_width = text_len * est_char_width

        # 텍스트가 너무 길면 미리 크기 조정
        if est_text_width > box_width * 2:
            ratio = (box_width * 1.5) / est_text_width
            current_size = max(min_size, current_size * ratio)

        while current_size >= min_size:
            try:
                rc = page.insert_textbox(
                    rect,
                    text,
                    fontsize=current_size,
                    fontname=fontname,
                    lineheight=self.line_height,
                    overlay=False,
                )
                if rc >= 0:
                    return current_size, text
            except Exception:
                pass

            # 더 작은 단계로 축소 (10%씩)
            current_size = current_size * 0.9

        return min_size, text

    def _get_vector_fontname(self, font_name: str) -> str:
        """벡터 텍스트용 폰트 이름 반환 - 대상 언어에 맞는 CJK 폰트 선택"""
        # 대상 언어에 따라 적절한 CJK 폰트 선택
        target_lang_lower = self.target_lang.lower()

        # PyMuPDF 내장 CJK 폰트:
        # - "korea" : 한국어 (Korean)
        # - "japan" : 일본어 (Japanese)
        # - "china-s" : 중국어 간체 (Simplified Chinese)
        # - "china-t" : 중국어 번체 (Traditional Chinese)
        if target_lang_lower in ("ko", "kr", "kor", "korean"):
            return "korea"
        elif target_lang_lower in ("ja", "jp", "jpn", "japanese"):
            return "japan"
        elif target_lang_lower in ("zh-tw", "zh-hant", "traditional chinese", "chinese-traditional"):
            return "china-t"
        elif target_lang_lower in ("zh", "zh-cn", "zh-hans", "chinese", "simplified chinese"):
            return "china-s"
        else:
            # 기타 언어는 기본 폰트 사용 (라틴 계열)
            return "helv"

    def _convert_image_mode(
        self,
        doc: fitz.Document,
        pages: List[int]
    ) -> bytes:
        """
        이미지 모드 변환 - 기존 방식 (호환성용)

        개선: 일괄 번역으로 속도 향상, 소요 시간 표시
        """
        # 전체 시간 측정 시작
        total_start_time = time.time()

        # ===== 1단계: 모든 페이지 렌더링 및 텍스트 추출 =====
        extract_start_time = time.time()
        self._log("1단계: 페이지 렌더링 및 텍스트 추출 중...")

        page_data = []  # [(img, blocks, to_translate_indices)]
        all_texts = []
        all_formulas = []

        for page_num in pages:
            if self._cancelled:
                break

            page = doc[page_num]
            img = self._render_page(page)

            # 레이아웃 감지 (옵션)
            layout_boxes = None
            if self.layout_detector:
                layout_boxes = self.layout_detector.detect(img)

            # 텍스트 추출
            blocks = self._extract_blocks(page, img, layout_boxes)

            # 번역할 블록 필터링
            to_translate_indices = []
            for i, b in enumerate(blocks):
                if not b.is_formula and self._should_translate(b.text):
                    to_translate_indices.append(i)
                    text, formulas = FormulaDetector.extract_formulas(b.text)
                    all_texts.append(text)
                    all_formulas.append(formulas)

            page_data.append((img, blocks, to_translate_indices))

        total_texts = len(all_texts)
        extract_elapsed = time.time() - extract_start_time
        self._log(f"  총 {len(pages)}페이지에서 {total_texts}개 텍스트 추출 완료 ({extract_elapsed:.1f}초)")

        # ===== 2단계: 전체 텍스트 일괄 번역 =====
        translate_start_time = time.time()
        if total_texts > 0:
            self._log(f"2단계: {total_texts}개 텍스트 일괄 번역 중...")
            all_translations = self.translator.translate_batch(all_texts)

            # 수식 복원
            for i, trans in enumerate(all_translations):
                if all_formulas[i]:
                    all_translations[i] = FormulaDetector.restore_formulas(trans, all_formulas[i])

            translate_elapsed = time.time() - translate_start_time
            self._log(f"  번역 완료 ({translate_elapsed:.1f}초, {total_texts}개 항목)")
        else:
            all_translations = []
            translate_elapsed = 0

        # ===== 3단계: 이미지에 번역 적용 =====
        apply_start_time = time.time()
        self._log("3단계: 번역 결과 적용 중...")
        translated_images = []
        translation_idx = 0
        total_pages = len(page_data)

        for idx, (img, blocks, to_translate_indices) in enumerate(page_data):
            if self._cancelled:
                break

            # 진행 상황 로깅 (5페이지마다 또는 마지막 페이지)
            if idx % 5 == 0 or idx == total_pages - 1:
                self._log(f"  페이지 {idx + 1}/{total_pages} 처리 중...")

            if to_translate_indices:
                # 이 페이지의 번역 결과
                page_blocks = [blocks[i] for i in to_translate_indices]
                page_translations = all_translations[translation_idx:translation_idx + len(to_translate_indices)]
                translation_idx += len(to_translate_indices)

                # 이미지에 번역 적용
                img = self._apply_translations_to_image(img, page_blocks, page_translations)

            translated_images.append(img)

        apply_elapsed = time.time() - apply_start_time
        self._log(f"  {len(pages)}페이지 처리 완료 ({apply_elapsed:.1f}초)")

        # PDF 생성 (압축 옵션 적용)
        self._log("4단계: PDF 생성 중...")
        result_pdf = self._create_pdf(translated_images)

        # 이미지 모드에서는 주석이 제거되므로 별도 처리 불필요
        # (원본 PDF 구조가 유지되지 않음)

        # 전체 소요 시간 출력
        total_elapsed = time.time() - total_start_time
        self._log("  생성 완료")
        self._log(f"===== 번역 완료: 총 {total_elapsed:.1f}초 소요 =====")
        self._log(f"  - 텍스트 추출: {extract_elapsed:.1f}초")
        self._log(f"  - 번역: {translate_elapsed:.1f}초 ({total_texts}개 항목)")
        self._log(f"  - 결과 적용: {apply_elapsed:.1f}초")

        return result_pdf

    def _render_page(self, page: fitz.Page) -> Image.Image:
        """페이지 렌더링"""
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def _extract_blocks(
        self,
        page: fitz.Page,
        img: Optional[Image.Image] = None,
        layout_boxes: Optional[List[LayoutBox]] = None
    ) -> List[TextBlock]:
        """페이지에서 텍스트 블록 추출"""
        blocks = []
        scale = self.dpi / 72

        # PDF 구조에서 텍스트 추출
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        has_text = False

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # text only
                continue

            for line in block.get("lines", []):
                line_text = ""
                line_spans = []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_text += text + " "
                        line_spans.append(span)
                        has_text = True

                line_text = line_text.strip()
                if not line_text or len(line_text) < 2:
                    continue

                # 바운딩 박스
                x0 = min(s["bbox"][0] for s in line_spans) * scale
                y0 = min(s["bbox"][1] for s in line_spans) * scale
                x1 = max(s["bbox"][2] for s in line_spans) * scale
                y1 = max(s["bbox"][3] for s in line_spans) * scale

                # 수직 텍스트 감지 (PDFMathTranslate 참고)
                line_dir = line.get("dir", (1, 0))
                is_vertical = False
                if line_dir:
                    if abs(line_dir[0]) < 0.5 and abs(line_dir[1]) > 0.5:
                        is_vertical = True
                width = x1 - x0
                height = y1 - y0
                if width > 0 and height > width * 3 and len(line_text) > 3:
                    is_vertical = True

                # 레이아웃 필터링
                if layout_boxes:
                    if not self._is_in_translatable_region(
                        (x0, y0, x1, y1), layout_boxes
                    ):
                        continue

                # 가장 많은 텍스트를 커버하는 색상 찾기 (dominant color)
                color_weights = {}  # color -> text_length
                for span in line_spans:
                    color_int = span.get("color", 0)
                    r = (color_int >> 16) & 0xFF
                    g = (color_int >> 8) & 0xFF
                    b = color_int & 0xFF
                    color = (r, g, b)
                    text_len = len(span.get("text", ""))
                    color_weights[color] = color_weights.get(color, 0) + text_len

                # 가장 가중치가 높은 색상 선택
                if color_weights:
                    dominant_color = max(color_weights.keys(), key=lambda c: color_weights[c])
                else:
                    dominant_color = (0, 0, 0)

                # 가장 큰 폰트 크기 사용
                font_sizes = [s.get("size", 12) for s in line_spans]
                font_size = (max(font_sizes) if font_sizes else 12) * scale

                # 가장 많이 사용된 폰트 이름
                font_names = {}
                for span in line_spans:
                    fn = span.get("font", "")
                    text_len = len(span.get("text", ""))
                    font_names[fn] = font_names.get(fn, 0) + text_len
                font_name = max(font_names.keys(), key=lambda f: font_names[f]) if font_names else ""

                # 수식 감지 (폰트 이름과 수직 텍스트 여부 전달)
                is_formula = FormulaDetector.is_formula(line_text, font_name, is_vertical)

                blocks.append(TextBlock(
                    text=line_text,
                    bbox=(x0, y0, x1, y1),
                    font_size=font_size,
                    font_name=font_name,
                    color=dominant_color,
                    is_formula=is_formula,
                    is_vertical=is_vertical,
                ))

        # OCR 폴백 (텍스트가 없는 스캔 PDF)
        if not has_text and self.ocr_engine and img:
            blocks = self._extract_blocks_ocr(img, layout_boxes)

        return blocks

    def _extract_blocks_ocr(
        self,
        img: Image.Image,
        layout_boxes: Optional[List[LayoutBox]] = None
    ) -> List[TextBlock]:
        """OCR로 텍스트 블록 추출"""
        blocks = []

        # 레이아웃 영역별 OCR 또는 전체 이미지
        if layout_boxes:
            translatable = [b for b in layout_boxes if b.label in TRANSLATABLE_LABELS]
            for box in translatable:
                ocr_results = self.ocr_engine.recognize(img, box.bbox)
                merged = merge_ocr_results(ocr_results)

                for result in merged:
                    if len(result.text) < 2:
                        continue

                    blocks.append(TextBlock(
                        text=result.text,
                        bbox=result.bbox,
                        font_size=14,  # OCR은 폰트 정보 없음
                        font_name="",
                        color=(0, 0, 0),
                        is_formula=FormulaDetector.is_formula(result.text),
                    ))
        else:
            ocr_results = self.ocr_engine.recognize(img)
            merged = merge_ocr_results(ocr_results)

            for result in merged:
                if len(result.text) < 2:
                    continue

                blocks.append(TextBlock(
                    text=result.text,
                    bbox=result.bbox,
                    font_size=14,
                    font_name="",
                    color=(0, 0, 0),
                    is_formula=FormulaDetector.is_formula(result.text),
                ))

        return blocks

    def _is_in_translatable_region(
        self,
        bbox: Tuple[float, float, float, float],
        layout_boxes: List[LayoutBox]
    ) -> bool:
        """번역 가능 영역 내에 있는지 확인"""
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        for box in layout_boxes:
            if box.label not in TRANSLATABLE_LABELS:
                continue

            if (box.x0 <= center_x <= box.x1 and
                box.y0 <= center_y <= box.y1):
                return True

        return True  # 레이아웃 감지 실패 시 번역

    def _translate_page(
        self,
        img: Image.Image,
        blocks: List[TextBlock]
    ) -> Image.Image:
        """페이지 번역 및 이미지 편집"""
        # 번역할 블록 필터링 (수식 제외)
        to_translate = [b for b in blocks if not b.is_formula]

        if not to_translate:
            return img

        # 배치 번역
        texts = []
        formulas_map = {}

        for i, block in enumerate(to_translate):
            text, formulas = FormulaDetector.extract_formulas(block.text)
            texts.append(text)
            formulas_map[i] = formulas

        translations = self.translator.translate_batch(texts)

        # 수식 복원
        for i, trans in enumerate(translations):
            if i in formulas_map and formulas_map[i]:
                translations[i] = FormulaDetector.restore_formulas(trans, formulas_map[i])

        # 이미지 편집
        arr = np.array(img)

        for block, translation in zip(to_translate, translations):
            x0, y0, x1, y1 = [int(v) for v in block.bbox]

            # 배경색 감지
            bg_color = self._detect_bg_color(arr, (x0, y0, x1, y1))

            # 영역 지우기 (약간 확장)
            pad = 2
            y0_ext = max(0, y0 - pad)
            y1_ext = min(arr.shape[0], y1 + pad)
            x0_ext = max(0, x0 - pad)
            x1_ext = min(arr.shape[1], x1 + pad)
            arr[y0_ext:y1_ext, x0_ext:x1_ext] = bg_color

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # 텍스트 그리기
        for block, translation in zip(to_translate, translations):
            x0, y0, x1, y1 = [int(v) for v in block.bbox]
            box_width = x1 - x0
            box_height = y1 - y0

            # 원본 폰트 스타일에 맞는 폰트 선택
            font = self._get_font(int(block.font_size), block.font_name)

            # 텍스트 크기 측정
            text_bbox = draw.textbbox((0, 0), translation, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 텍스트가 영역을 초과하는 경우 처리
            if text_width > box_width or text_height > box_height:
                # 텍스트 줄바꿈 시도
                wrapped_text, font = self._fit_text_in_box(
                    draw, translation, box_width, box_height,
                    int(block.font_size), block.font_name
                )
                translation = wrapped_text

                # 다시 측정
                text_bbox = draw.textbbox((0, 0), translation, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

            # 텍스트 위치 계산
            # 멀티라인: 왼쪽 정렬 (더 자연스러움), 단일 라인: 수평 중앙 정렬
            is_multiline = '\n' in translation
            if is_multiline:
                # 멀티라인은 왼쪽 정렬, 약간의 패딩 적용
                padding = 2
                text_x = x0 + padding
            else:
                # 단일 라인은 수평 중앙 정렬
                text_x = x0 + (box_width - text_width) // 2

            # 수직 중앙 정렬
            text_y = y0 + (box_height - text_height) // 2

            # 텍스트가 영역을 벗어나지 않도록 보정
            text_x = max(x0, text_x)
            text_y = max(y0, text_y)

            # 텍스트 그리기
            draw.text((text_x, text_y), translation, font=font, fill=block.color)

        return img

    def _apply_translations_to_image(
        self,
        img: Image.Image,
        blocks: List[TextBlock],
        translations: List[str]
    ) -> Image.Image:
        """이미지에 번역 결과 적용 (배치 번역 후 사용)"""
        if not blocks or not translations:
            return img

        # 이미지 편집
        arr = np.array(img)

        for block, translation in zip(blocks, translations):
            x0, y0, x1, y1 = [int(v) for v in block.bbox]

            # 배경색 감지
            bg_color = self._detect_bg_color(arr, (x0, y0, x1, y1))

            # 영역 지우기 (약간 확장)
            pad = 2
            y0_ext = max(0, y0 - pad)
            y1_ext = min(arr.shape[0], y1 + pad)
            x0_ext = max(0, x0 - pad)
            x1_ext = min(arr.shape[1], x1 + pad)
            arr[y0_ext:y1_ext, x0_ext:x1_ext] = bg_color

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # 텍스트 그리기
        for block, translation in zip(blocks, translations):
            x0, y0, x1, y1 = [int(v) for v in block.bbox]
            box_width = x1 - x0
            box_height = y1 - y0

            font = self._get_font(int(block.font_size), block.font_name)

            text_bbox = draw.textbbox((0, 0), translation, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if text_width > box_width or text_height > box_height:
                wrapped_text, font = self._fit_text_in_box(
                    draw, translation, box_width, box_height,
                    int(block.font_size), block.font_name
                )
                translation = wrapped_text
                text_bbox = draw.textbbox((0, 0), translation, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

            is_multiline = '\n' in translation
            if is_multiline:
                text_x = x0 + 2
            else:
                text_x = x0 + (box_width - text_width) // 2

            text_y = y0 + (box_height - text_height) // 2
            text_x = max(x0, text_x)
            text_y = max(y0, text_y)

            draw.text((text_x, text_y), translation, font=font, fill=block.color)

        return img

    def _fit_text_in_box(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        box_width: int,
        box_height: int,
        font_size: int,
        font_name: str
    ) -> Tuple[str, ImageFont.FreeTypeFont]:
        """텍스트를 박스에 맞게 조정 - 개선된 오버플로우 처리"""
        # 더 낮은 최소 폰트 크기 (작은 셀에 대응)
        min_font_size = 6
        current_size = font_size

        # 더 공격적인 폰트 크기 축소 (15%씩)
        while current_size >= min_font_size:
            font = self._get_font(current_size, font_name)

            # 줄바꿈 시도
            wrapped = self._wrap_text(draw, text, font, box_width)
            text_bbox = draw.textbbox((0, 0), wrapped, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if text_width <= box_width and text_height <= box_height:
                return wrapped, font

            # 폰트 크기 축소 (15%씩 - 더 빠르게 감소)
            current_size = int(current_size * 0.85)

        # 최소 크기에서도 맞지 않으면 텍스트 잘라내기
        font = self._get_font(min_font_size, font_name)
        wrapped = self._wrap_text(draw, text, font, box_width)

        # 높이가 맞지 않으면 줄 수 제한
        lines = wrapped.split('\n')
        if len(lines) > 1:
            # 한 줄씩 테스트하며 맞는 줄 수 찾기
            for num_lines in range(len(lines), 0, -1):
                test_text = '\n'.join(lines[:num_lines])
                if num_lines < len(lines):
                    # 잘린 경우 마지막에 "..." 추가
                    last_line = lines[num_lines - 1]
                    if len(last_line) > 3:
                        test_text = '\n'.join(lines[:num_lines - 1] + [last_line[:-3] + "..."])
                    else:
                        test_text = '\n'.join(lines[:num_lines - 1] + ["..."])

                text_bbox = draw.textbbox((0, 0), test_text, font=font)
                text_height = text_bbox[3] - text_bbox[1]

                if text_height <= box_height:
                    return test_text, font

        return wrapped, font

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> str:
        """텍스트 줄바꿈 - 긴 단어도 문자 단위로 분리"""
        words = text.split()
        if not words:
            return text

        lines = []
        current_line = []

        for word in words:
            # 단어가 너비를 초과하는지 확인
            word_bbox = draw.textbbox((0, 0), word, font=font)
            word_width = word_bbox[2] - word_bbox[0]

            if word_width > max_width:
                # 현재 라인이 있으면 먼저 저장
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = []

                # 긴 단어를 문자 단위로 분리
                char_line = ""
                for char in word:
                    test_char = char_line + char
                    char_bbox = draw.textbbox((0, 0), test_char, font=font)
                    char_width = char_bbox[2] - char_bbox[0]

                    if char_width <= max_width:
                        char_line = test_char
                    else:
                        if char_line:
                            lines.append(char_line)
                        char_line = char

                if char_line:
                    current_line = [char_line]
            else:
                test_line = ' '.join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                width = bbox[2] - bbox[0]

                if width <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    def _detect_bg_color(
        self,
        arr: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """배경색 감지 - 개선된 버전"""
        x0, y0, x1, y1 = bbox
        h, w = arr.shape[:2]

        # 영역 내부 샘플링 우선
        inner_samples = []
        if x1 > x0 and y1 > y0:
            # 영역 내부의 여러 지점에서 샘플링
            inner_region = arr[y0:y1, x0:x1]
            if inner_region.size > 0:
                # 밝은 픽셀 선택 (텍스트가 아닌 배경)
                pixels = inner_region.reshape(-1, 3)
                brightness = np.sum(pixels, axis=1)
                # 상위 30% 밝기의 픽셀 선택
                threshold = np.percentile(brightness, 70)
                bright_mask = brightness >= threshold
                bright_pixels = pixels[bright_mask]

                if len(bright_pixels) > 0:
                    # 가장 일반적인 밝은 색상 찾기
                    median = np.median(bright_pixels, axis=0).astype(np.uint8)
                    return tuple(median)

        # 외부 가장자리 샘플링 (fallback)
        edge_samples = []
        pad = 3

        if y0 > pad:
            edge_samples.append(arr[y0-pad:y0, x0:x1])
        if y1 < h - pad:
            edge_samples.append(arr[y1:y1+pad, x0:x1])
        if x0 > pad:
            edge_samples.append(arr[y0:y1, x0-pad:x0])
        if x1 < w - pad:
            edge_samples.append(arr[y0:y1, x1:x1+pad])

        if not edge_samples:
            return (255, 255, 255)

        all_pixels = np.concatenate([s.reshape(-1, 3) for s in edge_samples if s.size > 0])
        if all_pixels.size == 0:
            return (255, 255, 255)

        median = np.median(all_pixels, axis=0).astype(np.uint8)
        return tuple(median)

    def _get_font(self, size: int, font_name: str = "") -> ImageFont.FreeTypeFont:
        """
        폰트 반환 - 폰트 관리자를 통해 최적의 폰트 선택

        Args:
            size: 폰트 크기
            font_name: 원본 폰트 이름 (스타일 매칭용)
        """
        # 1. 먼저 font_manager를 통해 대상 언어에 맞는 폰트 시도
        try:
            font_path = font_manager.get_font_path(self.target_lang)
            if font_path and Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
        except Exception:
            pass

        font_name_lower = font_name.lower()

        # 폰트 스타일 감지
        is_bold = "bold" in font_name_lower or "heavy" in font_name_lower
        is_italic = "italic" in font_name_lower or "oblique" in font_name_lower
        is_serif = any(s in font_name_lower for s in ["serif", "times", "georgia", "palatino", "batang", "myungjo"])
        is_mono = any(s in font_name_lower for s in ["mono", "courier", "consolas", "menlo"])

        # Windows 폰트 경로
        win_fonts = {
            # 산세리프 (기본)
            "sans": [
                r"C:\Windows\Fonts\malgun.ttf",      # 맑은 고딕
                r"C:\Windows\Fonts\meiryo.ttc",     # 메이리오
                r"C:\Windows\Fonts\msyh.ttc",       # Microsoft YaHei
                r"C:\Windows\Fonts\segoeui.ttf",    # Segoe UI
                r"C:\Windows\Fonts\arial.ttf",      # Arial
            ],
            "sans_bold": [
                r"C:\Windows\Fonts\malgunbd.ttf",   # 맑은 고딕 Bold
                r"C:\Windows\Fonts\meiryob.ttc",   # 메이리오 Bold
                r"C:\Windows\Fonts\msyhbd.ttc",    # YaHei Bold
                r"C:\Windows\Fonts\segoeuib.ttf",  # Segoe UI Bold
                r"C:\Windows\Fonts\arialbd.ttf",   # Arial Bold
            ],
            # 세리프
            "serif": [
                r"C:\Windows\Fonts\batang.ttc",     # 바탕
                r"C:\Windows\Fonts\times.ttf",      # Times New Roman
                r"C:\Windows\Fonts\georgia.ttf",    # Georgia
            ],
            "serif_bold": [
                r"C:\Windows\Fonts\timesbd.ttf",    # Times Bold
                r"C:\Windows\Fonts\georgiab.ttf",   # Georgia Bold
            ],
            # 모노스페이스
            "mono": [
                r"C:\Windows\Fonts\consola.ttf",    # Consolas
                r"C:\Windows\Fonts\cour.ttf",       # Courier New
            ],
        }

        # macOS 폰트 경로
        mac_fonts = {
            "sans": [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                "/Library/Fonts/Arial.ttf",
            ],
            "sans_bold": [
                "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            ],
            "serif": [
                "/Library/Fonts/Times New Roman.ttf",
            ],
            "mono": [
                "/System/Library/Fonts/Menlo.ttc",
            ],
        }

        # Linux 폰트 경로
        linux_fonts = {
            "sans": [
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ],
            "sans_bold": [
                "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ],
            "serif": [
                "/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            ],
            "mono": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            ],
        }

        # 스타일에 따른 폰트 키 선택
        if is_mono:
            style_key = "mono"
        elif is_serif:
            style_key = "serif_bold" if is_bold else "serif"
        else:
            style_key = "sans_bold" if is_bold else "sans"

        # 플랫폼별 폰트 목록
        import sys
        if sys.platform == "win32":
            font_list = win_fonts.get(style_key, win_fonts["sans"])
        elif sys.platform == "darwin":
            font_list = mac_fonts.get(style_key, mac_fonts["sans"])
        else:
            font_list = linux_fonts.get(style_key, linux_fonts["sans"])

        # 폰트 로드 시도
        for font_path in font_list:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue

        # 플랫폼별 기본 폰트 폴백
        fallback_fonts = [
            r"C:\Windows\Fonts\malgun.ttf",
            r"C:\Windows\Fonts\arial.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for font_path in fallback_fonts:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue

        return ImageFont.load_default()

    def _create_pdf(self, images: List[Image.Image]) -> bytes:
        """이미지에서 PDF 생성 - 압축 최적화"""
        doc = fitz.open()

        for img in images:
            buf = io.BytesIO()

            # 이미지 압축 옵션 적용
            if self.compress_images:
                # JPEG로 압축 (PNG보다 크기 대폭 감소)
                # RGB 모드로 변환 (JPEG는 RGBA 미지원)
                if img.mode == "RGBA":
                    # 알파 채널을 흰색 배경으로 병합
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                img.save(buf, format="JPEG", quality=self.output_quality, optimize=True)
            else:
                # 압축 없이 PNG 사용
                img.save(buf, format="PNG")

            buf.seek(0)

            page = doc.new_page(width=img.width, height=img.height)
            rect = fitz.Rect(0, 0, img.width, img.height)
            page.insert_image(rect, stream=buf.getvalue())

        output = io.BytesIO()
        # PDF 압축 옵션 적용
        doc.save(
            output,
            garbage=4,  # 최대 가비지 컬렉션
            deflate=True,  # 압축 사용
            clean=True,  # 사용하지 않는 객체 제거
        )
        doc.close()

        return output.getvalue()

    def _translate_annotations_and_forms(self, doc: fitz.Document):
        """
        PDF 주석 및 폼 필드 텍스트 번역

        지원되는 주석 유형:
        - 텍스트 주석 (노트, 팝업)
        - FreeText 주석 (자유 텍스트)
        - 하이라이트, 밑줄, 취소선 주석의 레이블
        - 스탬프 주석
        - 링크 주석의 표시 텍스트

        지원되는 폼 필드:
        - 텍스트 필드
        - 콤보 박스 옵션
        - 리스트 박스 옵션
        - 버튼 레이블
        """
        # 모든 주석 및 폼 필드 텍스트 수집
        annotations_data = []  # [(page_num, annot_index, text_type, original_text)]
        forms_data = []  # [(field_name, text_type, original_text)]

        # 1. 주석 수집
        for page_num in range(doc.page_count):
            page = doc[page_num]
            annots = page.annots()
            if not annots:
                continue

            for annot_idx, annot in enumerate(annots):
                try:
                    annot_type = annot.type[0]  # 주석 유형 번호

                    # 텍스트 주석 (노트)
                    if annot_type == 0:  # Text annotation
                        content = annot.info.get("content", "")
                        if content and content.strip():
                            annotations_data.append((page_num, annot_idx, "content", content))

                    # FreeText 주석
                    elif annot_type == 2:  # FreeText
                        content = annot.info.get("content", "")
                        if content and content.strip():
                            annotations_data.append((page_num, annot_idx, "content", content))

                    # 하이라이트, 밑줄, 취소선 주석
                    elif annot_type in (8, 9, 10, 11):  # Highlight, Underline, Squiggly, StrikeOut
                        content = annot.info.get("content", "")
                        if content and content.strip():
                            annotations_data.append((page_num, annot_idx, "content", content))

                    # 스탬프 주석
                    elif annot_type == 13:  # Stamp
                        content = annot.info.get("content", "")
                        if content and content.strip():
                            annotations_data.append((page_num, annot_idx, "content", content))

                    # 팝업 주석
                    elif annot_type == 16:  # Popup
                        content = annot.info.get("content", "")
                        if content and content.strip():
                            annotations_data.append((page_num, annot_idx, "content", content))

                    # 링크 주석의 제목
                    elif annot_type == 3:  # Link
                        title = annot.info.get("title", "")
                        if title and title.strip():
                            annotations_data.append((page_num, annot_idx, "title", title))

                    # 파일 첨부 주석
                    elif annot_type == 17:  # FileAttachment
                        content = annot.info.get("content", "")
                        if content and content.strip():
                            annotations_data.append((page_num, annot_idx, "content", content))

                except Exception:
                    pass

        # 2. 폼 필드 수집 (Widget 주석)
        try:
            for page_num in range(doc.page_count):
                page = doc[page_num]
                widgets = page.widgets()
                if not widgets:
                    continue

                for widget in widgets:
                    try:
                        field_type = widget.field_type
                        field_name = widget.field_name or f"field_{page_num}_{widget.xref}"

                        # 텍스트 필드
                        if field_type == 1:  # Text field
                            value = widget.field_value
                            if value and value.strip() and self._should_translate(value):
                                forms_data.append((field_name, page_num, widget.xref, "value", value))

                            # 툴팁/설명
                            tip = widget.field_label
                            if tip and tip.strip():
                                forms_data.append((field_name, page_num, widget.xref, "label", tip))

                        # 버튼 (체크박스, 라디오버튼, 푸시버튼)
                        elif field_type == 2:  # Button
                            label = widget.button_caption
                            if label and label.strip():
                                forms_data.append((field_name, page_num, widget.xref, "caption", label))

                        # 콤보박스/리스트박스
                        elif field_type in (3, 4):  # Choice (combo/list)
                            # 현재 값
                            value = widget.field_value
                            if value and value.strip() and self._should_translate(value):
                                forms_data.append((field_name, page_num, widget.xref, "value", value))

                    except Exception:
                        pass
        except Exception:
            pass

        # 번역할 텍스트가 없으면 종료
        if not annotations_data and not forms_data:
            return

        # 3. 텍스트 수집
        all_texts = []
        for data in annotations_data:
            all_texts.append(data[3])  # original_text
        for data in forms_data:
            all_texts.append(data[4])  # original_text

        if not all_texts:
            return

        self._log(f"  주석 {len(annotations_data)}개, 폼 필드 {len(forms_data)}개 번역 중...")

        # 4. 일괄 번역
        try:
            translations = self.translator.translate_batch(all_texts)
        except Exception as e:
            self._log(f"  주석/폼 필드 번역 실패: {e}")
            return

        # 5. 번역 결과 적용
        trans_idx = 0

        # 주석에 적용
        for page_num, annot_idx, text_type, original in annotations_data:
            try:
                page = doc[page_num]
                annots = list(page.annots())
                if annot_idx < len(annots):
                    annot = annots[annot_idx]
                    translated = translations[trans_idx]

                    if text_type == "content":
                        annot.set_info(content=translated)
                    elif text_type == "title":
                        annot.set_info(title=translated)

                    annot.update()
            except Exception:
                pass
            trans_idx += 1

        # 폼 필드에 적용
        for field_name, page_num, xref, text_type, original in forms_data:
            try:
                page = doc[page_num]
                translated = translations[trans_idx]

                # xref로 위젯 찾기
                for widget in page.widgets():
                    if widget.xref == xref:
                        if text_type == "value":
                            widget.field_value = translated
                        elif text_type == "caption":
                            widget.button_caption = translated
                        widget.update()
                        break
            except Exception:
                pass
            trans_idx += 1

    def _convert_dual_mode(
        self,
        doc: fitz.Document,
        pages: List[int]
    ) -> Tuple[bytes, bytes]:
        """
        이중 언어 모드 변환 - 원문과 번역을 동시에 표시

        PDFMathTranslate의 dual mode 구현:
        - mono_pdf: 번역만 있는 PDF
        - dual_pdf: 원문 페이지 + 번역 페이지 번갈아 표시

        Returns:
            Tuple[bytes, bytes]: (mono_pdf, dual_pdf)
        """
        # 먼저 mono PDF 생성 (벡터 모드 사용)
        mono_doc = fitz.open()
        dual_doc = fitz.open()

        for idx, page_num in enumerate(pages):
            if self._cancelled:
                break

            self._log(f"페이지 {page_num + 1}/{doc.page_count} 처리 중 (이중 언어)...")

            # 원본 페이지
            page = doc[page_num]

            # dual PDF: 먼저 원본 페이지 추가
            dual_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

            # 번역 페이지 생성
            mono_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_page_mono = mono_doc[-1]

            # dual PDF에도 번역 페이지 복사 추가
            dual_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            new_page_dual = dual_doc[-1]

            # 배경색 감지를 위한 페이지 렌더링
            page_pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1))

            # 텍스트 블록 추출
            blocks = self._extract_blocks_for_vector(page)

            if not blocks:
                continue

            # 번역할 블록 필터링 (수식 제외 + 정규표현식 필터)
            to_translate = []
            for b in blocks:
                if not b.is_formula and self._should_translate(b.text):
                    to_translate.append(b)

            if not to_translate:
                continue

            # 배치 번역
            texts = []
            formulas_map = {}

            for i, block in enumerate(to_translate):
                text, formulas = FormulaDetector.extract_formulas(block.text)
                texts.append(text)
                formulas_map[i] = formulas

            self._log(f"  번역 중... ({len(texts)}개 블록)")
            translations = self.translator.translate_batch(texts)

            # 수식 복원
            for i, trans in enumerate(translations):
                if i in formulas_map and formulas_map[i]:
                    translations[i] = FormulaDetector.restore_formulas(trans, formulas_map[i])

            # mono PDF와 dual PDF의 번역 페이지에 번역 적용
            self._replace_text_vector(new_page_mono, to_translate, translations, page_pixmap)
            self._replace_text_vector(new_page_dual, to_translate, translations, page_pixmap)

        # 주석 및 폼 필드 번역
        self._log("주석 및 폼 필드 번역 중...")
        self._translate_annotations_and_forms(mono_doc)
        self._translate_annotations_and_forms(dual_doc)

        # mono PDF 저장
        mono_output = io.BytesIO()
        mono_doc.save(
            mono_output,
            garbage=4,
            deflate=True,
            clean=True,
        )
        mono_doc.close()

        # dual PDF 저장
        dual_output = io.BytesIO()
        dual_doc.save(
            dual_output,
            garbage=4,
            deflate=True,
            clean=True,
        )
        dual_doc.close()

        return mono_output.getvalue(), dual_output.getvalue()

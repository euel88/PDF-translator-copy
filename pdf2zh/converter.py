"""
PDF 변환 모듈 - PDFMathTranslate 구조 기반
핵심 PDF 파싱 및 변환 로직
"""
import io
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import numpy as np

from pdf2zh.translator import BaseTranslator, create_translator
from pdf2zh.doclayout import LayoutDetector, LayoutBox, TRANSLATABLE_LABELS
from pdf2zh.ocr import TesseractOCR, OCRResult, merge_ocr_results


@dataclass
class TextBlock:
    """텍스트 블록"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float
    font_name: str
    color: Tuple[int, int, int]
    is_formula: bool = False

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
    """수식 감지기"""

    # 수식 패턴
    PATTERNS = [
        r'\$[^$]+\$',           # LaTeX inline
        r'\$\$[^$]+\$\$',       # LaTeX block
        r'\\[\(\[][^\\]*\\[\)\]]',  # LaTeX brackets
        r'[α-ωΑ-Ω]+',           # Greek letters
        r'[∑∏∫∂∇√∞∝∀∃∈∉⊂⊃⊆⊇∪∩]+',  # Math symbols
    ]

    @classmethod
    def is_formula(cls, text: str) -> bool:
        """텍스트가 수식인지 확인"""
        for pattern in cls.PATTERNS:
            if re.search(pattern, text):
                return True
        # 숫자/기호 비율 체크
        alpha_count = sum(1 for c in text if c.isalpha())
        total = len(text.replace(" ", ""))
        if total > 0 and alpha_count / total < 0.3:
            return True
        return False

    @classmethod
    def extract_formulas(cls, text: str) -> Tuple[str, List[str]]:
        """수식 추출 및 플레이스홀더 적용"""
        formulas = []
        result = text

        for pattern in cls.PATTERNS:
            matches = list(re.finditer(pattern, result))
            for match in reversed(matches):
                formula = match.group()
                idx = len(formulas)
                placeholder = f"{{{{v{idx}}}}}"
                formulas.append(formula)
                result = result[:match.start()] + placeholder + result[match.end():]

        return result, formulas

    @classmethod
    def restore_formulas(cls, text: str, formulas: List[str]) -> str:
        """플레이스홀더를 수식으로 복원"""
        result = text
        for idx, formula in enumerate(formulas):
            placeholder = f"{{{{v{idx}}}}}"
            result = result.replace(placeholder, formula)
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
    ):
        self.translator = translator
        self.dpi = dpi
        self.callback = callback
        self.use_layout_detection = use_layout_detection
        self.use_ocr = use_ocr
        self.source_lang = source_lang
        self._cancelled = False

        # 레이아웃 감지기
        self.layout_detector = LayoutDetector() if use_layout_detection else None

        # OCR 엔진
        self.ocr_engine = TesseractOCR(source_lang) if use_ocr else None

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

            translated_images = []

            for idx, page_num in enumerate(pages):
                if self._cancelled:
                    return TranslateResult(error="취소됨")

                self._log(f"페이지 {page_num + 1}/{doc.page_count} 처리 중...")

                # 페이지 렌더링
                page = doc[page_num]
                img = self._render_page(page)

                # 레이아웃 감지 (옵션)
                layout_boxes = None
                if self.layout_detector:
                    layout_boxes = self.layout_detector.detect(img)
                    self._log(f"  레이아웃 영역: {len(layout_boxes)}개")

                # 텍스트 추출
                blocks = self._extract_blocks(page, img, layout_boxes)
                self._log(f"  텍스트 블록: {len(blocks)}개")

                if blocks:
                    # 번역
                    img = self._translate_page(img, blocks)

                translated_images.append(img)

            # PDF 생성
            self._log("PDF 생성 중...")
            mono_pdf = self._create_pdf(translated_images)

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(mono_pdf)
                self._log(f"저장됨: {output_path}")

            doc.close()

            return TranslateResult(
                mono_pdf=mono_pdf,
                output_path=output_path,
                page_count=len(pages),
                success=True,
            )

        except Exception as e:
            self._log(f"오류: {str(e)}")
            return TranslateResult(error=str(e))

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

                # 레이아웃 필터링
                if layout_boxes:
                    if not self._is_in_translatable_region(
                        (x0, y0, x1, y1), layout_boxes
                    ):
                        continue

                first = line_spans[0]
                font_size = first.get("size", 12) * scale
                font_name = first.get("font", "")

                color_int = first.get("color", 0)
                r = (color_int >> 16) & 0xFF
                g = (color_int >> 8) & 0xFF
                b = color_int & 0xFF

                is_formula = FormulaDetector.is_formula(line_text)

                blocks.append(TextBlock(
                    text=line_text,
                    bbox=(x0, y0, x1, y1),
                    font_size=font_size,
                    font_name=font_name,
                    color=(r, g, b),
                    is_formula=is_formula,
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

            # 텍스트 위치 (수직 중앙 정렬)
            text_x = x0
            text_y = y0 + (box_height - text_height) // 2

            # 텍스트 그리기
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
        """텍스트를 박스에 맞게 조정"""
        min_font_size = 8
        current_size = font_size

        while current_size >= min_font_size:
            font = self._get_font(current_size, font_name)

            # 줄바꿈 시도
            wrapped = self._wrap_text(draw, text, font, box_width)
            text_bbox = draw.textbbox((0, 0), wrapped, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if text_width <= box_width and text_height <= box_height:
                return wrapped, font

            # 폰트 크기 축소
            current_size = int(current_size * 0.9)

        # 최소 크기로 반환
        font = self._get_font(min_font_size, font_name)
        wrapped = self._wrap_text(draw, text, font, box_width)
        return wrapped, font

    def _wrap_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int
    ) -> str:
        """텍스트 줄바꿈"""
        words = text.split()
        if not words:
            return text

        lines = []
        current_line = []

        for word in words:
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
        """배경색 감지"""
        x0, y0, x1, y1 = bbox
        h, w = arr.shape[:2]

        samples = []
        pad = 3

        if y0 > pad:
            samples.append(arr[y0-pad:y0, x0:x1])
        if y1 < h - pad:
            samples.append(arr[y1:y1+pad, x0:x1])

        if not samples:
            return (255, 255, 255)

        import numpy as np
        all_pixels = np.concatenate([s.reshape(-1, 3) for s in samples if s.size > 0])
        if all_pixels.size == 0:
            return (255, 255, 255)

        median = np.median(all_pixels, axis=0).astype(np.uint8)
        return tuple(median)

    def _get_font(self, size: int, font_name: str = "") -> ImageFont.FreeTypeFont:
        """
        폰트 반환 - 원본 폰트 스타일에 맞게 선택

        Args:
            size: 폰트 크기
            font_name: 원본 폰트 이름 (스타일 매칭용)
        """
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
        """이미지에서 PDF 생성"""
        doc = fitz.open()

        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            page = doc.new_page(width=img.width, height=img.height)
            rect = fitz.Rect(0, 0, img.width, img.height)
            page.insert_image(rect, stream=buf.getvalue())

        output = io.BytesIO()
        doc.save(output)
        doc.close()

        return output.getvalue()

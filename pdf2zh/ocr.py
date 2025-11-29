"""
OCR 모듈 - PDFMathTranslate 구조 기반
Tesseract OCR 통합
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np

from pdf2zh.config import config, TESSERACT_LANGS


@dataclass
class OCRResult:
    """OCR 결과"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    confidence: float
    lang: str = ""

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]


class TesseractOCR:
    """Tesseract OCR 엔진"""

    def __init__(self, lang: str = "eng"):
        self.lang = TESSERACT_LANGS.get(lang, lang)
        self._available = None
        self._setup_path()

    def _setup_path(self):
        """Tesseract 경로 설정"""
        if sys.platform == "win32":
            paths = [
                r"C:\Program Files\Tesseract-OCR",
                r"C:\Program Files (x86)\Tesseract-OCR",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR"),
            ]
            for path in paths:
                tesseract_exe = os.path.join(path, "tesseract.exe")
                if os.path.exists(tesseract_exe):
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    try:
                        import pytesseract
                        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
                    except ImportError:
                        pass
                    break

    def is_available(self) -> bool:
        """Tesseract 사용 가능 여부"""
        if self._available is not None:
            return self._available

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception:
            self._available = False

        return self._available

    def recognize(
        self,
        image: Image.Image,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> List[OCRResult]:
        """
        이미지에서 텍스트 인식

        Args:
            image: PIL 이미지
            bbox: 인식할 영역 (None이면 전체)

        Returns:
            OCRResult 리스트
        """
        if not self.is_available():
            return []

        try:
            import pytesseract

            # 영역 크롭
            if bbox:
                x0, y0, x1, y1 = [int(v) for v in bbox]
                image = image.crop((x0, y0, x1, y1))
                offset_x, offset_y = x0, y0
            else:
                offset_x, offset_y = 0, 0

            # OCR 수행
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )

            results = []
            n_boxes = len(data['text'])

            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])

                if not text or conf < 0:
                    continue

                x = data['left'][i] + offset_x
                y = data['top'][i] + offset_y
                w = data['width'][i]
                h = data['height'][i]

                results.append(OCRResult(
                    text=text,
                    bbox=(x, y, x + w, y + h),
                    confidence=conf / 100.0,
                    lang=self.lang
                ))

            return results

        except Exception as e:
            print(f"OCR 오류: {e}")
            return []

    def recognize_text(self, image: Image.Image) -> str:
        """이미지에서 텍스트만 추출"""
        if not self.is_available():
            return ""

        try:
            import pytesseract
            return pytesseract.image_to_string(image, lang=self.lang).strip()
        except Exception:
            return ""


class OCREngine:
    """OCR 엔진 팩토리"""

    _engines: Dict[str, Any] = {}

    @classmethod
    def get_engine(cls, engine_type: str = "tesseract", lang: str = "eng") -> Any:
        """OCR 엔진 반환"""
        key = f"{engine_type}_{lang}"

        if key not in cls._engines:
            if engine_type == "tesseract":
                cls._engines[key] = TesseractOCR(lang)
            else:
                raise ValueError(f"Unknown OCR engine: {engine_type}")

        return cls._engines[key]

    @classmethod
    def is_available(cls, engine_type: str = "tesseract") -> bool:
        """OCR 엔진 사용 가능 여부"""
        try:
            engine = cls.get_engine(engine_type)
            return engine.is_available()
        except Exception:
            return False


def merge_ocr_results(
    results: List[OCRResult],
    line_threshold: float = 10
) -> List[OCRResult]:
    """
    OCR 결과를 라인별로 병합

    Args:
        results: OCR 결과 리스트
        line_threshold: 같은 라인으로 판단할 y 좌표 차이

    Returns:
        병합된 OCR 결과
    """
    if not results:
        return []

    # y 좌표로 정렬
    sorted_results = sorted(results, key=lambda r: (r.y0, r.x0))

    merged = []
    current_line = [sorted_results[0]]

    for result in sorted_results[1:]:
        last = current_line[-1]

        # 같은 라인인지 확인
        if abs(result.y0 - last.y0) < line_threshold:
            current_line.append(result)
        else:
            # 라인 병합
            merged.append(_merge_line(current_line))
            current_line = [result]

    # 마지막 라인
    if current_line:
        merged.append(_merge_line(current_line))

    return merged


def _merge_line(results: List[OCRResult]) -> OCRResult:
    """라인 내 결과 병합"""
    if len(results) == 1:
        return results[0]

    # x 좌표로 정렬
    sorted_results = sorted(results, key=lambda r: r.x0)

    text = " ".join(r.text for r in sorted_results)
    x0 = min(r.x0 for r in sorted_results)
    y0 = min(r.y0 for r in sorted_results)
    x1 = max(r.x1 for r in sorted_results)
    y1 = max(r.y1 for r in sorted_results)
    conf = sum(r.confidence for r in sorted_results) / len(sorted_results)

    return OCRResult(
        text=text,
        bbox=(x0, y0, x1, y1),
        confidence=conf,
        lang=sorted_results[0].lang
    )


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """OCR을 위한 이미지 전처리"""
    import numpy as np

    # 그레이스케일 변환
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image

    arr = np.array(gray)

    # 이진화 (Otsu's method)
    try:
        import cv2
        _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)
    except ImportError:
        # OpenCV 없이 간단한 이진화
        threshold = np.mean(arr)
        binary = (arr > threshold).astype(np.uint8) * 255
        return Image.fromarray(binary)


def detect_text_regions(
    image: Image.Image,
    min_area: int = 100
) -> List[Tuple[float, float, float, float]]:
    """텍스트 영역 감지"""
    try:
        import cv2
        import numpy as np

        # 그레이스케일
        gray = np.array(image.convert('L'))

        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 모폴로지 연산
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 컨투어 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area:
                regions.append((float(x), float(y), float(x + w), float(y + h)))

        return regions

    except ImportError:
        # OpenCV 없으면 전체 이미지 반환
        w, h = image.size
        return [(0, 0, w, h)]

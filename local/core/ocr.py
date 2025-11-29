"""
OCR 모듈 (Tesseract)
"""
import platform
import os
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image
import numpy as np


@dataclass
class TextBox:
    """감지된 텍스트 박스"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float

    @property
    def x1(self): return self.bbox[0]
    @property
    def y1(self): return self.bbox[1]
    @property
    def x2(self): return self.bbox[2]
    @property
    def y2(self): return self.bbox[3]
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1


@dataclass
class OCRResult:
    """OCR 결과"""
    boxes: List[TextBox]
    full_text: str
    image_size: Tuple[int, int]

    @property
    def box_count(self) -> int:
        return len(self.boxes)


class OCR:
    """Tesseract OCR 엔진"""

    # 언어 코드 매핑
    LANG_MAP = {
        "eng": "eng", "en": "eng", "english": "eng",
        "kor": "kor", "ko": "kor", "korean": "kor",
        "jpn": "jpn", "ja": "jpn", "japanese": "jpn",
        "chi_sim": "chi_sim", "zh": "chi_sim", "chinese": "chi_sim",
        "chi_tra": "chi_tra",
        "fra": "fra", "fr": "fra", "french": "fra",
        "deu": "deu", "de": "deu", "german": "deu",
        "rus": "rus", "ru": "rus", "russian": "rus",
        "spa": "spa", "es": "spa", "spanish": "spa",
        "por": "por", "pt": "por", "portuguese": "por",
        "ita": "ita", "it": "ita", "italian": "ita",
        "vie": "vie", "vi": "vie", "vietnamese": "vie",
        "tha": "tha", "th": "tha", "thai": "tha",
        "ara": "ara", "ar": "ara", "arabic": "ara",
    }

    def __init__(self, lang: str = "eng"):
        self._lang = self.LANG_MAP.get(lang.lower(), "eng")
        self._setup_tesseract()

    def _setup_tesseract(self):
        """Tesseract 경로 설정"""
        import pytesseract

        if platform.system() == "Windows":
            paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]
            for p in paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break

    def set_language(self, lang: str):
        """언어 설정"""
        self._lang = self.LANG_MAP.get(lang.lower(), "eng")

    def extract(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        """
        이미지에서 텍스트 추출

        Args:
            image: PIL 이미지
            confidence_threshold: 신뢰도 임계값 (0-1)

        Returns:
            OCRResult
        """
        import pytesseract

        # 신뢰도를 0-100 스케일로 변환
        conf_thresh = confidence_threshold * 100

        # OCR 실행
        data = pytesseract.image_to_data(
            image,
            lang=self._lang,
            output_type=pytesseract.Output.DICT
        )

        boxes = []
        n = len(data["text"])

        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if not text or conf < conf_thresh:
                continue

            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]

            boxes.append(TextBox(
                text=text,
                bbox=(x, y, x + w, y + h),
                confidence=conf / 100.0,
            ))

        # 인접 박스 병합 (같은 줄)
        merged = self._merge_boxes(boxes)

        # 전체 텍스트
        full_text = "\n".join(b.text for b in merged)

        return OCRResult(
            boxes=merged,
            full_text=full_text,
            image_size=(image.width, image.height),
        )

    def _merge_boxes(
        self,
        boxes: List[TextBox],
        y_threshold: int = 10,
        x_gap: int = 30
    ) -> List[TextBox]:
        """인접 박스를 줄 단위로 병합"""
        if not boxes:
            return []

        # Y 좌표로 정렬
        sorted_boxes = sorted(boxes, key=lambda b: (b.y1, b.x1))

        lines: List[List[TextBox]] = []
        current_line = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            last = current_line[-1]

            # 같은 줄인지 확인
            same_line = abs(box.y1 - last.y1) < y_threshold
            close_x = box.x1 - last.x2 < x_gap

            if same_line and close_x:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]

        lines.append(current_line)

        # 각 줄을 하나의 TextBox로 병합
        merged = []
        for line in lines:
            text = " ".join(b.text for b in line)
            x1 = min(b.x1 for b in line)
            y1 = min(b.y1 for b in line)
            x2 = max(b.x2 for b in line)
            y2 = max(b.y2 for b in line)
            conf = min(b.confidence for b in line)

            merged.append(TextBox(
                text=text,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
            ))

        return merged

    @staticmethod
    def is_available() -> bool:
        """Tesseract 사용 가능 여부 확인"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    @staticmethod
    def get_available_languages() -> List[str]:
        """사용 가능한 언어 목록"""
        try:
            import pytesseract
            return pytesseract.get_languages()
        except Exception:
            return []

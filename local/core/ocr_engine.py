"""
Tesseract 기반 OCR 엔진
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image
import platform
import os


@dataclass
class TextRegion:
    """텍스트 영역 정보"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    polygon: Optional[List[Tuple[int, int]]] = None  # 4점 다각형 좌표


@dataclass
class OCRResult:
    """OCR 결과"""
    regions: List[TextRegion]
    full_text: str
    image_size: Tuple[int, int]  # (width, height)


class OCREngine:
    """Tesseract 기반 OCR 엔진"""

    _instance = None
    _initialized = False
    _current_lang = "eng"

    # 언어 코드 매핑 (앱 언어 -> Tesseract 언어 코드)
    LANG_MAP = {
        "en": "eng",
        "eng": "eng",
        "english": "eng",
        "ko": "kor",
        "kor": "kor",
        "korean": "kor",
        "ja": "jpn",
        "jpn": "jpn",
        "japan": "jpn",
        "japanese": "jpn",
        "ch": "chi_sim",
        "chi_sim": "chi_sim",
        "chi_tra": "chi_tra",
        "chinese": "chi_sim",
        "fr": "fra",
        "fra": "fra",
        "french": "fra",
        "de": "deu",
        "deu": "deu",
        "german": "deu",
        "ru": "rus",
        "rus": "rus",
        "russian": "rus",
        "es": "spa",
        "spa": "spa",
        "spanish": "spa",
        "pt": "por",
        "por": "por",
        "portuguese": "por",
        "it": "ita",
        "ita": "ita",
        "italian": "ita",
        "ar": "ara",
        "ara": "ara",
        "arabic": "ara",
        "vi": "vie",
        "vie": "vie",
        "vietnamese": "vie",
        "th": "tha",
        "tha": "tha",
        "thai": "tha",
    }

    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not OCREngine._initialized:
            self._setup_tesseract()
            OCREngine._initialized = True

    def _setup_tesseract(self):
        """Tesseract 경로 설정"""
        import pytesseract

        # Windows에서 Tesseract 경로 자동 탐색
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

    def set_language(self, lang: str):
        """OCR 언어 변경"""
        lang_lower = lang.lower()
        OCREngine._current_lang = self.LANG_MAP.get(lang_lower, "eng")

    def extract_text(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        """
        이미지에서 텍스트 추출

        Args:
            image: numpy 배열 형태의 이미지 (RGB)
            confidence_threshold: 신뢰도 임계값 (0-100 스케일을 0-1로 변환)

        Returns:
            OCRResult: OCR 결과
        """
        import pytesseract

        # 이미지 크기
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        # PIL 이미지로 변환
        pil_image = Image.fromarray(image)

        # OCR 실행 (상세 정보 포함)
        # confidence_threshold를 0-100 스케일로 변환
        conf_threshold = confidence_threshold * 100

        try:
            data = pytesseract.image_to_data(
                pil_image,
                lang=OCREngine._current_lang,
                output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR 실패: {e}\n\nTesseract가 설치되어 있는지 확인하세요.")

        regions = []
        texts = []

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])

            # 빈 텍스트나 낮은 신뢰도 스킵
            if not text or conf < conf_threshold:
                continue

            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]

            bbox = (x, y, x + w, y + h)

            region = TextRegion(
                text=text,
                bbox=bbox,
                confidence=conf / 100.0,  # 0-1 스케일로 변환
                polygon=None
            )
            regions.append(region)
            texts.append(text)

        # 인접한 단어들을 줄 단위로 병합
        merged_regions = self._merge_words_to_lines(regions)

        return OCRResult(
            regions=merged_regions,
            full_text="\n".join([r.text for r in merged_regions]),
            image_size=(width, height)
        )

    def _merge_words_to_lines(
        self,
        regions: List[TextRegion],
        y_threshold: int = 10,
        x_gap_threshold: int = 30
    ) -> List[TextRegion]:
        """
        단어들을 줄 단위로 병합

        Args:
            regions: 텍스트 영역 목록
            y_threshold: Y축 병합 임계값
            x_gap_threshold: X축 간격 임계값

        Returns:
            병합된 텍스트 영역 목록
        """
        if not regions:
            return []

        # Y 좌표로 정렬 후 X 좌표로 정렬
        sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

        lines = []
        current_line = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            last = current_line[-1]

            # 같은 줄인지 확인 (Y 좌표 비교)
            same_line = abs(region.bbox[1] - last.bbox[1]) < y_threshold

            # X축으로 너무 멀지 않은지 확인
            x_close = region.bbox[0] - last.bbox[2] < x_gap_threshold

            if same_line and x_close:
                current_line.append(region)
            else:
                lines.append(current_line)
                current_line = [region]

        lines.append(current_line)

        # 각 줄을 하나의 TextRegion으로 병합
        merged = []
        for line in lines:
            if not line:
                continue

            text = " ".join([r.text for r in line])
            x1 = min(r.bbox[0] for r in line)
            y1 = min(r.bbox[1] for r in line)
            x2 = max(r.bbox[2] for r in line)
            y2 = max(r.bbox[3] for r in line)
            confidence = min(r.confidence for r in line)

            merged.append(TextRegion(
                text=text,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                polygon=None
            ))

        return merged

    def extract_text_from_file(
        self,
        image_path: str,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        """
        이미지 파일에서 텍스트 추출

        Args:
            image_path: 이미지 파일 경로
            confidence_threshold: 신뢰도 임계값

        Returns:
            OCRResult: OCR 결과
        """
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))
        return self.extract_text(image_np, confidence_threshold)

    def extract_text_from_pil(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        """
        PIL 이미지에서 텍스트 추출

        Args:
            image: PIL Image 객체
            confidence_threshold: 신뢰도 임계값

        Returns:
            OCRResult: OCR 결과
        """
        image_np = np.array(image.convert("RGB"))
        return self.extract_text(image_np, confidence_threshold)

    @staticmethod
    def merge_nearby_regions(
        regions: List[TextRegion],
        x_threshold: int = 20,
        y_threshold: int = 10
    ) -> List[TextRegion]:
        """
        인접한 텍스트 영역 병합

        Args:
            regions: 텍스트 영역 목록
            x_threshold: X축 병합 임계값
            y_threshold: Y축 병합 임계값

        Returns:
            병합된 텍스트 영역 목록
        """
        if not regions:
            return []

        # Y 좌표로 정렬
        sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

        merged = []
        current = sorted_regions[0]

        for region in sorted_regions[1:]:
            # 같은 줄인지 확인
            y_overlap = (
                abs(current.bbox[1] - region.bbox[1]) < y_threshold or
                abs(current.bbox[3] - region.bbox[3]) < y_threshold
            )

            # X축으로 가까운지 확인
            x_close = region.bbox[0] - current.bbox[2] < x_threshold

            if y_overlap and x_close:
                # 병합
                new_bbox = (
                    min(current.bbox[0], region.bbox[0]),
                    min(current.bbox[1], region.bbox[1]),
                    max(current.bbox[2], region.bbox[2]),
                    max(current.bbox[3], region.bbox[3])
                )
                current = TextRegion(
                    text=current.text + " " + region.text,
                    bbox=new_bbox,
                    confidence=min(current.confidence, region.confidence),
                    polygon=None
                )
            else:
                merged.append(current)
                current = region

        merged.append(current)
        return merged

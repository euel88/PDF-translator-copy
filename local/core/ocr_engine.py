"""
PaddleOCR 기반 OCR 엔진
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image


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
    """PaddleOCR 기반 OCR 엔진"""

    _instance = None
    _ocr = None

    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if OCREngine._ocr is None:
            self._initialize_ocr()

    def _initialize_ocr(self, lang: str = "en"):
        """PaddleOCR 초기화"""
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 초기화 (GPU 사용 가능시 자동 사용)
            OCREngine._ocr = PaddleOCR(
                use_angle_cls=True,  # 텍스트 방향 감지
                lang=lang,
                show_log=False,
                use_gpu=True,  # GPU 사용 시도
            )
        except Exception as e:
            print(f"PaddleOCR 초기화 실패: {e}")
            # GPU 없으면 CPU로 재시도
            try:
                from paddleocr import PaddleOCR

                OCREngine._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    show_log=False,
                    use_gpu=False,
                )
            except Exception as e2:
                raise RuntimeError(f"PaddleOCR 초기화 실패: {e2}")

    def set_language(self, lang: str):
        """OCR 언어 변경"""
        self._initialize_ocr(lang)

    def extract_text(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> OCRResult:
        """
        이미지에서 텍스트 추출

        Args:
            image: numpy 배열 형태의 이미지 (RGB)
            confidence_threshold: 신뢰도 임계값

        Returns:
            OCRResult: OCR 결과
        """
        if OCREngine._ocr is None:
            raise RuntimeError("OCR 엔진이 초기화되지 않았습니다")

        # 이미지 크기
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        # OCR 실행
        result = OCREngine._ocr.ocr(image, cls=True)

        regions = []
        texts = []

        if result and result[0]:
            for line in result[0]:
                if line is None:
                    continue

                # PaddleOCR 결과 형식: [[좌표], (텍스트, 신뢰도)]
                polygon_points = line[0]
                text, confidence = line[1]

                if confidence < confidence_threshold:
                    continue

                # 다각형 좌표를 바운딩 박스로 변환
                xs = [p[0] for p in polygon_points]
                ys = [p[1] for p in polygon_points]
                bbox = (
                    int(min(xs)),
                    int(min(ys)),
                    int(max(xs)),
                    int(max(ys))
                )

                # 다각형 좌표 저장
                polygon = [(int(p[0]), int(p[1])) for p in polygon_points]

                region = TextRegion(
                    text=text,
                    bbox=bbox,
                    confidence=confidence,
                    polygon=polygon
                )
                regions.append(region)
                texts.append(text)

        return OCRResult(
            regions=regions,
            full_text="\n".join(texts),
            image_size=(width, height)
        )

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

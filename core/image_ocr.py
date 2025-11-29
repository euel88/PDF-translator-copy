"""
Image OCR Module
이미지에서 텍스트를 추출하는 OCR 모듈

RapidOCR을 사용하여 이미지 내 텍스트 감지 및 추출
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """OCR로 감지된 텍스트 영역"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    polygon: List[List[int]]  # 4개의 꼭지점 좌표


@dataclass
class OCRResult:
    """OCR 결과"""
    regions: List[TextRegion]
    image_size: Tuple[int, int]  # (width, height)
    language: str


class ImageOCR:
    """이미지 OCR 처리 클래스"""

    _instance = None
    _ocr_engine = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._ocr_engine is None:
            self._initialize_ocr()

    def _initialize_ocr(self):
        """OCR 엔진 초기화"""
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr_engine = RapidOCR()
            logger.info("RapidOCR 엔진 초기화 완료")
        except ImportError:
            logger.warning("RapidOCR을 사용할 수 없습니다. 설치가 필요합니다.")
            self._ocr_engine = None
        except Exception as e:
            logger.error(f"OCR 엔진 초기화 실패: {e}")
            self._ocr_engine = None

    @property
    def is_available(self) -> bool:
        """OCR 엔진 사용 가능 여부"""
        return self._ocr_engine is not None

    def extract_text(
        self,
        image: np.ndarray,
        lang: str = "auto"
    ) -> OCRResult:
        """
        이미지에서 텍스트 추출

        Args:
            image: numpy 배열 형태의 이미지 (BGR 또는 RGB)
            lang: 언어 코드 (auto, ko, en, zh, ja 등)

        Returns:
            OCRResult: OCR 결과
        """
        if not self.is_available:
            logger.warning("OCR 엔진을 사용할 수 없습니다")
            return OCRResult(regions=[], image_size=(0, 0), language=lang)

        try:
            # 이미지 크기
            if len(image.shape) == 3:
                height, width = image.shape[:2]
            else:
                height, width = image.shape

            # RapidOCR 실행
            result, elapse = self._ocr_engine(image)

            regions = []
            if result:
                for item in result:
                    polygon = item[0]  # 4개의 꼭지점
                    text = item[1]
                    confidence = item[2]

                    # bounding box 계산
                    x_coords = [p[0] for p in polygon]
                    y_coords = [p[1] for p in polygon]
                    bbox = (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    )

                    regions.append(TextRegion(
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                        polygon=[[int(p[0]), int(p[1])] for p in polygon]
                    ))

            # elapse는 리스트 또는 딕셔너리일 수 있음
            if isinstance(elapse, (list, tuple)):
                elapse_str = f"{sum(elapse):.2f}s" if elapse else "N/A"
            elif isinstance(elapse, dict):
                elapse_str = f"{sum(elapse.values()):.2f}s" if elapse else "N/A"
            elif isinstance(elapse, (int, float)):
                elapse_str = f"{elapse:.2f}s"
            else:
                elapse_str = str(elapse)
            logger.info(f"OCR 완료: {len(regions)}개 텍스트 영역 감지 (소요시간: {elapse_str})")

            return OCRResult(
                regions=regions,
                image_size=(width, height),
                language=lang
            )

        except Exception as e:
            logger.error(f"OCR 처리 실패: {e}")
            return OCRResult(regions=[], image_size=(0, 0), language=lang)

    def extract_text_from_file(
        self,
        image_path: str,
        lang: str = "auto"
    ) -> OCRResult:
        """
        이미지 파일에서 텍스트 추출

        Args:
            image_path: 이미지 파일 경로
            lang: 언어 코드

        Returns:
            OCRResult: OCR 결과
        """
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
            return self.extract_text(image, lang)
        except Exception as e:
            logger.error(f"이미지 파일 OCR 실패: {e}")
            return OCRResult(regions=[], image_size=(0, 0), language=lang)

    def merge_nearby_regions(
        self,
        regions: List[TextRegion],
        threshold: int = 20
    ) -> List[TextRegion]:
        """
        인접한 텍스트 영역을 병합

        Args:
            regions: 텍스트 영역 리스트
            threshold: 병합 거리 임계값 (픽셀)

        Returns:
            병합된 텍스트 영역 리스트
        """
        if not regions:
            return []

        # Y 좌표로 정렬
        sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))

        merged = []
        current_group = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            last = current_group[-1]

            # 같은 줄에 있고 가까운 경우 병합
            y_overlap = (
                min(last.bbox[3], region.bbox[3]) -
                max(last.bbox[1], region.bbox[1])
            )
            x_gap = region.bbox[0] - last.bbox[2]

            if y_overlap > 0 and x_gap < threshold:
                current_group.append(region)
            else:
                # 현재 그룹 병합
                merged.append(self._merge_group(current_group))
                current_group = [region]

        # 마지막 그룹 처리
        if current_group:
            merged.append(self._merge_group(current_group))

        return merged

    def _merge_group(self, group: List[TextRegion]) -> TextRegion:
        """텍스트 영역 그룹을 하나로 병합"""
        if len(group) == 1:
            return group[0]

        # 텍스트 결합
        text = " ".join(r.text for r in group)

        # bbox 계산
        x1 = min(r.bbox[0] for r in group)
        y1 = min(r.bbox[1] for r in group)
        x2 = max(r.bbox[2] for r in group)
        y2 = max(r.bbox[3] for r in group)

        # 평균 confidence
        confidence = sum(r.confidence for r in group) / len(group)

        # polygon은 전체 bbox의 4 꼭지점
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        return TextRegion(
            text=text,
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            polygon=polygon
        )


# 싱글톤 인스턴스
_ocr_instance: Optional[ImageOCR] = None


def get_ocr_engine() -> ImageOCR:
    """OCR 엔진 싱글톤 인스턴스 반환"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = ImageOCR()
    return _ocr_instance


def ocr_image(
    image: np.ndarray,
    lang: str = "auto",
    merge_nearby: bool = True
) -> OCRResult:
    """
    이미지에서 텍스트 추출 (편의 함수)

    Args:
        image: numpy 배열 형태의 이미지
        lang: 언어 코드
        merge_nearby: 인접 영역 병합 여부

    Returns:
        OCRResult: OCR 결과
    """
    ocr = get_ocr_engine()
    result = ocr.extract_text(image, lang)

    if merge_nearby and result.regions:
        result.regions = ocr.merge_nearby_regions(result.regions)

    return result

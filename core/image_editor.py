"""
Image Text Editor Module
이미지 내 텍스트를 편집하는 모듈

원본 텍스트를 지우고 번역된 텍스트로 교체
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TextReplacement:
    """텍스트 교체 정보"""
    original_text: str
    translated_text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    polygon: List[List[int]]
    font_size: Optional[int] = None
    font_color: Optional[Tuple[int, int, int]] = None
    background_color: Optional[Tuple[int, int, int]] = None


class ImageTextEditor:
    """이미지 내 텍스트 편집 클래스"""

    def __init__(self):
        self._font_path = None
        self._pil_available = False
        self._cv2_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        """의존성 확인"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            self._pil_available = True
        except ImportError:
            logger.warning("PIL을 사용할 수 없습니다")

        try:
            import cv2
            self._cv2_available = True
        except ImportError:
            logger.warning("OpenCV를 사용할 수 없습니다")

    def _get_font(self, size: int = 20, lang: str = "ko"):
        """적절한 폰트 로드"""
        from PIL import ImageFont

        # 폰트 경로 후보들
        font_paths = [
            # Linux 시스템 폰트
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            # 시스템 폰트 디렉토리
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

        # babeldoc에서 다운로드한 폰트 경로 확인
        try:
            from babeldoc.assets.assets import get_font_and_metadata
            font_name = "GoNotoKurrent-Regular.ttf"
            if lang in ["ko", "ja", "zh", "zh-cn", "zh-tw"]:
                font_name = "SourceHanSerifCN-Regular.ttf"
            cached_font, _ = get_font_and_metadata(font_name)
            font_paths.insert(0, str(cached_font))
        except Exception:
            pass

        # 사용 가능한 폰트 찾기
        for font_path in font_paths:
            try:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, size)
            except Exception:
                continue

        # 기본 폰트 사용
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _estimate_font_size(
        self,
        bbox: Tuple[int, int, int, int],
        text: str,
        max_iterations: int = 10
    ) -> int:
        """
        bbox에 맞는 폰트 크기 추정

        Args:
            bbox: 텍스트 영역 (x1, y1, x2, y2)
            text: 삽입할 텍스트
            max_iterations: 최대 반복 횟수

        Returns:
            적절한 폰트 크기
        """
        from PIL import ImageFont

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # 초기 추정 (높이 기반)
        font_size = max(int(height * 0.8), 10)

        for _ in range(max_iterations):
            font = self._get_font(font_size)
            if font is None:
                return font_size

            try:
                # 텍스트 크기 측정
                bbox_text = font.getbbox(text)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]

                # 크기 조정
                if text_width > width or text_height > height:
                    font_size = int(font_size * 0.9)
                elif text_width < width * 0.5 and text_height < height * 0.5:
                    font_size = int(font_size * 1.1)
                else:
                    break
            except Exception:
                break

        return max(font_size, 8)

    def _get_dominant_color(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        is_background: bool = True
    ) -> Tuple[int, int, int]:
        """
        영역의 주요 색상 추출

        Args:
            image: numpy 배열 이미지
            bbox: 영역 (x1, y1, x2, y2)
            is_background: 배경색 추출 여부 (False면 텍스트 색상)

        Returns:
            RGB 색상 튜플
        """
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2]

        if region.size == 0:
            return (255, 255, 255) if is_background else (0, 0, 0)

        # 평균 색상 계산
        if len(region.shape) == 3:
            avg_color = np.mean(region, axis=(0, 1))
            if region.shape[2] == 3:
                # BGR to RGB
                return (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
            return tuple(int(c) for c in avg_color[:3])
        else:
            avg = int(np.mean(region))
            return (avg, avg, avg)

    def _inpaint_region(
        self,
        image: np.ndarray,
        polygon: List[List[int]]
    ) -> np.ndarray:
        """
        영역을 인페인팅하여 텍스트 제거

        Args:
            image: numpy 배열 이미지
            polygon: 영역 다각형 좌표

        Returns:
            인페인팅된 이미지
        """
        import cv2

        # 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # 마스크 확장 (텍스트 경계 포함)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 인페인팅
        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return result

    def _fill_region_with_color(
        self,
        image: np.ndarray,
        polygon: List[List[int]],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        영역을 단색으로 채우기

        Args:
            image: numpy 배열 이미지
            polygon: 영역 다각형 좌표
            color: RGB 색상

        Returns:
            편집된 이미지
        """
        import cv2

        result = image.copy()
        pts = np.array(polygon, dtype=np.int32)
        # BGR 색상으로 변환
        bgr_color = (color[2], color[1], color[0])
        cv2.fillPoly(result, [pts], bgr_color)

        return result

    def replace_text(
        self,
        image: np.ndarray,
        replacements: List[TextReplacement],
        lang: str = "ko",
        use_inpaint: bool = True
    ) -> np.ndarray:
        """
        이미지 내 텍스트 교체

        Args:
            image: numpy 배열 이미지 (BGR)
            replacements: 텍스트 교체 정보 리스트
            lang: 대상 언어
            use_inpaint: 인페인팅 사용 여부

        Returns:
            편집된 이미지
        """
        if not self._pil_available or not self._cv2_available:
            logger.error("PIL 또는 OpenCV를 사용할 수 없습니다")
            return image

        import cv2
        from PIL import Image, ImageDraw

        result = image.copy()

        for replacement in replacements:
            try:
                # 1. 원본 텍스트 영역 처리
                if use_inpaint:
                    # 인페인팅으로 원본 텍스트 제거
                    result = self._inpaint_region(result, replacement.polygon)
                else:
                    # 배경색으로 채우기
                    bg_color = replacement.background_color
                    if bg_color is None:
                        bg_color = self._get_dominant_color(
                            image, replacement.bbox, is_background=True
                        )
                    result = self._fill_region_with_color(
                        result, replacement.polygon, bg_color
                    )

                # 2. 번역된 텍스트 삽입
                # BGR to RGB for PIL
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(result_rgb)
                draw = ImageDraw.Draw(pil_image)

                # 폰트 크기 결정
                font_size = replacement.font_size
                if font_size is None:
                    font_size = self._estimate_font_size(
                        replacement.bbox,
                        replacement.translated_text
                    )

                font = self._get_font(font_size, lang)

                # 텍스트 색상 결정
                text_color = replacement.font_color
                if text_color is None:
                    # 배경과 대비되는 색상 사용
                    bg = self._get_dominant_color(result, replacement.bbox, True)
                    brightness = (bg[0] * 299 + bg[1] * 587 + bg[2] * 114) / 1000
                    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

                # 텍스트 위치 계산 (중앙 정렬)
                x1, y1, x2, y2 = replacement.bbox
                if font:
                    text_bbox = font.getbbox(replacement.translated_text)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                else:
                    text_width = len(replacement.translated_text) * font_size // 2
                    text_height = font_size

                x = x1 + (x2 - x1 - text_width) // 2
                y = y1 + (y2 - y1 - text_height) // 2

                # 텍스트 그리기
                draw.text((x, y), replacement.translated_text, fill=text_color, font=font)

                # RGB to BGR
                result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            except Exception as e:
                logger.error(f"텍스트 교체 실패: {e}")
                continue

        return result

    def replace_text_simple(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        new_text: str,
        lang: str = "ko"
    ) -> np.ndarray:
        """
        단일 텍스트 영역 교체 (간편 함수)

        Args:
            image: numpy 배열 이미지
            bbox: 텍스트 영역 (x1, y1, x2, y2)
            new_text: 새 텍스트
            lang: 대상 언어

        Returns:
            편집된 이미지
        """
        x1, y1, x2, y2 = bbox
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        replacement = TextReplacement(
            original_text="",
            translated_text=new_text,
            bbox=bbox,
            polygon=polygon
        )

        return self.replace_text(image, [replacement], lang)


def replace_image_text(
    image: np.ndarray,
    replacements: List[Dict[str, Any]],
    lang: str = "ko",
    use_inpaint: bool = True
) -> np.ndarray:
    """
    이미지 내 텍스트 교체 (편의 함수)

    Args:
        image: numpy 배열 이미지
        replacements: 교체 정보 리스트
            [{"original": str, "translated": str, "bbox": tuple, "polygon": list}, ...]
        lang: 대상 언어
        use_inpaint: 인페인팅 사용 여부

    Returns:
        편집된 이미지
    """
    editor = ImageTextEditor()

    replacement_objects = []
    for r in replacements:
        replacement_objects.append(TextReplacement(
            original_text=r.get("original", ""),
            translated_text=r.get("translated", ""),
            bbox=r.get("bbox", (0, 0, 0, 0)),
            polygon=r.get("polygon", []),
            font_size=r.get("font_size"),
            font_color=r.get("font_color"),
            background_color=r.get("background_color")
        ))

    return editor.replace_text(image, replacement_objects, lang, use_inpaint)

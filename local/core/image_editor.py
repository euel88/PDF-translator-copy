"""
이미지 편집 모듈 - 텍스트 영역 지우기 및 번역 텍스트 삽입
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from pathlib import Path
import cv2

from .ocr_engine import TextRegion


class ImageEditor:
    """이미지 편집 클래스"""

    # 기본 폰트 경로 (시스템별)
    DEFAULT_FONTS = [
        # Windows
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def __init__(self, font_path: Optional[str] = None):
        """
        초기화

        Args:
            font_path: 사용할 폰트 경로 (None이면 자동 탐색)
        """
        self.font_path = font_path or self._find_font()

    def _find_font(self) -> Optional[str]:
        """시스템에서 사용 가능한 폰트 찾기"""
        for font_path in self.DEFAULT_FONTS:
            if Path(font_path).exists():
                return font_path
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """지정된 크기의 폰트 가져오기"""
        if self.font_path and Path(self.font_path).exists():
            return ImageFont.truetype(self.font_path, size)
        else:
            # 기본 폰트 사용
            return ImageFont.load_default()

    def calculate_font_size(
        self,
        text: str,
        bbox: Tuple[int, int, int, int],
        max_font_size: int = 100,
        min_font_size: int = 8,
        padding: int = 4
    ) -> int:
        """
        텍스트가 박스에 맞는 폰트 크기 계산

        Args:
            text: 텍스트
            bbox: (x1, y1, x2, y2)
            max_font_size: 최대 폰트 크기
            min_font_size: 최소 폰트 크기
            padding: 패딩

        Returns:
            적절한 폰트 크기
        """
        box_width = bbox[2] - bbox[0] - padding * 2
        box_height = bbox[3] - bbox[1] - padding * 2

        for size in range(max_font_size, min_font_size - 1, -1):
            font = self._get_font(size)

            # 텍스트 크기 측정
            dummy_img = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if text_width <= box_width and text_height <= box_height:
                return size

        return min_font_size

    def erase_text_region(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        method: str = "inpaint"
    ) -> Image.Image:
        """
        텍스트 영역 지우기

        Args:
            image: PIL 이미지
            bbox: (x1, y1, x2, y2)
            method: 지우기 방법 ("inpaint", "fill_white", "fill_avg")

        Returns:
            수정된 이미지
        """
        img_array = np.array(image)
        x1, y1, x2, y2 = bbox

        # 경계 확인
        h, w = img_array.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if method == "inpaint":
            # OpenCV 인페인팅 사용
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # BGR 변환 (OpenCV는 BGR 사용)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        elif method == "fill_white":
            # 흰색으로 채우기
            result = img_array.copy()
            result[y1:y2, x1:x2] = 255

        elif method == "fill_avg":
            # 주변 색상 평균으로 채우기
            result = img_array.copy()

            # 주변 영역 색상 평균 계산
            border = 5
            top = img_array[max(0, y1-border):y1, x1:x2] if y1 > 0 else np.array([])
            bottom = img_array[y2:min(h, y2+border), x1:x2] if y2 < h else np.array([])
            left = img_array[y1:y2, max(0, x1-border):x1] if x1 > 0 else np.array([])
            right = img_array[y1:y2, x2:min(w, x2+border)] if x2 < w else np.array([])

            all_border = []
            for arr in [top, bottom, left, right]:
                if arr.size > 0:
                    all_border.append(arr.reshape(-1, 3))

            if all_border:
                avg_color = np.mean(np.vstack(all_border), axis=0).astype(np.uint8)
            else:
                avg_color = np.array([255, 255, 255], dtype=np.uint8)

            result[y1:y2, x1:x2] = avg_color

        else:
            result = img_array

        return Image.fromarray(result)

    def draw_text(
        self,
        image: Image.Image,
        text: str,
        bbox: Tuple[int, int, int, int],
        font_size: Optional[int] = None,
        text_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Optional[Tuple[int, int, int]] = None,
        align: str = "center"
    ) -> Image.Image:
        """
        이미지에 텍스트 그리기

        Args:
            image: PIL 이미지
            text: 텍스트
            bbox: (x1, y1, x2, y2)
            font_size: 폰트 크기 (None이면 자동 계산)
            text_color: 텍스트 색상
            bg_color: 배경 색상 (None이면 투명)
            align: 정렬 ("left", "center", "right")

        Returns:
            수정된 이미지
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)

        x1, y1, x2, y2 = bbox

        # 배경 채우기
        if bg_color:
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)

        # 폰트 크기 계산
        if font_size is None:
            font_size = self.calculate_font_size(text, bbox)

        font = self._get_font(font_size)

        # 텍스트 크기 측정
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 위치 계산
        box_width = x2 - x1
        box_height = y2 - y1

        if align == "left":
            x = x1 + 2
        elif align == "right":
            x = x2 - text_width - 2
        else:  # center
            x = x1 + (box_width - text_width) // 2

        y = y1 + (box_height - text_height) // 2

        # 텍스트 그리기
        draw.text((x, y), text, font=font, fill=text_color)

        return img

    def replace_text(
        self,
        image: Image.Image,
        region: TextRegion,
        new_text: str,
        erase_method: str = "inpaint",
        text_color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Optional[Tuple[int, int, int]] = None
    ) -> Image.Image:
        """
        텍스트 영역을 새 텍스트로 교체

        Args:
            image: PIL 이미지
            region: 텍스트 영역
            new_text: 새 텍스트
            erase_method: 지우기 방법
            text_color: 텍스트 색상
            bg_color: 배경 색상

        Returns:
            수정된 이미지
        """
        # 1. 원본 텍스트 지우기
        img = self.erase_text_region(image, region.bbox, erase_method)

        # 2. 새 텍스트 그리기
        img = self.draw_text(
            img,
            new_text,
            region.bbox,
            text_color=text_color,
            bg_color=bg_color
        )

        return img

    def replace_all_text(
        self,
        image: Image.Image,
        regions: List[TextRegion],
        translations: List[str],
        erase_method: str = "inpaint",
        text_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Image.Image:
        """
        모든 텍스트 영역을 번역으로 교체

        Args:
            image: PIL 이미지
            regions: 텍스트 영역 목록
            translations: 번역 텍스트 목록
            erase_method: 지우기 방법
            text_color: 텍스트 색상

        Returns:
            수정된 이미지
        """
        if len(regions) != len(translations):
            raise ValueError("영역과 번역 개수가 일치하지 않습니다")

        img = image.copy()

        # 모든 영역 지우기
        for region in regions:
            img = self.erase_text_region(img, region.bbox, erase_method)

        # 모든 번역 텍스트 그리기
        for region, translation in zip(regions, translations):
            if translation.strip():
                img = self.draw_text(
                    img,
                    translation,
                    region.bbox,
                    text_color=text_color
                )

        return img

    def detect_text_color(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """
        텍스트 영역의 주요 색상 감지

        Args:
            image: PIL 이미지
            bbox: (x1, y1, x2, y2)

        Returns:
            (R, G, B) 색상
        """
        img_array = np.array(image)
        x1, y1, x2, y2 = bbox

        region = img_array[y1:y2, x1:x2]

        if region.size == 0:
            return (0, 0, 0)

        # 가장 어두운 색상 (텍스트일 가능성 높음)
        gray = np.mean(region, axis=2)
        darkest_idx = np.unravel_index(np.argmin(gray), gray.shape)

        color = region[darkest_idx[0], darkest_idx[1]]
        return tuple(color.tolist())

    def detect_background_color(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """
        텍스트 영역의 배경 색상 감지

        Args:
            image: PIL 이미지
            bbox: (x1, y1, x2, y2)

        Returns:
            (R, G, B) 색상
        """
        img_array = np.array(image)
        x1, y1, x2, y2 = bbox
        h, w = img_array.shape[:2]

        # 주변 영역에서 배경색 추출
        border = 10
        samples = []

        # 상단
        if y1 > border:
            samples.append(img_array[y1-border:y1, x1:x2])
        # 하단
        if y2 + border < h:
            samples.append(img_array[y2:y2+border, x1:x2])

        if not samples:
            # 영역 내에서 가장 밝은 색상
            region = img_array[y1:y2, x1:x2]
            gray = np.mean(region, axis=2)
            brightest_idx = np.unravel_index(np.argmax(gray), gray.shape)
            color = region[brightest_idx[0], brightest_idx[1]]
            return tuple(color.tolist())

        all_samples = np.vstack([s.reshape(-1, 3) for s in samples])
        avg_color = np.mean(all_samples, axis=0).astype(np.uint8)
        return tuple(avg_color.tolist())

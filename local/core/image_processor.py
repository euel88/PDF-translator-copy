"""
이미지 처리 모듈 - 텍스트 지우기 및 삽입
"""
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from .ocr import TextBox


class ImageProcessor:
    """이미지 처리 클래스"""

    # 시스템별 기본 폰트 경로
    DEFAULT_FONTS = [
        # Windows
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\meiryo.ttc",
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_font()

    def _find_font(self) -> Optional[str]:
        """시스템에서 사용 가능한 폰트 찾기"""
        for p in self.DEFAULT_FONTS:
            if Path(p).exists():
                return p
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """폰트 객체 반환"""
        if self.font_path and Path(self.font_path).exists():
            return ImageFont.truetype(self.font_path, size)
        return ImageFont.load_default()

    def erase_region(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        method: str = "inpaint"
    ) -> Image.Image:
        """
        이미지 영역 지우기

        Args:
            image: PIL 이미지
            bbox: (x1, y1, x2, y2)
            method: "inpaint" | "white" | "blur"
        """
        arr = np.array(image)
        x1, y1, x2, y2 = bbox
        h, w = arr.shape[:2]

        # 경계 검사
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if method == "inpaint":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            result = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        elif method == "white":
            result = arr.copy()
            result[y1:y2, x1:x2] = 255

        elif method == "blur":
            result = arr.copy()
            region = result[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(region, (21, 21), 0)
            result[y1:y2, x1:x2] = blurred

        else:
            result = arr

        return Image.fromarray(result)

    def draw_text(
        self,
        image: Image.Image,
        text: str,
        bbox: Tuple[int, int, int, int],
        font_size: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        bg_color: Optional[Tuple[int, int, int]] = None,
    ) -> Image.Image:
        """
        이미지에 텍스트 그리기

        Args:
            image: PIL 이미지
            text: 텍스트
            bbox: (x1, y1, x2, y2)
            font_size: 폰트 크기 (None이면 자동)
            color: 텍스트 색상
            bg_color: 배경 색상 (None이면 투명)
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox

        # 배경 채우기
        if bg_color:
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)

        # 폰트 크기 계산
        if font_size is None:
            font_size = self._fit_font_size(text, x2 - x1, y2 - y1)

        font = self._get_font(font_size)

        # 텍스트 크기
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        # 중앙 정렬
        x = x1 + (x2 - x1 - tw) // 2
        y = y1 + (y2 - y1 - th) // 2

        draw.text((x, y), text, font=font, fill=color)
        return img

    def _fit_font_size(
        self,
        text: str,
        max_width: int,
        max_height: int,
        min_size: int = 8,
        max_size: int = 100,
    ) -> int:
        """박스에 맞는 폰트 크기 계산"""
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)

        for size in range(max_size, min_size - 1, -1):
            font = self._get_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w <= max_width - 4 and h <= max_height - 4:
                return size

        return min_size

    def replace_text(
        self,
        image: Image.Image,
        box: TextBox,
        new_text: str,
        erase_method: str = "inpaint",
        text_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """텍스트 교체"""
        # 원본 텍스트 지우기
        img = self.erase_region(image, box.bbox, erase_method)
        # 새 텍스트 그리기
        img = self.draw_text(img, new_text, box.bbox, color=text_color)
        return img

    def replace_all_text(
        self,
        image: Image.Image,
        boxes: List[TextBox],
        translations: List[str],
        erase_method: str = "inpaint",
    ) -> Image.Image:
        """모든 텍스트 교체"""
        if len(boxes) != len(translations):
            raise ValueError("boxes와 translations 길이가 다릅니다")

        img = image.copy()

        # 모든 영역 지우기
        for box in boxes:
            img = self.erase_region(img, box.bbox, erase_method)

        # 모든 번역 텍스트 그리기
        for box, trans in zip(boxes, translations):
            if trans.strip():
                img = self.draw_text(img, trans, box.bbox)

        return img

    def detect_text_color(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """텍스트 영역의 주요 색상 감지"""
        arr = np.array(image)
        x1, y1, x2, y2 = bbox
        region = arr[y1:y2, x1:x2]

        if region.size == 0:
            return (0, 0, 0)

        # 가장 어두운 색 (텍스트일 가능성)
        gray = np.mean(region, axis=2)
        idx = np.unravel_index(np.argmin(gray), gray.shape)
        color = region[idx[0], idx[1]]

        return tuple(int(c) for c in color)

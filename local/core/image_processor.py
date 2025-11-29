"""
이미지 처리 모듈 - 텍스트 지우기 및 삽입 (개선된 버전)
"""
from pathlib import Path
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from core.pdf_handler import TextBlock


class ImageProcessor:
    """이미지 처리 클래스"""

    # 시스템별 기본 폰트 경로
    DEFAULT_FONTS = [
        # Windows - 한국어/일본어/중국어 지원 폰트
        r"C:\Windows\Fonts\malgun.ttf",      # 맑은 고딕
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\meiryo.ttc",       # 일본어
        r"C:\Windows\Fonts\msyh.ttc",         # 중국어
        r"C:\Windows\Fonts\arial.ttf",
        # macOS
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path or self._find_font()
        self._font_cache = {}

    def _find_font(self) -> Optional[str]:
        """시스템에서 사용 가능한 폰트 찾기"""
        for p in self.DEFAULT_FONTS:
            if Path(p).exists():
                return p
        return None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """폰트 객체 반환 (캐시 사용)"""
        size = max(8, int(size))  # 최소 8pt
        cache_key = (self.font_path, size)

        if cache_key not in self._font_cache:
            if self.font_path and Path(self.font_path).exists():
                self._font_cache[cache_key] = ImageFont.truetype(self.font_path, size)
            else:
                self._font_cache[cache_key] = ImageFont.load_default()

        return self._font_cache[cache_key]

    def erase_region(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        method: str = "white",
        padding: int = 2
    ) -> Image.Image:
        """
        이미지 영역 지우기

        Args:
            image: PIL 이미지
            bbox: (x0, y0, x1, y1)
            method: "inpaint" | "white" | "blur"
            padding: 여백
        """
        arr = np.array(image)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = arr.shape[:2]

        # 패딩 적용
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        if method == "inpaint":
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            result = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        elif method == "white":
            result = arr.copy()
            # 주변 배경색 감지
            bg_color = self._detect_background_color(arr, (x1, y1, x2, y2))
            result[y1:y2, x1:x2] = bg_color

        elif method == "blur":
            result = arr.copy()
            region = result[y1:y2, x1:x2]
            if region.size > 0:
                blurred = cv2.GaussianBlur(region, (21, 21), 0)
                result[y1:y2, x1:x2] = blurred
        else:
            result = arr

        return Image.fromarray(result)

    def _detect_background_color(
        self,
        arr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        sample_width: int = 5
    ) -> Tuple[int, int, int]:
        """텍스트 영역 주변의 배경색 감지"""
        x1, y1, x2, y2 = bbox
        h, w = arr.shape[:2]

        samples = []

        # 상단
        if y1 > sample_width:
            samples.append(arr[y1-sample_width:y1, x1:x2])
        # 하단
        if y2 < h - sample_width:
            samples.append(arr[y2:y2+sample_width, x1:x2])
        # 좌측
        if x1 > sample_width:
            samples.append(arr[y1:y2, x1-sample_width:x1])
        # 우측
        if x2 < w - sample_width:
            samples.append(arr[y1:y2, x2:x2+sample_width])

        if not samples:
            return (255, 255, 255)

        # 모든 샘플 합치기
        all_pixels = np.concatenate([s.reshape(-1, 3) for s in samples if s.size > 0])
        if all_pixels.size == 0:
            return (255, 255, 255)

        # 중앙값 사용 (이상치에 강건)
        median_color = np.median(all_pixels, axis=0).astype(np.uint8)
        return tuple(median_color)

    def draw_text(
        self,
        image: Image.Image,
        text: str,
        bbox: Tuple[float, float, float, float],
        font_size: Optional[float] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        align: str = "left",
    ) -> Image.Image:
        """
        이미지에 텍스트 그리기 (정확한 위치)

        Args:
            image: PIL 이미지
            text: 텍스트
            bbox: (x0, y0, x1, y1)
            font_size: 폰트 크기 (None이면 bbox 높이 기반 자동)
            color: 텍스트 색상
            align: "left" | "center" | "right"
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = [int(v) for v in bbox]

        box_width = x1 - x0
        box_height = y1 - y0

        # 폰트 크기 결정
        if font_size is None:
            # bbox 높이의 80%를 폰트 크기로
            font_size = max(8, int(box_height * 0.8))

        # 폰트 크기 조정 (박스에 맞게)
        font_size = self._adjust_font_size(draw, text, box_width, box_height, font_size)
        font = self._get_font(font_size)

        # 텍스트 크기 계산
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # X 위치 (정렬)
        if align == "center":
            x = x0 + (box_width - text_width) // 2
        elif align == "right":
            x = x1 - text_width
        else:  # left
            x = x0

        # Y 위치 (수직 중앙)
        y = y0 + (box_height - text_height) // 2

        draw.text((x, y), text, font=font, fill=color)
        return img

    def _adjust_font_size(
        self,
        draw: ImageDraw.Draw,
        text: str,
        max_width: int,
        max_height: int,
        initial_size: int,
        min_size: int = 8,
    ) -> int:
        """박스에 맞게 폰트 크기 조정"""
        size = initial_size

        while size > min_size:
            font = self._get_font(size)
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w <= max_width and h <= max_height:
                return size

            size -= 1

        return min_size

    def replace_text_block(
        self,
        image: Image.Image,
        block: TextBlock,
        new_text: str,
        erase_method: str = "white",
    ) -> Image.Image:
        """TextBlock의 텍스트 교체"""
        # 원본 텍스트 지우기
        bbox = (block.x0, block.y0, block.x1, block.y1)
        img = self.erase_region(image, bbox, erase_method)

        # 새 텍스트 그리기 (원본 폰트 크기와 색상 사용)
        img = self.draw_text(
            img,
            new_text,
            bbox,
            font_size=block.font_size,
            color=block.color,
            align="left",
        )
        return img

    def replace_all_blocks(
        self,
        image: Image.Image,
        blocks: List[TextBlock],
        translations: List[str],
        erase_method: str = "white",
    ) -> Image.Image:
        """모든 TextBlock 교체"""
        if len(blocks) != len(translations):
            raise ValueError("blocks와 translations 길이가 다릅니다")

        img = image.copy()

        # 먼저 모든 영역 지우기
        for block in blocks:
            bbox = (block.x0, block.y0, block.x1, block.y1)
            img = self.erase_region(img, bbox, erase_method)

        # 모든 번역 텍스트 그리기
        for block, trans in zip(blocks, translations):
            if trans.strip():
                bbox = (block.x0, block.y0, block.x1, block.y1)
                img = self.draw_text(
                    img,
                    trans,
                    bbox,
                    font_size=block.font_size,
                    color=block.color,
                    align="left",
                )

        return img

    # OCR TextBox 호환 메서드 (스캔 PDF용)
    def replace_text(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
        new_text: str,
        font_size: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        erase_method: str = "white",
    ) -> Image.Image:
        """단일 영역 텍스트 교체"""
        img = self.erase_region(image, bbox, erase_method)
        img = self.draw_text(img, new_text, bbox, font_size, color)
        return img

    def replace_all_text(
        self,
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        translations: List[str],
        font_sizes: Optional[List[int]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        erase_method: str = "white",
    ) -> Image.Image:
        """모든 영역 텍스트 교체"""
        if len(bboxes) != len(translations):
            raise ValueError("bboxes와 translations 길이가 다릅니다")

        img = image.copy()

        # 먼저 모든 영역 지우기
        for bbox in bboxes:
            img = self.erase_region(img, bbox, erase_method)

        # 모든 번역 텍스트 그리기
        for i, (bbox, trans) in enumerate(zip(bboxes, translations)):
            if trans.strip():
                font_size = font_sizes[i] if font_sizes else None
                color = colors[i] if colors else (0, 0, 0)
                img = self.draw_text(img, trans, bbox, font_size, color)

        return img

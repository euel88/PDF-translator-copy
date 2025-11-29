"""
Image Text Editor Module
이미지 내 텍스트를 편집하는 모듈

원본 텍스트를 지우고 번역된 텍스트로 교체

개선사항:
- 텍스트 자동 줄바꿈
- 동적 폰트 크기 조정
- 정확한 텍스트 위치 계산
- 다국어 폰트 지원 강화
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import textwrap

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

        # 캐시된 폰트 경로 사용
        if hasattr(self, '_cached_font_path') and self._cached_font_path:
            try:
                return ImageFont.truetype(self._cached_font_path, size)
            except Exception:
                pass

        # 폰트 경로 후보들 (CJK 폰트 우선)
        font_paths = []

        # babeldoc에서 다운로드한 폰트 경로 확인 (우선순위 높음)
        try:
            from babeldoc.assets.assets import get_font_and_metadata
            # GoNotoKurrent는 다국어 지원이 좋음
            font_name = "GoNotoKurrent-Regular.ttf"
            cached_font, _ = get_font_and_metadata(font_name)
            font_paths.append(str(cached_font))
        except Exception:
            pass

        # 홈 디렉토리의 babeldoc 캐시 확인
        try:
            import os
            home = os.path.expanduser("~")
            babeldoc_cache = os.path.join(home, ".cache", "babeldoc", "fonts")
            if os.path.exists(babeldoc_cache):
                for font_file in os.listdir(babeldoc_cache):
                    if font_file.endswith(('.ttf', '.otf', '.ttc')):
                        font_paths.append(os.path.join(babeldoc_cache, font_file))
        except Exception:
            pass

        # Linux 시스템 폰트
        font_paths.extend([
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ])

        # 사용 가능한 폰트 찾기
        for font_path in font_paths:
            try:
                if Path(font_path).exists():
                    self._cached_font_path = font_path
                    # 처음 로드 시에만 로깅
                    if not hasattr(self, '_font_logged'):
                        logger.info(f"폰트 로드 성공: {font_path}")
                        self._font_logged = True
                    return ImageFont.truetype(font_path, size)
            except Exception as e:
                logger.debug(f"폰트 로드 실패 ({font_path}): {e}")
                continue

        # 기본 폰트 사용
        logger.warning("적절한 폰트를 찾을 수 없어 기본 폰트 사용 (한글 지원 불가)")
        try:
            return ImageFont.load_default()
        except Exception:
            logger.error("기본 폰트도 로드 실패")
            return None

    def _estimate_font_size(
        self,
        bbox: Tuple[int, int, int, int],
        text: str,
        max_iterations: int = 20,
        lang: str = "en"
    ) -> int:
        """
        bbox에 맞는 폰트 크기 추정 (개선된 알고리즘)

        Args:
            bbox: 텍스트 영역 (x1, y1, x2, y2)
            text: 삽입할 텍스트
            max_iterations: 최대 반복 횟수
            lang: 대상 언어

        Returns:
            적절한 폰트 크기
        """
        from PIL import ImageFont

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width <= 0 or height <= 0:
            return 12

        # 초기 추정: 높이 기반이지만 텍스트 길이도 고려
        # 한 줄에 들어갈 수 있는 대략적인 문자 수 추정
        avg_char_width_ratio = 0.6  # 평균 문자 너비 비율
        estimated_chars_per_line = max(1, int(width / (height * avg_char_width_ratio)))

        # 텍스트가 한 줄에 들어갈 수 있으면 높이 기반
        if len(text) <= estimated_chars_per_line:
            font_size = max(int(height * 0.85), 8)
        else:
            # 여러 줄이 필요하면 더 작은 폰트 시작
            lines_needed = (len(text) / estimated_chars_per_line) + 1
            font_size = max(int(height / lines_needed * 0.9), 8)

        min_font_size = 8
        max_font_size = max(int(height * 0.95), 10)

        best_font_size = min_font_size

        for _ in range(max_iterations):
            font = self._get_font(font_size, lang)
            if font is None:
                return max(font_size, min_font_size)

            try:
                # 텍스트를 줄바꿈하여 실제 필요한 크기 계산
                wrapped_text = self._wrap_text(text, font, width)
                lines = wrapped_text.split('\n')

                # 전체 텍스트 높이 계산
                total_height = 0
                max_line_width = 0
                line_height = font_size * 1.2  # 줄 간격

                for line in lines:
                    if line.strip():
                        text_bbox = font.getbbox(line)
                        line_width = text_bbox[2] - text_bbox[0]
                        max_line_width = max(max_line_width, line_width)
                    total_height += line_height

                # 크기 조정
                fits_width = max_line_width <= width * 1.05  # 5% 여유
                fits_height = total_height <= height * 1.05

                if fits_width and fits_height:
                    best_font_size = font_size
                    # 더 큰 크기 시도
                    if font_size < max_font_size:
                        font_size = min(int(font_size * 1.1), max_font_size)
                    else:
                        break
                else:
                    # 크기 줄이기
                    font_size = int(font_size * 0.9)
                    if font_size < min_font_size:
                        break

            except Exception as e:
                logger.debug(f"폰트 크기 추정 오류: {e}")
                break

        return max(best_font_size, min_font_size)

    def _wrap_text(
        self,
        text: str,
        font,
        max_width: int
    ) -> str:
        """
        텍스트를 주어진 너비에 맞게 줄바꿈

        Args:
            text: 원본 텍스트
            font: PIL 폰트 객체
            max_width: 최대 너비 (픽셀)

        Returns:
            줄바꿈된 텍스트
        """
        if font is None:
            return text

        words = text.split()
        if not words:
            return text

        lines = []
        current_line = []
        current_width = 0

        for word in words:
            try:
                word_bbox = font.getbbox(word + " ")
                word_width = word_bbox[2] - word_bbox[0]
            except Exception:
                word_width = len(word) * 10  # 추정값

            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines) if lines else text

    def _calculate_text_position(
        self,
        bbox: Tuple[int, int, int, int],
        text: str,
        font,
        align: str = "center"
    ) -> Tuple[int, int]:
        """
        텍스트 위치 계산 (개선된 알고리즘)

        Args:
            bbox: 영역 (x1, y1, x2, y2)
            text: 텍스트
            font: 폰트 객체
            align: 정렬 방식 (left, center, right)

        Returns:
            (x, y) 위치
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if font:
            try:
                text_bbox = font.getbbox(text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                # ascent/descent 고려
                ascent = -text_bbox[1]  # 보통 음수
            except Exception:
                text_width = len(text) * 10
                text_height = 12
                ascent = 10
        else:
            text_width = len(text) * 10
            text_height = 12
            ascent = 10

        # X 위치 계산
        if align == "left":
            x = x1 + 2  # 약간의 패딩
        elif align == "right":
            x = x2 - text_width - 2
        else:  # center
            x = x1 + (width - text_width) // 2

        # Y 위치: 수직 중앙 정렬, baseline 고려
        y = y1 + (height - text_height) // 2

        return max(x, x1), max(y, y1)

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

        if not polygon or len(polygon) < 3:
            return image

        img_height, img_width = image.shape[:2]

        # polygon 좌표 검증 및 클리핑
        valid_polygon = []
        for pt in polygon:
            if len(pt) >= 2:
                x = max(0, min(int(pt[0]), img_width - 1))
                y = max(0, min(int(pt[1]), img_height - 1))
                valid_polygon.append([x, y])

        if len(valid_polygon) < 3:
            return image

        # 마스크 생성
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array(valid_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # 마스크가 비어있으면 원본 반환
        if np.sum(mask) == 0:
            return image

        # 마스크 확장 (텍스트 경계 포함)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 인페인팅
        try:
            result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return result
        except Exception as e:
            logger.debug(f"인페인팅 실패: {e}")
            return image

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

        if not polygon or len(polygon) < 3:
            return image

        img_height, img_width = image.shape[:2]

        # polygon 좌표 검증 및 클리핑
        valid_polygon = []
        for pt in polygon:
            if len(pt) >= 2:
                x = max(0, min(int(pt[0]), img_width - 1))
                y = max(0, min(int(pt[1]), img_height - 1))
                valid_polygon.append([x, y])

        if len(valid_polygon) < 3:
            return image

        result = image.copy()
        pts = np.array(valid_polygon, dtype=np.int32)
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
        이미지 내 텍스트 교체 (개선된 버전)

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

        # 먼저 모든 텍스트 영역을 인페인팅/지우기
        for replacement in replacements:
            try:
                if use_inpaint:
                    result = self._inpaint_region(result, replacement.polygon)
                else:
                    bg_color = replacement.background_color
                    if bg_color is None:
                        bg_color = self._get_dominant_color(
                            image, replacement.bbox, is_background=True
                        )
                    result = self._fill_region_with_color(
                        result, replacement.polygon, bg_color
                    )
            except Exception as e:
                logger.debug(f"텍스트 영역 지우기 실패: {e}")

        # BGR to RGB for PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(result_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 번역된 텍스트 삽입
        text_drawn_count = 0
        for replacement in replacements:
            try:
                x1, y1, x2, y2 = replacement.bbox
                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    logger.debug(f"박스 크기 무효: {box_width}x{box_height}")
                    continue

                translated_text = replacement.translated_text.strip()
                if not translated_text:
                    logger.debug("번역 텍스트 비어있음")
                    continue

                # 폰트 크기 결정 (개선된 알고리즘 사용)
                font_size = replacement.font_size
                if font_size is None:
                    font_size = self._estimate_font_size(
                        replacement.bbox,
                        translated_text,
                        lang=lang
                    )

                font = self._get_font(font_size, lang)
                if font is None:
                    logger.warning(f"폰트 로드 실패 (크기: {font_size})")

                # 텍스트 색상 결정
                text_color = replacement.font_color
                if text_color is None:
                    # 배경과 대비되는 색상 사용
                    bg = self._get_dominant_color(result, replacement.bbox, True)
                    brightness = (bg[0] * 299 + bg[1] * 587 + bg[2] * 114) / 1000
                    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

                # 텍스트를 bbox에 맞게 줄바꿈
                wrapped_text = self._wrap_text(translated_text, font, box_width - 4)
                lines = wrapped_text.split('\n')

                # 줄 높이 계산
                line_height = font_size * 1.2

                # 전체 텍스트 높이 계산
                total_text_height = len(lines) * line_height

                # 시작 Y 위치 (수직 중앙 정렬)
                start_y = y1 + (box_height - total_text_height) / 2

                # 각 줄 그리기
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue

                    # 이 줄의 너비 계산
                    if font:
                        try:
                            line_bbox = font.getbbox(line)
                            line_width = line_bbox[2] - line_bbox[0]
                        except Exception:
                            line_width = len(line) * font_size // 2
                    else:
                        line_width = len(line) * font_size // 2

                    # X 위치 (수평 중앙 정렬)
                    x = x1 + (box_width - line_width) // 2
                    y = start_y + i * line_height

                    # 경계 확인
                    x = max(x1, min(x, x2 - line_width))
                    y = max(y1, min(y, y2 - font_size))

                    # 텍스트 그리기
                    draw.text((x, y), line, fill=text_color, font=font)

                text_drawn_count += 1

            except Exception as e:
                logger.error(f"텍스트 교체 실패 '{replacement.translated_text[:20]}...': {e}")
                continue

        logger.info(f"텍스트 렌더링 완료: {text_drawn_count}/{len(replacements)}개 텍스트 그려짐")

        # RGB to BGR
        result = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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

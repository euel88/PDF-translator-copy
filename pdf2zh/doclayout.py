"""
문서 레이아웃 감지 모듈 - PDFMathTranslate 구조 기반
DocLayout-YOLO ONNX 모델 사용
차트, 표, 목차, 주석 등 문서 요소 완벽 보존
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from PIL import Image

from pdf2zh.config import config


class ElementType(Enum):
    """문서 요소 타입"""
    TEXT = "text"
    TITLE = "title"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    REFERENCE = "reference"
    # 추가된 요소 타입
    TOC = "toc"                    # 목차 (Table of Contents)
    FOOTNOTE = "footnote"          # 각주
    ENDNOTE = "endnote"            # 미주
    ANNOTATION = "annotation"      # 주석
    SIDEBAR = "sidebar"            # 사이드바
    WATERMARK = "watermark"        # 워터마크
    PAGE_NUMBER = "page_number"    # 페이지 번호
    CHART = "chart"                # 차트/그래프
    DIAGRAM = "diagram"            # 다이어그램
    CODE = "code"                  # 코드 블록
    QUOTE = "quote"                # 인용문
    ABSTRACT = "abstract"          # 초록
    KEYWORDS = "keywords"          # 키워드
    BIBLIOGRAPHY = "bibliography"  # 참고문헌


@dataclass
class LayoutBox:
    """레이아웃 박스 - 향상된 버전"""
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    label: str  # text, title, figure, table, formula, etc.
    confidence: float
    # 추가 속성
    content: Optional[str] = None         # 추출된 내용
    children: List['LayoutBox'] = field(default_factory=list)  # 하위 요소
    parent: Optional['LayoutBox'] = None  # 상위 요소
    metadata: Dict[str, Any] = field(default_factory=dict)  # 메타데이터
    preserve: bool = False                # 보존 여부 (번역하지 않음)

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]
    @property
    def width(self): return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, other: 'LayoutBox') -> bool:
        """다른 박스를 포함하는지 확인"""
        return (self.x0 <= other.x0 and self.y0 <= other.y0 and
                self.x1 >= other.x1 and self.y1 >= other.y1)

    def intersects(self, other: 'LayoutBox') -> bool:
        """다른 박스와 겹치는지 확인"""
        return not (self.x1 < other.x0 or other.x1 < self.x0 or
                    self.y1 < other.y0 or other.y1 < self.y0)

    def intersection_area(self, other: 'LayoutBox') -> float:
        """겹치는 영역 크기"""
        x0 = max(self.x0, other.x0)
        y0 = max(self.y0, other.y0)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        return (x1 - x0) * (y1 - y0)


# 레이아웃 레이블 (확장)
LAYOUT_LABELS = {
    0: "text",
    1: "title",
    2: "figure",
    3: "table",
    4: "formula",
    5: "caption",
    6: "header",
    7: "footer",
    8: "list",
    9: "reference",
    # 확장된 레이블
    10: "toc",
    11: "footnote",
    12: "annotation",
    13: "chart",
    14: "diagram",
    15: "code",
    16: "quote",
    17: "abstract",
    18: "bibliography",
    19: "page_number",
    20: "watermark",
}

# 번역 대상 레이블
TRANSLATABLE_LABELS = {
    "text", "title", "caption", "list", "header", "footer",
    "quote", "abstract"  # 인용문과 초록도 번역
}

# 보존 대상 레이블 (번역하지 않음)
PRESERVE_LABELS = {
    "figure", "table", "formula", "reference",
    "chart", "diagram", "code", "watermark", "page_number",
    "toc", "footnote", "annotation", "bibliography"
}

# 부분 번역 대상 (제목/레이블만 번역)
PARTIAL_TRANSLATE_LABELS = {"toc", "bibliography"}


class OnnxModel:
    """ONNX 모델 래퍼"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.input_shape = None
        self._loaded = False

    def load(self) -> bool:
        """모델 로드"""
        if self._loaded:
            return True

        try:
            import onnxruntime as ort

            if self.model_path and os.path.exists(self.model_path):
                self.session = ort.InferenceSession(
                    self.model_path,
                    providers=['CPUExecutionProvider']
                )
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                self._loaded = True
                return True
        except ImportError:
            pass
        except Exception as e:
            print(f"ONNX 모델 로드 실패: {e}")

        return False

    def predict(self, image: np.ndarray) -> np.ndarray:
        """예측 수행"""
        if not self._loaded:
            return np.array([])

        # 이미지 전처리
        input_data = self._preprocess(image)

        # 추론
        outputs = self.session.run(None, {self.input_name: input_data})

        return outputs[0]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 리사이즈
        target_size = (640, 640)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        from PIL import Image
        img = Image.fromarray(image)
        img = img.resize(target_size)

        # 정규화
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        arr = np.expand_dims(arr, 0)  # 배치 차원 추가

        return arr


class ModelInstance:
    """모델 인스턴스 싱글톤"""

    _instance: Optional['ModelInstance'] = None
    _model: Optional[OnnxModel] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, model_path: Optional[str] = None) -> Optional[OnnxModel]:
        """모델 인스턴스 반환"""
        if cls._model is None and model_path:
            cls._model = OnnxModel(model_path)
            cls._model.load()
        return cls._model

    @classmethod
    def is_available(cls) -> bool:
        """모델 사용 가능 여부"""
        return cls._model is not None and cls._model._loaded


class LayoutDetector:
    """레이아웃 감지기 - PDFMathTranslate 수준의 정밀 감지"""

    # 텍스트 패턴 감지용 정규식
    TOC_PATTERNS = [
        r'(?i)^(table\s+of\s+)?contents?$',
        r'(?i)^목\s*차$',
        r'(?i)^目\s*[次录錄]$',
        r'(?i)^inhalt(sverzeichnis)?$',
        r'(?i)^(table\s+des\s+)?mati[èe]res?$',
    ]

    BIBLIOGRAPHY_PATTERNS = [
        r'(?i)^(references?|bibliography|works?\s+cited)$',
        r'(?i)^참\s*고\s*문\s*헌$',
        r'(?i)^参考文[献獻]$',
        r'(?i)^literatur(verzeichnis)?$',
        r'(?i)^bibliographie$',
    ]

    ABSTRACT_PATTERNS = [
        r'(?i)^abstract$',
        r'(?i)^요\s*약$',
        r'(?i)^摘\s*要$',
        r'(?i)^zusammenfassung$',
        r'(?i)^r[ée]sum[ée]$',
    ]

    FOOTNOTE_PATTERNS = [
        r'^\d+[\.\)]\s',           # "1. " or "1) "
        r'^\[\d+\]\s',             # "[1] "
        r'^[†‡§¶]\s',              # 특수 기호
        r'^\*+\s',                 # "* " or "** "
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self._find_model()
        self.model = None
        self.use_fallback = True

        if self.model_path:
            self.model = OnnxModel(self.model_path)
            if self.model.load():
                self.use_fallback = False

    def _find_model(self) -> Optional[str]:
        """모델 파일 찾기"""
        # 일반적인 모델 경로들
        paths = [
            Path.home() / ".cache" / "pdf2zh" / "doclayout.onnx",
            Path(__file__).parent / "models" / "doclayout.onnx",
            Path("models") / "doclayout.onnx",
        ]

        for p in paths:
            if p.exists():
                return str(p)

        return None

    def detect(self, image: Image.Image) -> List[LayoutBox]:
        """레이아웃 감지"""
        if self.use_fallback:
            return self._detect_fallback(image)

        return self._detect_onnx(image)

    def detect_from_pdf_page(
        self,
        page_dict: Dict[str, Any],
        page_width: float,
        page_height: float
    ) -> List[LayoutBox]:
        """
        PDF 페이지 구조에서 직접 레이아웃 감지
        PyMuPDF의 get_text("dict") 결과를 분석
        """
        boxes = []

        for block in page_dict.get("blocks", []):
            block_type = block.get("type", 0)
            bbox = block.get("bbox", (0, 0, 0, 0))

            if block_type == 0:  # 텍스트 블록
                text_content = self._extract_block_text(block)
                label = self._classify_text_block(text_content, bbox, page_width, page_height)
                preserve = label in PRESERVE_LABELS

                boxes.append(LayoutBox(
                    bbox=bbox,
                    label=label,
                    confidence=0.8,
                    content=text_content,
                    preserve=preserve
                ))

            elif block_type == 1:  # 이미지 블록
                # 이미지 특성 분석
                label = self._classify_image_block(block, bbox, page_width, page_height)

                boxes.append(LayoutBox(
                    bbox=bbox,
                    label=label,
                    confidence=0.9,
                    preserve=True  # 이미지는 항상 보존
                ))

        # 테이블 감지 (구조 분석 기반)
        table_boxes = self._detect_tables_from_structure(page_dict, page_width, page_height)
        boxes.extend(table_boxes)

        # 중복 제거 및 병합
        boxes = self.merge_overlapping(boxes)

        return boxes

    def _extract_block_text(self, block: Dict) -> str:
        """블록에서 텍스트 추출"""
        texts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    texts.append(text)
        return " ".join(texts)

    def _classify_text_block(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float
    ) -> str:
        """텍스트 블록 분류"""
        import re

        text_lower = text.lower().strip()
        x0, y0, x1, y1 = bbox

        # 목차 감지
        for pattern in self.TOC_PATTERNS:
            if re.match(pattern, text_lower):
                return "toc"

        # 참고문헌 감지
        for pattern in self.BIBLIOGRAPHY_PATTERNS:
            if re.match(pattern, text_lower):
                return "bibliography"

        # 초록 감지
        for pattern in self.ABSTRACT_PATTERNS:
            if re.match(pattern, text_lower):
                return "abstract"

        # 각주 감지 (페이지 하단)
        if y0 > page_height * 0.85:
            for pattern in self.FOOTNOTE_PATTERNS:
                if re.match(pattern, text):
                    return "footnote"

        # 페이지 번호 감지
        if y0 > page_height * 0.9 or y1 < page_height * 0.1:
            if re.match(r'^-?\s*\d+\s*-?$', text) or re.match(r'^[ivxlcdm]+$', text_lower):
                return "page_number"

        # 헤더 감지 (상단)
        if y1 < page_height * 0.1:
            return "header"

        # 푸터 감지 (하단)
        if y0 > page_height * 0.9:
            return "footer"

        # 제목 감지 (중앙 정렬, 큰 폰트 등)
        block_width = x1 - x0
        margin_left = x0
        margin_right = page_width - x1
        is_centered = abs(margin_left - margin_right) < page_width * 0.1
        is_short = len(text) < 100

        if is_centered and is_short:
            return "title"

        # 인용문 감지
        if text.startswith('"') or text.startswith("'") or text.startswith("「"):
            return "quote"

        # 리스트 감지
        list_patterns = [
            r'^[\-•◦▪▸►]\s',
            r'^\d+[\.\)]\s',
            r'^[a-z][\.\)]\s',
            r'^[ivx]+[\.\)]\s',
        ]
        for pattern in list_patterns:
            if re.match(pattern, text):
                return "list"

        # 코드 블록 감지
        code_indicators = ['def ', 'function ', 'class ', 'import ', 'from ', '#include', 'public ', 'private ']
        if any(ind in text for ind in code_indicators):
            return "code"

        return "text"

    def _classify_image_block(
        self,
        block: Dict,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float
    ) -> str:
        """이미지 블록 분류"""
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0

        # 매우 작은 이미지는 아이콘/장식으로 간주
        if width < page_width * 0.1 and height < page_height * 0.1:
            return "figure"

        # 정사각형에 가까우면 차트/다이어그램 가능성
        aspect_ratio = width / max(height, 1)
        if 0.8 < aspect_ratio < 1.2:
            return "chart"

        # 넓고 낮은 이미지는 표 또는 다이어그램
        if aspect_ratio > 2:
            return "diagram"

        return "figure"

    def _detect_tables_from_structure(
        self,
        page_dict: Dict,
        page_width: float,
        page_height: float
    ) -> List[LayoutBox]:
        """구조 분석으로 테이블 감지"""
        tables = []

        # 그리드 패턴 감지 (정렬된 텍스트 블록)
        blocks = page_dict.get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == 0]

        if len(text_blocks) < 3:
            return tables

        # 수평 정렬 감지
        y_positions = {}
        for block in text_blocks:
            bbox = block.get("bbox", (0, 0, 0, 0))
            y_center = (bbox[1] + bbox[3]) / 2
            # 10픽셀 오차 허용
            y_key = round(y_center / 10) * 10
            if y_key not in y_positions:
                y_positions[y_key] = []
            y_positions[y_key].append(bbox)

        # 같은 행에 3개 이상의 블록이 있으면 테이블 가능성
        table_rows = [bboxes for bboxes in y_positions.values() if len(bboxes) >= 3]

        if len(table_rows) >= 2:
            # 테이블 영역 계산
            all_bboxes = [bbox for bboxes in table_rows for bbox in bboxes]
            x0 = min(b[0] for b in all_bboxes)
            y0 = min(b[1] for b in all_bboxes)
            x1 = max(b[2] for b in all_bboxes)
            y1 = max(b[3] for b in all_bboxes)

            tables.append(LayoutBox(
                bbox=(x0, y0, x1, y1),
                label="table",
                confidence=0.7,
                preserve=True
            ))

        return tables

    def _detect_onnx(self, image: Image.Image) -> List[LayoutBox]:
        """ONNX 모델로 감지"""
        arr = np.array(image)
        predictions = self.model.predict(arr)

        boxes = []
        h, w = arr.shape[:2]

        for pred in predictions:
            if len(pred) >= 6:
                x0, y0, x1, y1, conf, class_id = pred[:6]

                if conf < 0.5:
                    continue

                # 좌표 스케일링
                x0 = x0 * w / 640
                y0 = y0 * h / 640
                x1 = x1 * w / 640
                y1 = y1 * h / 640

                label = LAYOUT_LABELS.get(int(class_id), "text")
                preserve = label in PRESERVE_LABELS

                boxes.append(LayoutBox(
                    bbox=(x0, y0, x1, y1),
                    label=label,
                    confidence=float(conf),
                    preserve=preserve
                ))

        return boxes

    def _detect_fallback(self, image: Image.Image) -> List[LayoutBox]:
        """폴백: 전체 페이지를 텍스트 영역으로 처리"""
        w, h = image.size
        margin = 50

        return [LayoutBox(
            bbox=(margin, margin, w - margin, h - margin),
            label="text",
            confidence=1.0
        )]

    def filter_translatable(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """번역 가능한 영역만 필터링"""
        return [b for b in boxes if b.label in TRANSLATABLE_LABELS and not b.preserve]

    def filter_preserve(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """보존할 영역만 필터링"""
        return [b for b in boxes if b.label in PRESERVE_LABELS or b.preserve]

    def filter_partial_translate(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """부분 번역 대상 필터링"""
        return [b for b in boxes if b.label in PARTIAL_TRANSLATE_LABELS]

    def sort_reading_order(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """읽기 순서로 정렬 (상단에서 하단, 좌에서 우)"""
        return sorted(boxes, key=lambda b: (b.y0, b.x0))

    def group_by_column(
        self,
        boxes: List[LayoutBox],
        page_width: float,
        column_threshold: float = 0.4
    ) -> List[List[LayoutBox]]:
        """다단 레이아웃 그룹핑"""
        if not boxes:
            return []

        # 중앙선 기준으로 좌우 분리
        center = page_width / 2
        left_boxes = [b for b in boxes if b.center[0] < center * (1 - column_threshold / 2)]
        right_boxes = [b for b in boxes if b.center[0] > center * (1 + column_threshold / 2)]
        center_boxes = [b for b in boxes if b not in left_boxes and b not in right_boxes]

        columns = []
        if left_boxes:
            columns.append(self.sort_reading_order(left_boxes))
        if center_boxes:
            columns.append(self.sort_reading_order(center_boxes))
        if right_boxes:
            columns.append(self.sort_reading_order(right_boxes))

        return columns if columns else [self.sort_reading_order(boxes)]

    def merge_overlapping(
        self,
        boxes: List[LayoutBox],
        iou_threshold: float = 0.5
    ) -> List[LayoutBox]:
        """겹치는 박스 병합"""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
        merged = []

        for box in boxes:
            should_add = True
            for existing in merged:
                if self._calculate_iou(box, existing) > iou_threshold:
                    should_add = False
                    break

            if should_add:
                merged.append(box)

        return merged

    def _calculate_iou(self, box1: LayoutBox, box2: LayoutBox) -> float:
        """IoU 계산"""
        x0 = max(box1.x0, box2.x0)
        y0 = max(box1.y0, box2.y0)
        x1 = min(box1.x1, box2.x1)
        y1 = min(box1.y1, box2.y1)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        intersection = (x1 - x0) * (y1 - y0)
        union = box1.area + box2.area - intersection

        return intersection / union if union > 0 else 0.0

    def build_hierarchy(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """박스 계층 구조 구축 (부모-자식 관계)"""
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)

        for i, box in enumerate(boxes):
            for j, other in enumerate(boxes):
                if i != j and box.contains(other) and other.parent is None:
                    other.parent = box
                    box.children.append(other)

        # 루트 박스만 반환
        return [b for b in boxes if b.parent is None]


def download_model(model_dir: Optional[str] = None) -> Optional[str]:
    """모델 다운로드"""
    if model_dir is None:
        model_dir = Path.home() / ".cache" / "pdf2zh"

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "doclayout.onnx"

    if model_path.exists():
        return str(model_path)

    # 모델 다운로드 URL (예시)
    model_url = "https://github.com/PDFMathTranslate/PDFMathTranslate/releases/download/v1.0/doclayout.onnx"

    try:
        import urllib.request
        print(f"모델 다운로드 중: {model_url}")
        urllib.request.urlretrieve(model_url, str(model_path))
        print(f"모델 저장됨: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"모델 다운로드 실패: {e}")
        return None

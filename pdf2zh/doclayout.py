"""
문서 레이아웃 감지 모듈 - PDFMathTranslate 구조 기반
DocLayout-YOLO ONNX 모델 사용
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
from PIL import Image

from pdf2zh.config import config


@dataclass
class LayoutBox:
    """레이아웃 박스"""
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    label: str  # text, title, figure, table, formula, etc.
    confidence: float

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


# 레이아웃 레이블
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
}

# 번역 대상 레이블
TRANSLATABLE_LABELS = {"text", "title", "caption", "list", "header", "footer"}

# 보존 대상 레이블 (번역하지 않음)
PRESERVE_LABELS = {"figure", "table", "formula", "reference"}


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
    """레이아웃 감지기"""

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

                boxes.append(LayoutBox(
                    bbox=(x0, y0, x1, y1),
                    label=label,
                    confidence=float(conf)
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
        return [b for b in boxes if b.label in TRANSLATABLE_LABELS]

    def filter_preserve(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """보존할 영역만 필터링"""
        return [b for b in boxes if b.label in PRESERVE_LABELS]

    def sort_reading_order(self, boxes: List[LayoutBox]) -> List[LayoutBox]:
        """읽기 순서로 정렬 (상단에서 하단, 좌에서 우)"""
        return sorted(boxes, key=lambda b: (b.y0, b.x0))

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

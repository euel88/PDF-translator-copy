"""
PDF 처리 모듈 - PyMuPDF 기반 텍스트 추출
"""
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Generator, Tuple, Optional
from PIL import Image
import fitz  # PyMuPDF


@dataclass
class TextBlock:
    """텍스트 블록 (PDF 구조에서 추출)"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF coordinates
    font_size: float
    font_name: str
    color: Tuple[int, int, int]  # RGB
    block_type: str  # "text" or "image"

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


@dataclass
class PageInfo:
    """페이지 정보"""
    number: int
    width: float
    height: float
    has_text: bool
    has_images: bool


@dataclass
class PDFInfo:
    """PDF 정보"""
    path: str
    page_count: int
    file_size: int
    title: str
    author: str
    is_scanned: bool


class PDFHandler:
    """PDF 처리 클래스"""

    def __init__(self, path: str, dpi: int = 150):
        self.path = Path(path)
        self.dpi = dpi
        self._doc: fitz.Document = None

    def open(self):
        self._doc = fitz.open(str(self.path))

    def close(self):
        if self._doc:
            self._doc.close()
            self._doc = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def doc(self) -> fitz.Document:
        if self._doc is None:
            self.open()
        return self._doc

    def get_info(self) -> PDFInfo:
        doc = self.doc
        meta = doc.metadata or {}
        is_scanned = self._check_if_scanned()

        return PDFInfo(
            path=str(self.path),
            page_count=doc.page_count,
            file_size=self.path.stat().st_size,
            title=meta.get("title", ""),
            author=meta.get("author", ""),
            is_scanned=is_scanned,
        )

    def _check_if_scanned(self) -> bool:
        """스캔된 PDF인지 확인 (텍스트 레이어 없음)"""
        doc = self.doc
        if doc.page_count == 0:
            return False

        sample_pages = min(3, doc.page_count)
        total_text_len = 0

        for i in range(sample_pages):
            page = doc[i]
            text = page.get_text()
            total_text_len += len(text.strip())

        avg_text = total_text_len / sample_pages
        return avg_text < 100

    def get_page_info(self, page_num: int) -> PageInfo:
        page = self.doc[page_num]
        rect = page.rect
        text = page.get_text().strip()
        images = page.get_images()

        return PageInfo(
            number=page_num,
            width=rect.width,
            height=rect.height,
            has_text=len(text) > 0,
            has_images=len(images) > 0,
        )

    def render_page(self, page_num: int) -> Image.Image:
        """페이지를 이미지로 렌더링"""
        page = self.doc[page_num]
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def get_scale_factor(self, page_num: int) -> float:
        """PDF 좌표를 이미지 좌표로 변환하는 스케일 팩터"""
        return self.dpi / 72

    def extract_text_blocks(self, page_num: int, min_text_len: int = 1) -> List[TextBlock]:
        """
        PDF 구조에서 직접 텍스트 블록 추출 (정확한 위치)

        Args:
            page_num: 페이지 번호
            min_text_len: 최소 텍스트 길이

        Returns:
            TextBlock 리스트
        """
        page = self.doc[page_num]
        blocks = []

        # dict 형식으로 텍스트 추출 (가장 상세한 정보)
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict.get("blocks", []):
            # 이미지 블록 스킵
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue

            # 라인별 처리
            for line in block.get("lines", []):
                line_text = ""
                line_spans = []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        line_text += text + " "
                        line_spans.append(span)

                line_text = line_text.strip()

                if len(line_text) < min_text_len:
                    continue

                if not line_spans:
                    continue

                # 라인의 바운딩 박스 계산
                x0 = min(s["bbox"][0] for s in line_spans)
                y0 = min(s["bbox"][1] for s in line_spans)
                x1 = max(s["bbox"][2] for s in line_spans)
                y1 = max(s["bbox"][3] for s in line_spans)

                # 첫 번째 span에서 폰트 정보
                first_span = line_spans[0]
                font_size = first_span.get("size", 12)
                font_name = first_span.get("font", "")

                # 색상 (정수를 RGB로 변환)
                color_int = first_span.get("color", 0)
                r = (color_int >> 16) & 0xFF
                g = (color_int >> 8) & 0xFF
                b = color_int & 0xFF

                blocks.append(TextBlock(
                    text=line_text,
                    bbox=(x0, y0, x1, y1),
                    font_size=font_size,
                    font_name=font_name,
                    color=(r, g, b),
                    block_type="text",
                ))

        return blocks

    def extract_text_blocks_scaled(
        self,
        page_num: int,
        min_text_len: int = 1
    ) -> List[TextBlock]:
        """
        이미지 좌표로 스케일된 텍스트 블록 추출
        """
        blocks = self.extract_text_blocks(page_num, min_text_len)
        scale = self.get_scale_factor(page_num)

        scaled_blocks = []
        for b in blocks:
            scaled_blocks.append(TextBlock(
                text=b.text,
                bbox=(
                    b.bbox[0] * scale,
                    b.bbox[1] * scale,
                    b.bbox[2] * scale,
                    b.bbox[3] * scale,
                ),
                font_size=b.font_size * scale,
                font_name=b.font_name,
                color=b.color,
                block_type=b.block_type,
            ))

        return scaled_blocks

    def render_pages(
        self, start: int = 0, end: int = None
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        if end is None:
            end = self.doc.page_count - 1
        for i in range(start, end + 1):
            yield i, self.render_page(i)

    def extract_text(self, page_num: int) -> str:
        page = self.doc[page_num]
        return page.get_text()

    @staticmethod
    def create_pdf(images: List[Image.Image], output_path: str):
        """이미지들로 PDF 생성"""
        doc = fitz.open()

        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            page = doc.new_page(width=img.width, height=img.height)
            rect = fitz.Rect(0, 0, img.width, img.height)
            page.insert_image(rect, stream=buf.getvalue())

        doc.save(output_path)
        doc.close()

    @staticmethod
    def merge_pdfs(pdf_paths: List[str], output_path: str):
        doc = fitz.open()

        for path in pdf_paths:
            src = fitz.open(path)
            doc.insert_pdf(src)
            src.close()

        doc.save(output_path)
        doc.close()

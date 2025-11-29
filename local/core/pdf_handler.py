"""
PDF 처리 모듈
"""
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Generator, Tuple
from PIL import Image
import fitz  # PyMuPDF


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
    is_scanned: bool  # 스캔된 PDF인지 (이미지 기반)


class PDFHandler:
    """PDF 처리 클래스"""

    def __init__(self, path: str, dpi: int = 150):
        self.path = Path(path)
        self.dpi = dpi
        self._doc: fitz.Document = None

    def open(self):
        """PDF 열기"""
        self._doc = fitz.open(str(self.path))

    def close(self):
        """PDF 닫기"""
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
        """PDF 정보 반환"""
        doc = self.doc
        meta = doc.metadata or {}

        # 스캔 PDF 여부 판단 (텍스트가 거의 없으면 스캔 PDF)
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
        """스캔된 PDF인지 확인"""
        doc = self.doc
        if doc.page_count == 0:
            return False

        # 첫 3페이지 샘플링
        sample_pages = min(3, doc.page_count)
        total_text_len = 0

        for i in range(sample_pages):
            page = doc[i]
            text = page.get_text()
            total_text_len += len(text.strip())

        # 페이지당 평균 텍스트 길이가 100자 미만이면 스캔 PDF로 판단
        avg_text = total_text_len / sample_pages
        return avg_text < 100

    def get_page_info(self, page_num: int) -> PageInfo:
        """페이지 정보 반환"""
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

    def render_pages(
        self, start: int = 0, end: int = None
    ) -> Generator[Tuple[int, Image.Image], None, None]:
        """여러 페이지를 이미지로 렌더링 (제너레이터)"""
        if end is None:
            end = self.doc.page_count - 1

        for i in range(start, end + 1):
            yield i, self.render_page(i)

    def extract_text(self, page_num: int) -> str:
        """페이지에서 텍스트 추출"""
        page = self.doc[page_num]
        return page.get_text()

    @staticmethod
    def create_pdf(images: List[Image.Image], output_path: str):
        """이미지들로 PDF 생성"""
        doc = fitz.open()

        for img in images:
            # PIL to bytes
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            # 새 페이지 추가
            page = doc.new_page(width=img.width, height=img.height)
            rect = fitz.Rect(0, 0, img.width, img.height)
            page.insert_image(rect, stream=buf.getvalue())

        doc.save(output_path)
        doc.close()

    @staticmethod
    def merge_pdfs(pdf_paths: List[str], output_path: str):
        """여러 PDF 병합"""
        doc = fitz.open()

        for path in pdf_paths:
            src = fitz.open(path)
            doc.insert_pdf(src)
            src.close()

        doc.save(output_path)
        doc.close()

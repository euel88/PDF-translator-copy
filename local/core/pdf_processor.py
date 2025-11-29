"""
PDF 처리 모듈 - 이미지 추출 및 PDF 생성
"""
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Generator
import numpy as np
from PIL import Image


@dataclass
class PDFInfo:
    """PDF 정보"""
    page_count: int
    file_size: int
    title: Optional[str]
    author: Optional[str]
    has_images: bool


@dataclass
class PageImage:
    """페이지 이미지"""
    page_number: int
    image: Image.Image
    width: int
    height: int


class PDFProcessor:
    """PDF 처리 클래스"""

    def __init__(self, pdf_path: str, dpi: int = 150):
        """
        초기화

        Args:
            pdf_path: PDF 파일 경로
            dpi: 렌더링 DPI
        """
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self._doc = None

    def _get_doc(self):
        """PyMuPDF 문서 객체 가져오기"""
        if self._doc is None:
            import fitz
            self._doc = fitz.open(str(self.pdf_path))
        return self._doc

    def close(self):
        """문서 닫기"""
        if self._doc:
            self._doc.close()
            self._doc = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_info(self) -> PDFInfo:
        """PDF 정보 가져오기"""
        doc = self._get_doc()

        metadata = doc.metadata
        has_images = False

        # 첫 페이지에서 이미지 확인
        if doc.page_count > 0:
            page = doc[0]
            images = page.get_images()
            has_images = len(images) > 0

        return PDFInfo(
            page_count=doc.page_count,
            file_size=self.pdf_path.stat().st_size,
            title=metadata.get("title"),
            author=metadata.get("author"),
            has_images=has_images
        )

    def get_page_count(self) -> int:
        """페이지 수 반환"""
        return self._get_doc().page_count

    def render_page(self, page_number: int) -> PageImage:
        """
        페이지를 이미지로 렌더링

        Args:
            page_number: 페이지 번호 (0부터 시작)

        Returns:
            PageImage: 렌더링된 페이지 이미지
        """
        doc = self._get_doc()

        if page_number < 0 or page_number >= doc.page_count:
            raise ValueError(f"유효하지 않은 페이지 번호: {page_number}")

        page = doc[page_number]

        # DPI에 맞춰 줌 계산 (기본 72 DPI)
        zoom = self.dpi / 72

        # 페이지를 픽스맵으로 렌더링
        import fitz
        mat = fitz.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=mat, alpha=False)

        # PIL 이미지로 변환
        image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples
        )

        return PageImage(
            page_number=page_number,
            image=image,
            width=pixmap.width,
            height=pixmap.height
        )

    def render_all_pages(self) -> Generator[PageImage, None, None]:
        """
        모든 페이지를 이미지로 렌더링 (제너레이터)

        Yields:
            PageImage: 각 페이지의 이미지
        """
        doc = self._get_doc()
        for i in range(doc.page_count):
            yield self.render_page(i)

    def render_page_range(
        self,
        start: int,
        end: int
    ) -> Generator[PageImage, None, None]:
        """
        지정된 범위의 페이지를 이미지로 렌더링

        Args:
            start: 시작 페이지 (0부터)
            end: 끝 페이지 (포함)

        Yields:
            PageImage: 각 페이지의 이미지
        """
        doc = self._get_doc()
        end = min(end, doc.page_count - 1)

        for i in range(start, end + 1):
            yield self.render_page(i)

    @staticmethod
    def create_pdf_from_images(
        images: List[Image.Image],
        output_path: str
    ):
        """
        이미지들로 PDF 생성

        Args:
            images: PIL Image 목록
            output_path: 출력 PDF 경로
        """
        if not images:
            raise ValueError("이미지가 없습니다")

        import fitz

        doc = fitz.open()

        for img in images:
            # PIL 이미지를 바이트로 변환
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # 새 페이지 추가
            img_rect = fitz.Rect(0, 0, img.width, img.height)
            page = doc.new_page(width=img.width, height=img.height)

            # 이미지 삽입
            page.insert_image(img_rect, stream=img_bytes.getvalue())

        doc.save(output_path)
        doc.close()

    @staticmethod
    def images_to_pdf_bytes(images: List[Image.Image]) -> bytes:
        """
        이미지들을 PDF 바이트로 변환

        Args:
            images: PIL Image 목록

        Returns:
            PDF 바이트 데이터
        """
        if not images:
            raise ValueError("이미지가 없습니다")

        import fitz

        doc = fitz.open()

        for img in images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            img_rect = fitz.Rect(0, 0, img.width, img.height)
            page = doc.new_page(width=img.width, height=img.height)
            page.insert_image(img_rect, stream=img_bytes.getvalue())

        # 바이트로 저장
        pdf_bytes = doc.tobytes()
        doc.close()

        return pdf_bytes

    def extract_images_from_page(
        self,
        page_number: int
    ) -> List[Tuple[Image.Image, Tuple[float, float, float, float]]]:
        """
        페이지에서 이미지 추출

        Args:
            page_number: 페이지 번호

        Returns:
            (이미지, 위치) 튜플 목록
        """
        doc = self._get_doc()
        page = doc[page_number]

        images = []
        image_list = page.get_images()

        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            if base_image:
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes))

                # 이미지 위치 찾기 (근사값)
                for img_rect in page.get_image_rects(xref):
                    bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                    images.append((pil_image, bbox))
                    break

        return images

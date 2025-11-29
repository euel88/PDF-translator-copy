"""
PDF 인터프리터 모듈 - PDFMathTranslate 구조 기반
PDF 구조 파싱 및 텍스트 추출
"""
import io
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
import fitz  # PyMuPDF


@dataclass
class PDFChar:
    """PDF 문자"""
    char: str
    bbox: Tuple[float, float, float, float]
    font_name: str
    font_size: float
    color: Tuple[int, int, int]
    flags: int = 0  # bold, italic 등

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 2**4)  # fitz font flags

    @property
    def is_italic(self) -> bool:
        return bool(self.flags & 2**1)


@dataclass
class PDFSpan:
    """PDF 텍스트 스팬 (동일 스타일 텍스트)"""
    text: str
    bbox: Tuple[float, float, float, float]
    font_name: str
    font_size: float
    color: Tuple[int, int, int]
    flags: int = 0
    origin: Tuple[float, float] = (0, 0)

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]


@dataclass
class PDFLine:
    """PDF 텍스트 라인"""
    spans: List[PDFSpan] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]

    def get_font_size(self) -> float:
        if not self.spans:
            return 12.0
        return max(s.font_size for s in self.spans)

    def get_color(self) -> Tuple[int, int, int]:
        if not self.spans:
            return (0, 0, 0)
        return self.spans[0].color


@dataclass
class PDFBlock:
    """PDF 텍스트 블록"""
    lines: List[PDFLine] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    block_type: str = "text"  # text, image, etc.

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]


@dataclass
class PDFPage:
    """PDF 페이지"""
    page_num: int
    width: float
    height: float
    blocks: List[PDFBlock] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    rotation: int = 0

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.block_type == "text")


class PDFInterpreter:
    """PDF 인터프리터"""

    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None

    @property
    def page_count(self) -> int:
        return self.doc.page_count if self.doc else 0

    def get_page(self, page_num: int) -> PDFPage:
        """페이지 정보 추출"""
        page = self.doc[page_num]
        rect = page.rect

        pdf_page = PDFPage(
            page_num=page_num,
            width=rect.width,
            height=rect.height,
            rotation=page.rotation
        )

        # 블록 추출
        pdf_page.blocks = self._extract_blocks(page)

        # 이미지 추출
        pdf_page.images = self._extract_images(page)

        return pdf_page

    def _extract_blocks(self, page: fitz.Page) -> List[PDFBlock]:
        """텍스트 블록 추출"""
        blocks = []
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        for block_data in page_dict.get("blocks", []):
            block_type = "image" if block_data.get("type") == 1 else "text"

            if block_type == "image":
                blocks.append(PDFBlock(
                    bbox=tuple(block_data.get("bbox", (0, 0, 0, 0))),
                    block_type="image"
                ))
                continue

            lines = []
            for line_data in block_data.get("lines", []):
                line = self._parse_line(line_data)
                if line.text.strip():
                    lines.append(line)

            if lines:
                x0 = min(l.x0 for l in lines)
                y0 = min(l.y0 for l in lines)
                x1 = max(l.x1 for l in lines)
                y1 = max(l.y1 for l in lines)

                blocks.append(PDFBlock(
                    lines=lines,
                    bbox=(x0, y0, x1, y1),
                    block_type="text"
                ))

        return blocks

    def _parse_line(self, line_data: Dict) -> PDFLine:
        """라인 파싱"""
        spans = []

        for span_data in line_data.get("spans", []):
            text = span_data.get("text", "")
            if not text:
                continue

            bbox = tuple(span_data.get("bbox", (0, 0, 0, 0)))
            font_name = span_data.get("font", "")
            font_size = span_data.get("size", 12)
            flags = span_data.get("flags", 0)
            origin = tuple(span_data.get("origin", (0, 0)))

            # 색상 파싱
            color_int = span_data.get("color", 0)
            r = (color_int >> 16) & 0xFF
            g = (color_int >> 8) & 0xFF
            b = color_int & 0xFF

            spans.append(PDFSpan(
                text=text,
                bbox=bbox,
                font_name=font_name,
                font_size=font_size,
                color=(r, g, b),
                flags=flags,
                origin=origin
            ))

        if not spans:
            return PDFLine()

        x0 = min(s.x0 for s in spans)
        y0 = min(s.y0 for s in spans)
        x1 = max(s.x1 for s in spans)
        y1 = max(s.y1 for s in spans)

        return PDFLine(spans=spans, bbox=(x0, y0, x1, y1))

    def _extract_images(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """이미지 정보 추출"""
        images = []

        for img_info in page.get_images():
            xref = img_info[0]

            try:
                base_image = self.doc.extract_image(xref)
                images.append({
                    "xref": xref,
                    "width": base_image.get("width", 0),
                    "height": base_image.get("height", 0),
                    "colorspace": base_image.get("colorspace", ""),
                    "ext": base_image.get("ext", ""),
                })
            except Exception:
                pass

        return images

    def iter_pages(self, page_nums: Optional[List[int]] = None) -> Iterator[PDFPage]:
        """페이지 순회"""
        if page_nums is None:
            page_nums = range(self.page_count)

        for num in page_nums:
            yield self.get_page(num)

    def extract_all_text(self) -> str:
        """전체 텍스트 추출"""
        texts = []
        for page_num in range(self.page_count):
            page = self.doc[page_num]
            texts.append(page.get_text())
        return "\n\n".join(texts)


class PDFWriter:
    """PDF 작성기"""

    def __init__(self):
        self.doc = fitz.open()

    def add_page(
        self,
        width: float,
        height: float,
        text_items: Optional[List[Dict]] = None,
        image: Optional[bytes] = None
    ) -> fitz.Page:
        """페이지 추가"""
        page = self.doc.new_page(width=width, height=height)

        if image:
            rect = fitz.Rect(0, 0, width, height)
            page.insert_image(rect, stream=image)

        if text_items:
            for item in text_items:
                self._insert_text(page, item)

        return page

    def _insert_text(self, page: fitz.Page, item: Dict):
        """텍스트 삽입"""
        text = item.get("text", "")
        x = item.get("x", 0)
        y = item.get("y", 0)
        font_size = item.get("font_size", 12)
        color = item.get("color", (0, 0, 0))
        font_name = item.get("font_name", "helv")

        # 색상 변환 (0-255 -> 0-1)
        color_float = tuple(c / 255.0 for c in color)

        try:
            page.insert_text(
                point=(x, y),
                text=text,
                fontsize=font_size,
                color=color_float,
                fontname=font_name
            )
        except Exception:
            # 폰트 없으면 기본 폰트로
            page.insert_text(
                point=(x, y),
                text=text,
                fontsize=font_size,
                color=color_float
            )

    def save(self, path: Optional[str] = None) -> bytes:
        """PDF 저장"""
        output = io.BytesIO()
        self.doc.save(output)
        data = output.getvalue()

        if path:
            with open(path, "wb") as f:
                f.write(data)

        return data

    def close(self):
        if self.doc:
            self.doc.close()


def extract_text_with_positions(path: str) -> List[Dict]:
    """위치 정보와 함께 텍스트 추출"""
    results = []

    with PDFInterpreter(path) as interp:
        for page in interp.iter_pages():
            for block in page.blocks:
                if block.block_type != "text":
                    continue

                for line in block.lines:
                    results.append({
                        "page": page.page_num,
                        "text": line.text,
                        "bbox": line.bbox,
                        "font_size": line.get_font_size(),
                        "color": line.get_color()
                    })

    return results


def merge_paragraphs(
    blocks: List[PDFBlock],
    line_spacing_threshold: float = 1.5
) -> List[PDFBlock]:
    """단락 병합"""
    if not blocks:
        return []

    merged = []
    current = None

    for block in blocks:
        if block.block_type != "text":
            if current:
                merged.append(current)
                current = None
            merged.append(block)
            continue

        if current is None:
            current = PDFBlock(
                lines=list(block.lines),
                bbox=block.bbox,
                block_type="text"
            )
        else:
            # 간격 확인
            gap = block.y0 - current.y1
            avg_height = sum(l.y1 - l.y0 for l in current.lines) / len(current.lines)

            if gap < avg_height * line_spacing_threshold:
                # 병합
                current.lines.extend(block.lines)
                current.bbox = (
                    min(current.x0, block.x0),
                    current.y0,
                    max(current.x1, block.x1),
                    block.y1
                )
            else:
                merged.append(current)
                current = PDFBlock(
                    lines=list(block.lines),
                    bbox=block.bbox,
                    block_type="text"
                )

    if current:
        merged.append(current)

    return merged

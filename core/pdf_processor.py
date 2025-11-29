"""
PDF Processing module using pdf2zh.
"""
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, BinaryIO
import logging

logger = logging.getLogger(__name__)


@dataclass
class PDFInfo:
    """Information about a PDF file."""
    filename: str
    page_count: int
    file_size: int
    title: Optional[str] = None
    author: Optional[str] = None
    has_text: bool = True
    is_encrypted: bool = False


class PDFProcessor:
    """
    PDF processing utilities.
    Handles PDF analysis, validation, and preparation for translation.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_pdf_info(self, file_path: str) -> PDFInfo:
        """
        Extract information from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            PDFInfo object with PDF metadata
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)

            info = PDFInfo(
                filename=Path(file_path).name,
                page_count=len(doc),
                file_size=os.path.getsize(file_path),
                title=doc.metadata.get("title"),
                author=doc.metadata.get("author"),
                has_text=self._check_has_text(doc),
                is_encrypted=doc.is_encrypted,
            )

            doc.close()
            return info

        except Exception as e:
            logger.error(f"Error reading PDF info: {e}")
            # Return basic info if PyMuPDF fails
            return PDFInfo(
                filename=Path(file_path).name,
                page_count=0,
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            )

    def _check_has_text(self, doc) -> bool:
        """Check if PDF has extractable text."""
        try:
            # Check first few pages for text
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    return True
            return False
        except Exception:
            return True  # Assume it has text

    def save_uploaded_file(self, uploaded_file: BinaryIO, filename: str) -> str:
        """
        Save an uploaded file to temporary storage.

        Args:
            uploaded_file: File-like object (from Streamlit upload)
            filename: Original filename

        Returns:
            Path to the saved file
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        file_path = os.path.join(self.temp_dir, safe_filename)

        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        return file_path

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        import re
        import uuid

        # Remove path components
        filename = Path(filename).name

        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

        # Add unique suffix
        base, ext = os.path.splitext(filename)
        unique_suffix = str(uuid.uuid4())[:8]

        return f"{base}_{unique_suffix}{ext}"

    def get_page_preview(
        self,
        file_path: str,
        page_num: int = 0,
        scale: float = 1.0
    ) -> Optional[bytes]:
        """
        Generate a preview image of a PDF page.

        Args:
            file_path: Path to the PDF file
            page_num: Page number (0-indexed)
            scale: Scale factor for the image

        Returns:
            PNG image as bytes, or None on error
        """
        try:
            import fitz

            doc = fitz.open(file_path)
            if page_num >= len(doc):
                page_num = 0

            page = doc[page_num]
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PNG
            img_bytes = pix.tobytes("png")

            doc.close()
            return img_bytes

        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            return None

    def validate_pdf(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate that a file is a valid PDF.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (is_valid, message)
        """
        if not os.path.exists(file_path):
            return False, "File does not exist"

        # Check file extension
        if not file_path.lower().endswith(".pdf"):
            return False, "File is not a PDF"

        # Check magic bytes
        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    return False, "File does not appear to be a valid PDF"
        except IOError as e:
            return False, f"Error reading file: {e}"

        # Try to open with PyMuPDF
        try:
            import fitz
            doc = fitz.open(file_path)
            if doc.is_encrypted:
                doc.close()
                return False, "PDF is encrypted/password-protected"
            doc.close()
        except Exception as e:
            return False, f"Error opening PDF: {e}"

        return True, "Valid PDF"

    def cleanup(self, file_path: str):
        """Remove a temporary file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

    def cleanup_all(self):
        """Remove all temporary files in the temp directory."""
        import shutil

        try:
            if os.path.exists(self.temp_dir):
                for item in os.listdir(self.temp_dir):
                    item_path = os.path.join(self.temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

"""
Core modules for PDF translation
"""
from .ocr_engine import OCREngine, TextRegion, OCRResult
from .translator import OpenAITranslator
from .pdf_processor import PDFProcessor
from .image_editor import ImageEditor

__all__ = [
    "OCREngine",
    "TextRegion",
    "OCRResult",
    "OpenAITranslator",
    "PDFProcessor",
    "ImageEditor",
]

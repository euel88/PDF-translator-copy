"""
Core module - PDF processing and translation logic.
핵심 모듈 - PDF 처리 및 번역 로직
"""
from .pdf_processor import PDFProcessor, PDFInfo
from .translator import PDFTranslator, TranslationProgress
from .cache import TranslationCache

# Image translation modules
from .image_ocr import (
    ImageOCR,
    OCRResult,
    TextRegion,
    get_ocr_engine,
    ocr_image
)
from .image_editor import (
    ImageTextEditor,
    TextReplacement,
    replace_image_text
)
from .image_translator import (
    ImageTranslator,
    ImageTranslationResult,
    PDFImageTranslator,
    translate_image,
    translate_pdf_images
)

__all__ = [
    # PDF processing
    "PDFProcessor",
    "PDFInfo",
    "PDFTranslator",
    "TranslationProgress",
    "TranslationCache",
    # Image OCR
    "ImageOCR",
    "OCRResult",
    "TextRegion",
    "get_ocr_engine",
    "ocr_image",
    # Image editing
    "ImageTextEditor",
    "TextReplacement",
    "replace_image_text",
    # Image translation
    "ImageTranslator",
    "ImageTranslationResult",
    "PDFImageTranslator",
    "translate_image",
    "translate_pdf_images",
]

"""
PDF Translator - PDFMathTranslate 구조 기반
"""
from pdf2zh.high_level import translate, translate_stream
from pdf2zh.converter import TranslateResult, PDFConverter, TextBlock
from pdf2zh.translator import BaseTranslator, create_translator
from pdf2zh.doclayout import LayoutDetector, LayoutBox
from pdf2zh.ocr import TesseractOCR, OCRResult
from pdf2zh.pdfinterp import PDFInterpreter, PDFPage, PDFBlock

__version__ = "1.0.0"
__author__ = "PDF Translator"
__all__ = [
    "translate",
    "translate_stream",
    "TranslateResult",
    "PDFConverter",
    "TextBlock",
    "BaseTranslator",
    "create_translator",
    "LayoutDetector",
    "LayoutBox",
    "TesseractOCR",
    "OCRResult",
    "PDFInterpreter",
    "PDFPage",
    "PDFBlock",
]

import logging
logger = logging.getLogger(__name__)

"""
PDF Translator - PDFMathTranslate 구조 기반
"""
from pdf2zh.high_level import translate, translate_stream
from pdf2zh.converter import TranslateResult, PDFConverter, TextBlock
from pdf2zh.translator import BaseTranslator, create_translator
from pdf2zh.doclayout import LayoutDetector, LayoutBox
from pdf2zh.ocr import TesseractOCR, OCRResult
from pdf2zh.pdfinterp import PDFInterpreter, PDFPage, PDFBlock
from pdf2zh.backend import TaskManager, run_server
from pdf2zh.mcp_server import MCPServer

__version__ = "1.0.0"
__author__ = "PDF Translator"
__all__ = [
    # 고수준 API
    "translate",
    "translate_stream",
    "TranslateResult",
    # 변환기
    "PDFConverter",
    "TextBlock",
    # 번역기
    "BaseTranslator",
    "create_translator",
    # 레이아웃
    "LayoutDetector",
    "LayoutBox",
    # OCR
    "TesseractOCR",
    "OCRResult",
    # PDF 파싱
    "PDFInterpreter",
    "PDFPage",
    "PDFBlock",
    # 백엔드
    "TaskManager",
    "run_server",
    "MCPServer",
]

import logging
logger = logging.getLogger(__name__)

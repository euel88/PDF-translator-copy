"""
PDF Translator - PDFMathTranslate 구조 기반
"""
from pdf2zh.high_level import translate, translate_stream
from pdf2zh.converter import TranslateResult

__version__ = "1.0.0"
__author__ = "PDF Translator"
__all__ = ["translate", "translate_stream", "TranslateResult"]

import logging
logger = logging.getLogger(__name__)

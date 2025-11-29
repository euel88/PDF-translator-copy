"""
문서 핸들러 패키지
다양한 문서 형식(PDF, Word, Excel, PowerPoint) 지원
"""
from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult
from pdf2zh.handlers.word import WordHandler
from pdf2zh.handlers.excel import ExcelHandler
from pdf2zh.handlers.pptx import PowerPointHandler

# 파일 확장자와 핸들러 매핑
HANDLERS = {
    '.docx': WordHandler,
    '.doc': WordHandler,  # .doc는 .docx로 변환 필요할 수 있음
    '.xlsx': ExcelHandler,
    '.xls': ExcelHandler,
    '.pptx': PowerPointHandler,
    '.ppt': PowerPointHandler,
}

# 지원 확장자 목록
SUPPORTED_EXTENSIONS = list(HANDLERS.keys())


def get_handler(file_path: str):
    """파일 확장자에 맞는 핸들러 클래스 반환"""
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    return HANDLERS.get(ext)


def is_supported(file_path: str) -> bool:
    """파일이 지원되는 형식인지 확인"""
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    return ext in HANDLERS


__all__ = [
    'BaseDocumentHandler',
    'DocumentResult',
    'WordHandler',
    'ExcelHandler',
    'PowerPointHandler',
    'HANDLERS',
    'SUPPORTED_EXTENSIONS',
    'get_handler',
    'is_supported',
]

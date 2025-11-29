"""
기본 문서 핸들러 추상 클래스
모든 문서 핸들러가 상속하는 기본 인터페이스
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List

from pdf2zh.translator import BaseTranslator


@dataclass
class DocumentResult:
    """문서 번역 결과"""
    output_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    translated_count: int = 0  # 번역된 항목 수
    total_count: int = 0  # 전체 항목 수

    @property
    def progress_percent(self) -> float:
        """진행률 (0-100)"""
        if self.total_count == 0:
            return 0.0
        return (self.translated_count / self.total_count) * 100


class BaseDocumentHandler(ABC):
    """문서 핸들러 기본 클래스"""

    # 지원하는 파일 확장자
    SUPPORTED_EXTENSIONS: List[str] = []

    def __init__(
        self,
        translator: BaseTranslator,
        callback: Optional[Callable[[str], None]] = None
    ):
        """
        Args:
            translator: 번역기 인스턴스
            callback: 진행 상황 콜백 함수
        """
        self.translator = translator
        self.callback = callback

    def log(self, message: str):
        """로그 메시지 출력"""
        if self.callback:
            self.callback(message)

    def translate_text(self, text: str) -> str:
        """단일 텍스트 번역"""
        if not text or not text.strip():
            return text
        return self.translator.translate(text)

    def translate_texts(self, texts: List[str]) -> List[str]:
        """여러 텍스트 배치 번역"""
        return self.translator.translate_batch(texts)

    @abstractmethod
    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> DocumentResult:
        """
        문서 변환 (번역)

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로 (None이면 자동 생성)

        Returns:
            DocumentResult: 변환 결과
        """
        pass

    def generate_output_path(self, input_path: str, suffix: str = "_translated") -> str:
        """출력 파일 경로 자동 생성"""
        inp = Path(input_path)
        return str(inp.parent / f"{inp.stem}{suffix}{inp.suffix}")

    @classmethod
    def supports(cls, file_path: str) -> bool:
        """파일 형식 지원 여부 확인"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

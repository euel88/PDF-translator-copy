"""
로깅 모듈
"""
from datetime import datetime
from typing import Callable, Optional


class Logger:
    """실시간 로거"""

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Args:
            callback: 로그 메시지를 받을 콜백 함수
        """
        self.callback = callback
        self.logs: list[str] = []

    def log(self, message: str, timestamp: bool = True):
        """로그 메시지 출력"""
        if timestamp:
            ts = datetime.now().strftime("%H:%M:%S")
            formatted = f"[{ts}] {message}"
        else:
            formatted = message

        self.logs.append(formatted)

        if self.callback:
            self.callback(formatted)
        else:
            print(formatted)

    def info(self, message: str):
        """정보 로그"""
        self.log(f"INFO: {message}")

    def error(self, message: str):
        """에러 로그"""
        self.log(f"ERROR: {message}")

    def warning(self, message: str):
        """경고 로그"""
        self.log(f"WARN: {message}")

    def debug(self, message: str):
        """디버그 로그"""
        self.log(f"DEBUG: {message}")

    def section(self, title: str):
        """섹션 구분선"""
        self.log("=" * 40, timestamp=False)
        self.log(title)
        self.log("=" * 40, timestamp=False)

    def clear(self):
        """로그 초기화"""
        self.logs.clear()

    def get_all(self) -> str:
        """전체 로그 반환"""
        return "\n".join(self.logs)

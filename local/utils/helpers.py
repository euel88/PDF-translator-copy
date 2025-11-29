"""
유틸리티 함수들
"""
from typing import List, Tuple, Optional
import re


def parse_page_range(page_range_str: str, max_pages: int) -> List[int]:
    """
    페이지 범위 문자열 파싱

    Args:
        page_range_str: 페이지 범위 (예: "1-5, 7, 10-12")
        max_pages: 최대 페이지 수

    Returns:
        페이지 번호 목록 (0-based)
    """
    if not page_range_str or not page_range_str.strip():
        return list(range(max_pages))

    pages = set()

    for part in page_range_str.split(","):
        part = part.strip()

        if "-" in part:
            # 범위
            match = re.match(r"(\d+)\s*-\s*(\d+)", part)
            if match:
                start = int(match.group(1)) - 1  # 0-based
                end = int(match.group(2)) - 1
                start = max(0, start)
                end = min(max_pages - 1, end)
                pages.update(range(start, end + 1))
        else:
            # 단일 페이지
            try:
                page = int(part) - 1  # 0-based
                if 0 <= page < max_pages:
                    pages.add(page)
            except ValueError:
                pass

    return sorted(pages)


def format_file_size(size_bytes: int) -> str:
    """
    파일 크기를 읽기 쉬운 형식으로 변환

    Args:
        size_bytes: 바이트 단위 크기

    Returns:
        형식화된 문자열 (예: "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def estimate_translation_time(
    page_count: int,
    avg_text_per_page: int = 500
) -> str:
    """
    번역 예상 시간 계산

    Args:
        page_count: 페이지 수
        avg_text_per_page: 페이지당 평균 텍스트 수

    Returns:
        예상 시간 문자열
    """
    # 대략적인 추정:
    # - OCR: 페이지당 약 2-3초
    # - 번역: 500자당 약 1초
    # - 이미지 편집: 페이지당 약 1초

    ocr_time = page_count * 2.5
    translation_time = page_count * (avg_text_per_page / 500)
    edit_time = page_count * 1

    total_seconds = ocr_time + translation_time + edit_time

    if total_seconds < 60:
        return f"약 {int(total_seconds)}초"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"약 {minutes}분"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"약 {hours}시간 {minutes}분"


def sanitize_filename(filename: str) -> str:
    """
    파일명에서 특수문자 제거

    Args:
        filename: 원본 파일명

    Returns:
        정리된 파일명
    """
    # 허용되지 않는 문자 제거
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "_", filename)

    # 연속된 밑줄 제거
    sanitized = re.sub(r"_+", "_", sanitized)

    # 앞뒤 공백 및 밑줄 제거
    sanitized = sanitized.strip(" _")

    return sanitized or "unnamed"

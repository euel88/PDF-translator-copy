"""
고수준 API - PDFMathTranslate 구조 기반
"""
import io
from pathlib import Path
from typing import Optional, List, Callable, BinaryIO, Union

from pdf2zh.converter import PDFConverter, TranslateResult
from pdf2zh.translator import create_translator, BaseTranslator
from pdf2zh.config import LANGUAGES


def translate(
    input_path: str,
    output_path: Optional[str] = None,
    source_lang: str = "English",
    target_lang: str = "Korean",
    service: str = "openai",
    pages: Optional[List[int]] = None,
    dpi: int = 150,
    callback: Optional[Callable[[str], None]] = None,
    **kwargs
) -> TranslateResult:
    """
    PDF 번역

    Args:
        input_path: 입력 PDF 경로
        output_path: 출력 PDF 경로 (None이면 자동 생성)
        source_lang: 원본 언어
        target_lang: 대상 언어
        service: 번역 서비스 (openai, google, deepl, ollama)
        pages: 번역할 페이지 목록 (None이면 전체)
        dpi: 렌더링 DPI
        callback: 로그 콜백 함수
        **kwargs: 번역기 추가 옵션

    Returns:
        TranslateResult
    """
    # 출력 경로 자동 생성
    if output_path is None:
        inp = Path(input_path)
        output_path = str(inp.parent / f"{inp.stem}_translated.pdf")

    # 언어 코드 변환
    src_code = LANGUAGES.get(source_lang, source_lang)
    tgt_code = LANGUAGES.get(target_lang, target_lang)

    # 번역기 생성
    translator = create_translator(
        service=service,
        source_lang=src_code,
        target_lang=tgt_code,
        **kwargs
    )

    # 변환기 생성 및 실행
    converter = PDFConverter(
        translator=translator,
        dpi=dpi,
        callback=callback,
    )

    return converter.convert(
        input_path=input_path,
        output_path=output_path,
        pages=pages,
    )


def translate_stream(
    input_stream: Union[bytes, BinaryIO],
    source_lang: str = "English",
    target_lang: str = "Korean",
    service: str = "openai",
    pages: Optional[List[int]] = None,
    dpi: int = 150,
    callback: Optional[Callable[[str], None]] = None,
    **kwargs
) -> TranslateResult:
    """
    스트림에서 PDF 번역

    Args:
        input_stream: 입력 PDF 바이트 또는 파일 객체
        source_lang: 원본 언어
        target_lang: 대상 언어
        service: 번역 서비스
        pages: 번역할 페이지 목록
        dpi: 렌더링 DPI
        callback: 로그 콜백
        **kwargs: 번역기 추가 옵션

    Returns:
        TranslateResult
    """
    import tempfile
    import os

    # 임시 파일에 저장
    if isinstance(input_stream, bytes):
        data = input_stream
    else:
        data = input_stream.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        temp_path = f.name

    try:
        result = translate(
            input_path=temp_path,
            output_path=None,  # 메모리에만
            source_lang=source_lang,
            target_lang=target_lang,
            service=service,
            pages=pages,
            dpi=dpi,
            callback=callback,
            **kwargs
        )
        return result
    finally:
        os.unlink(temp_path)


# CLI 진입점
def main():
    """CLI 메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="PDF 번역기")
    parser.add_argument("input", help="입력 PDF 파일")
    parser.add_argument("-o", "--output", help="출력 PDF 파일")
    parser.add_argument("-s", "--source", default="English", help="원본 언어")
    parser.add_argument("-t", "--target", default="Korean", help="대상 언어")
    parser.add_argument("--service", default="openai", help="번역 서비스")
    parser.add_argument("--pages", help="페이지 범위 (예: 1-5)")
    parser.add_argument("--dpi", type=int, default=150, help="렌더링 DPI")

    args = parser.parse_args()

    # 페이지 파싱
    pages = None
    if args.pages:
        parts = args.pages.split("-")
        if len(parts) == 2:
            pages = list(range(int(parts[0]) - 1, int(parts[1])))

    def log(msg):
        print(msg)

    result = translate(
        input_path=args.input,
        output_path=args.output,
        source_lang=args.source,
        target_lang=args.target,
        service=args.service,
        pages=pages,
        dpi=args.dpi,
        callback=log,
    )

    if result.success:
        print(f"완료: {result.page_count} 페이지")
    else:
        print(f"오류: {result.error}")


if __name__ == "__main__":
    main()

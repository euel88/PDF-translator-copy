"""
고수준 API - PDFMathTranslate 구조 기반
다양한 문서 형식 지원 (PDF, Word, Excel, PowerPoint)
URL 직접 번역, 이중 언어 출력 지원
"""
import io
import os
import re
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Callable, BinaryIO, Union

from pdf2zh.converter import PDFConverter, TranslateResult
from pdf2zh.translator import create_translator, BaseTranslator
from pdf2zh.config import LANGUAGES


def is_url(path: str) -> bool:
    """URL인지 확인"""
    return path.startswith(('http://', 'https://'))


def download_pdf_from_url(url: str, callback: Optional[Callable[[str], None]] = None) -> str:
    """
    URL에서 PDF 다운로드 (arxiv, 일반 PDF URL 지원)

    Args:
        url: PDF URL (arxiv 링크도 지원)
        callback: 로그 콜백

    Returns:
        임시 파일 경로
    """
    def log(msg: str):
        if callback:
            callback(msg)

    # arxiv URL 처리
    # https://arxiv.org/abs/2301.00001 -> https://arxiv.org/pdf/2301.00001.pdf
    if 'arxiv.org/abs/' in url:
        url = url.replace('/abs/', '/pdf/') + '.pdf'
        log(f"arxiv URL 변환: {url}")
    elif 'arxiv.org' in url and not url.endswith('.pdf'):
        # arxiv.org/pdf/2301.00001 형식일 수 있음
        if not url.endswith('.pdf'):
            url = url + '.pdf'

    log(f"PDF 다운로드 중: {url}")

    try:
        # User-Agent 설정 (일부 서버가 봇 차단)
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )

        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()

            # PDF 확인
            if not content.startswith(b'%PDF'):
                raise ValueError("다운로드된 파일이 PDF가 아닙니다")

            # 임시 파일 저장
            fd, temp_path = tempfile.mkstemp(suffix='.pdf')
            with os.fdopen(fd, 'wb') as f:
                f.write(content)

            log(f"다운로드 완료: {len(content) / 1024:.1f} KB")
            return temp_path

    except urllib.error.URLError as e:
        raise ConnectionError(f"URL 접근 실패: {e}")
    except Exception as e:
        raise RuntimeError(f"다운로드 실패: {e}")

# 지원하는 모든 파일 확장자
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.docx': 'Word',
    '.doc': 'Word',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
    '.pptx': 'PowerPoint',
    '.ppt': 'PowerPoint',
}


def get_file_type(file_path: str) -> Optional[str]:
    """파일 확장자로 문서 유형 확인"""
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(ext)


def translate(
    input_path: str,
    output_path: Optional[str] = None,
    source_lang: str = "English",
    target_lang: str = "Korean",
    service: str = "openai",
    pages: Optional[List[int]] = None,
    dpi: int = 150,
    callback: Optional[Callable[[str], None]] = None,
    # 최적화 옵션
    output_quality: int = 85,  # JPEG 품질 (1-100, 이미지 모드용)
    use_vector_text: bool = True,  # 벡터 텍스트 사용 (파일 크기 대폭 감소)
    compress_images: bool = True,  # 이미지 압축 사용
    # 이중 언어 출력
    dual_output: bool = False,  # 이중 언어 PDF 생성
    # 정규표현식 예외 처리
    exclude_patterns: Optional[List[str]] = None,  # 번역 제외 패턴 (정규표현식)
    include_patterns: Optional[List[str]] = None,  # 번역 포함 패턴 (정규표현식)
    # OCR 옵션
    use_ocr: bool = False,  # OCR 사용 (스캔된 PDF/이미지 기반 PDF용)
    **kwargs
) -> TranslateResult:
    """
    문서 번역 (PDF, Word, Excel, PowerPoint, URL 지원)

    Args:
        input_path: 입력 파일 경로 또는 URL (http://, https://)
        output_path: 출력 파일 경로 (None이면 자동 생성)
        source_lang: 원본 언어
        target_lang: 대상 언어
        service: 번역 서비스 (openai, google, deepl, ollama)
        pages: 번역할 페이지 목록 (PDF만 해당, None이면 전체)
        dpi: 렌더링 DPI (PDF만 해당)
        callback: 로그 콜백 함수
        output_quality: JPEG 품질 1-100 (이미지 모드용, 기본 85)
        use_vector_text: 벡터 텍스트 사용 여부 (True면 파일 크기 대폭 감소)
        compress_images: 이미지 압축 사용 여부
        dual_output: 이중 언어 PDF 생성 여부 (원문 + 번역)
        exclude_patterns: 번역에서 제외할 정규표현식 패턴 목록
        include_patterns: 번역에 포함할 정규표현식 패턴 목록 (지정 시 이 패턴만 번역)
        use_ocr: OCR 사용 여부 (스캔된 PDF 또는 이미지 기반 PDF에 필요)
        **kwargs: 번역기 추가 옵션

    Returns:
        TranslateResult
    """
    temp_file = None
    actual_path = input_path

    # URL 처리: 다운로드 후 번역
    if is_url(input_path):
        try:
            temp_file = download_pdf_from_url(input_path, callback)
            actual_path = temp_file
            # URL에서 파일명 추출하여 출력 경로 설정
            if output_path is None:
                # URL에서 파일명 추출
                url_path = input_path.split('?')[0]  # 쿼리 파라미터 제거
                filename = url_path.split('/')[-1]
                if not filename.endswith('.pdf'):
                    filename = 'downloaded.pdf'
                stem = filename.rsplit('.', 1)[0]
                output_path = f"{stem}_translated.pdf"
        except Exception as e:
            return TranslateResult(
                success=False,
                error=f"URL 다운로드 실패: {e}"
            )

    try:
        # 파일 형식 확인
        file_type = get_file_type(actual_path)
        if file_type is None and actual_path.endswith('.pdf'):
            file_type = 'PDF'
        if file_type is None:
            return TranslateResult(
                success=False,
                error=f"지원하지 않는 파일 형식입니다. 지원: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            )

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

        # 파일 형식에 따라 적절한 핸들러 사용
        if file_type == 'PDF':
            result = _translate_pdf(
                actual_path, output_path, translator,
                pages, dpi, callback,
                source_lang=src_code,
                target_lang=tgt_code,
                output_quality=output_quality,
                use_vector_text=use_vector_text,
                compress_images=compress_images,
                dual_output=dual_output,
                exclude_patterns=exclude_patterns,
                include_patterns=include_patterns,
                use_ocr=use_ocr,
            )
        else:
            result = _translate_document(
                actual_path, output_path, translator,
                file_type, callback
            )

        return result

    finally:
        # 임시 파일 정리
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass


def _translate_pdf(
    input_path: str,
    output_path: Optional[str],
    translator: BaseTranslator,
    pages: Optional[List[int]],
    dpi: int,
    callback: Optional[Callable[[str], None]],
    source_lang: str = "eng",
    target_lang: str = "ko",
    output_quality: int = 85,
    use_vector_text: bool = True,
    compress_images: bool = True,
    dual_output: bool = False,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    use_ocr: bool = False,
) -> TranslateResult:
    """PDF 번역"""
    # 출력 경로 자동 생성
    if output_path is None:
        inp = Path(input_path)
        output_path = str(inp.parent / f"{inp.stem}_translated.pdf")

    # 변환기 생성 및 실행 (최적화 옵션 적용)
    converter = PDFConverter(
        translator=translator,
        dpi=dpi,
        callback=callback,
        source_lang=source_lang,
        target_lang=target_lang,
        output_quality=output_quality,
        use_vector_text=use_vector_text,
        compress_images=compress_images,
        dual_output=dual_output,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        use_ocr=use_ocr,
    )

    return converter.convert(
        input_path=input_path,
        output_path=output_path,
        pages=pages,
    )


def _translate_document(
    input_path: str,
    output_path: Optional[str],
    translator: BaseTranslator,
    file_type: str,
    callback: Optional[Callable[[str], None]]
) -> TranslateResult:
    """Word, Excel, PowerPoint 번역"""
    from pdf2zh.handlers import get_handler, DocumentResult

    # 핸들러 가져오기
    handler_class = get_handler(input_path)
    if handler_class is None:
        return TranslateResult(
            success=False,
            error=f"핸들러를 찾을 수 없습니다: {file_type}"
        )

    # 핸들러 생성 및 실행
    handler = handler_class(translator=translator, callback=callback)
    result = handler.convert(input_path, output_path)

    # DocumentResult를 TranslateResult로 변환
    return TranslateResult(
        output_path=result.output_path,
        success=result.success,
        error=result.error,
        page_count=result.total_count  # 항목 수를 페이지 수로 매핑
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

    supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
    parser = argparse.ArgumentParser(
        description=f"문서 번역기 (지원 형식: {supported})"
    )
    parser.add_argument("input", help="입력 파일 (PDF, Word, Excel, PowerPoint)")
    parser.add_argument("-o", "--output", help="출력 파일")
    parser.add_argument("-s", "--source", default="English", help="원본 언어")
    parser.add_argument("-t", "--target", default="Korean", help="대상 언어")
    parser.add_argument("--service", default="openai", help="번역 서비스")
    parser.add_argument("--pages", help="페이지 범위 (PDF만 해당, 예: 1-5)")
    parser.add_argument("--dpi", type=int, default=150, help="렌더링 DPI (PDF만 해당)")

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
        file_type = get_file_type(args.input)
        if file_type == 'PDF':
            print(f"완료: {result.page_count} 페이지")
        else:
            print(f"완료: {result.page_count}개 항목 번역됨")
        print(f"출력: {result.output_path}")
    else:
        print(f"오류: {result.error}")


if __name__ == "__main__":
    main()

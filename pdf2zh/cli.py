#!/usr/bin/env python3
"""
PDF Translator CLI - PDFMathTranslate 수준의 완전한 CLI
"""
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List


def get_version():
    """버전 정보"""
    return "1.0.0"


def create_parser() -> argparse.ArgumentParser:
    """CLI 파서 생성"""
    parser = argparse.ArgumentParser(
        prog="pdf2zh",
        description="PDF 문서 번역기 - 레이아웃, 수식, 차트 보존",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 번역 (영어 -> 한국어, OpenAI)
  pdf2zh input.pdf

  # 특정 서비스와 언어 지정
  pdf2zh input.pdf -s google --from en --to ko

  # 페이지 범위 지정
  pdf2zh input.pdf -p 1-10,15,20-25

  # 멀티스레드 사용
  pdf2zh input.pdf -t 4

  # 커스텀 프롬프트 사용
  pdf2zh input.pdf --prompt "의학 논문 번역 전문가로서..."

  # 캐시 무시
  pdf2zh input.pdf --ignore-cache

  # 디렉토리 일괄 처리
  pdf2zh --dir ./pdfs --output ./translated

지원 서비스:
  openai, google, deepl, claude, ollama, xinference,
  azure, papago, gemini, llm (vLLM/LM Studio)

지원 언어:
  en (영어), ko (한국어), ja (일본어), zh (중국어),
  de (독일어), fr (프랑스어), es (스페인어), ru (러시아어),
  pt (포르투갈어), it (이탈리아어), vi (베트남어), th (태국어), ar (아랍어)
"""
    )

    # 위치 인자
    parser.add_argument(
        "input",
        nargs="?",
        help="입력 PDF 파일 경로"
    )

    # 기본 옵션
    parser.add_argument(
        "-o", "--output",
        help="출력 파일/디렉토리 경로"
    )

    parser.add_argument(
        "-s", "--service",
        default="openai",
        choices=["openai", "google", "deepl", "claude", "ollama",
                 "xinference", "azure", "papago", "gemini", "llm"],
        help="번역 서비스 (기본: openai)"
    )

    parser.add_argument(
        "--from", "-f",
        dest="source_lang",
        default="en",
        help="원본 언어 (기본: en)"
    )

    parser.add_argument(
        "--to", "-l",
        dest="target_lang",
        default="ko",
        help="대상 언어 (기본: ko)"
    )

    # 페이지 옵션
    parser.add_argument(
        "-p", "--pages",
        help="페이지 범위 (예: 1-10,15,20-25)"
    )

    # 성능 옵션
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="병렬 처리 스레드 수 (기본: 1)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="이미지 렌더링 DPI (기본: 150)"
    )

    # 고급 옵션
    parser.add_argument(
        "--prompt",
        help="커스텀 번역 프롬프트"
    )

    parser.add_argument(
        "--model",
        help="LLM 모델 지정 (서비스별 기본값 사용 안 함)"
    )

    parser.add_argument(
        "--api-key",
        help="API 키 (환경 변수 대신 사용)"
    )

    parser.add_argument(
        "--base-url",
        help="API 베이스 URL (OpenAI 호환 엔드포인트)"
    )

    # 캐시 옵션
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="번역 캐시 무시"
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="캐시 삭제"
    )

    # 출력 옵션
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="출력 이미지 품질 (1-100, 기본: 85)"
    )

    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="이미지 압축 비활성화"
    )

    parser.add_argument(
        "--image-mode",
        action="store_true",
        help="이미지 모드 사용 (벡터 모드 대신)"
    )

    # 레이아웃 옵션
    parser.add_argument(
        "--use-layout",
        action="store_true",
        help="레이아웃 감지 사용"
    )

    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="OCR 사용 (스캔 PDF용)"
    )

    # 일괄 처리
    parser.add_argument(
        "--dir",
        dest="input_dir",
        help="입력 디렉토리 (일괄 처리)"
    )

    # 서버 모드
    parser.add_argument(
        "--server",
        action="store_true",
        help="REST API 서버 시작"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="서버 호스트 (기본: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="서버 포트 (기본: 5000)"
    )

    # MCP 서버
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="MCP 서버 시작 (Claude Desktop용)"
    )

    # GUI
    parser.add_argument(
        "--gui",
        action="store_true",
        help="GUI 시작"
    )

    # 기타
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="상세 출력"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"pdf2zh {get_version()}"
    )

    parser.add_argument(
        "--download-fonts",
        action="store_true",
        help="Go Noto Universal 폰트 다운로드"
    )

    parser.add_argument(
        "--download-model",
        action="store_true",
        help="레이아웃 감지 모델 다운로드"
    )

    return parser


def parse_pages(pages_str: str) -> List[int]:
    """페이지 범위 파싱"""
    pages = []

    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))

    # 0-indexed로 변환
    return [p - 1 for p in pages if p > 0]


def translate_file(args) -> bool:
    """단일 파일 번역"""
    from pdf2zh.high_level import translate

    # 출력 경로 결정
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_translated{input_path.suffix}")

    # 페이지 범위
    pages = None
    if args.pages:
        pages = parse_pages(args.pages)

    # 번역기 옵션
    translator_kwargs = {
        "num_threads": args.threads,
        "use_cache": not args.ignore_cache,
    }

    if args.prompt:
        translator_kwargs["custom_prompt"] = args.prompt

    if args.model:
        translator_kwargs["model"] = args.model

    if args.api_key:
        translator_kwargs["api_key"] = args.api_key

    if args.base_url:
        translator_kwargs["base_url"] = args.base_url

    # 진행률 콜백
    def progress_callback(msg: str):
        if args.verbose:
            print(msg)

    # 번역 실행
    result = translate(
        input_path=args.input,
        output_path=output_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        service=args.service,
        pages=pages,
        dpi=args.dpi,
        output_quality=args.quality,
        use_vector_text=not args.image_mode,
        compress_images=not args.no_compress,
        callback=progress_callback,
        **translator_kwargs,
    )

    if result.success:
        print(f"번역 완료: {result.output_path}")
        print(f"페이지 수: {result.page_count}")
        return True
    else:
        print(f"번역 실패: {result.error}")
        return False


def translate_directory(args) -> bool:
    """디렉토리 일괄 번역"""
    from pdf2zh.high_level import translate

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"디렉토리를 찾을 수 없습니다: {input_dir}")
        return False

    # 출력 디렉토리
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_dir / "translated"

    output_dir.mkdir(parents=True, exist_ok=True)

    # PDF 파일 찾기
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"PDF 파일을 찾을 수 없습니다: {input_dir}")
        return False

    print(f"총 {len(pdf_files)}개 파일 번역 시작...")

    success_count = 0
    for i, pdf_file in enumerate(pdf_files):
        print(f"\n[{i+1}/{len(pdf_files)}] {pdf_file.name}")

        output_path = output_dir / f"{pdf_file.stem}_translated.pdf"

        translator_kwargs = {
            "num_threads": args.threads,
            "use_cache": not args.ignore_cache,
        }

        if args.prompt:
            translator_kwargs["custom_prompt"] = args.prompt

        if args.model:
            translator_kwargs["model"] = args.model

        result = translate(
            input_path=str(pdf_file),
            output_path=str(output_path),
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            service=args.service,
            dpi=args.dpi,
            output_quality=args.quality,
            use_vector_text=not args.image_mode,
            compress_images=not args.no_compress,
            **translator_kwargs,
        )

        if result.success:
            print(f"  완료: {output_path}")
            success_count += 1
        else:
            print(f"  실패: {result.error}")

    print(f"\n번역 완료: {success_count}/{len(pdf_files)} 성공")
    return success_count == len(pdf_files)


def start_server(args):
    """REST API 서버 시작"""
    try:
        from pdf2zh.backend import app
        print(f"서버 시작: http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.verbose)
    except ImportError:
        print("Flask가 설치되지 않았습니다: pip install flask")
        return False
    return True


def start_mcp_server(args):
    """MCP 서버 시작"""
    try:
        from pdf2zh.mcp_server import main as mcp_main
        mcp_main()
    except ImportError:
        print("MCP 서버 의존성이 설치되지 않았습니다")
        return False
    return True


def start_gui():
    """GUI 시작"""
    try:
        from pdf2zh.gui import run_gui
        run_gui()
    except ImportError:
        print("PyQt5가 설치되지 않았습니다: pip install PyQt5")
        return False
    return True


def download_fonts():
    """폰트 다운로드"""
    from pdf2zh.fonts import download_go_noto_fonts
    success = download_go_noto_fonts()
    if success:
        print("폰트 다운로드 완료")
    else:
        print("일부 폰트 다운로드 실패")
    return success


def download_model():
    """레이아웃 모델 다운로드"""
    from pdf2zh.doclayout import download_model as dl_model
    path = dl_model()
    if path:
        print(f"모델 다운로드 완료: {path}")
        return True
    else:
        print("모델 다운로드 실패")
        return False


def clear_cache():
    """캐시 삭제"""
    from pdf2zh.cache import cache
    cache.clear()
    print("캐시가 삭제되었습니다")


def main():
    """메인 함수"""
    parser = create_parser()
    args = parser.parse_args()

    # 특수 명령 처리
    if args.clear_cache:
        clear_cache()
        return 0

    if args.download_fonts:
        return 0 if download_fonts() else 1

    if args.download_model:
        return 0 if download_model() else 1

    if args.gui:
        return 0 if start_gui() else 1

    if args.server:
        return 0 if start_server(args) else 1

    if args.mcp:
        return 0 if start_mcp_server(args) else 1

    # 번역 실행
    if args.input_dir:
        return 0 if translate_directory(args) else 1

    if args.input:
        if not os.path.exists(args.input):
            print(f"파일을 찾을 수 없습니다: {args.input}")
            return 1
        return 0 if translate_file(args) else 1

    # 입력 없음 - 도움말 표시
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

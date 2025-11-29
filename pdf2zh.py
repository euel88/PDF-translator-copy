#!/usr/bin/env python3
"""
PDF Translator - 메인 진입점
PDFMathTranslate 구조 기반
"""
import sys
import os

# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


def check_dependencies():
    """필수 의존성 확인"""
    missing = []

    try:
        import fitz
    except ImportError:
        missing.append("PyMuPDF")

    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("필수 패키지가 설치되지 않았습니다:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n다음 명령어로 설치하세요:")
        print("  pip install -r requirements.txt")
        return False

    return True


def setup_tesseract():
    """Tesseract 경로 설정 (Windows)"""
    if sys.platform == "win32":
        import shutil
        if shutil.which("tesseract") is None:
            # Windows 기본 설치 경로들
            paths = [
                r"C:\Program Files\Tesseract-OCR",
                r"C:\Program Files (x86)\Tesseract-OCR",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR"),
            ]
            for path in paths:
                tesseract_exe = os.path.join(path, "tesseract.exe")
                if os.path.exists(tesseract_exe):
                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                    try:
                        import pytesseract
                        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
                    except ImportError:
                        pass
                    break


def main():
    """메인 함수"""
    if not check_dependencies():
        sys.exit(1)

    setup_tesseract()

    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--cli", "-c"]:
            # CLI 모드
            from pdf2zh.high_level import main as cli_main
            cli_main()
        elif arg in ["--help", "-h"]:
            print("PDF Translator")
            print()
            print("사용법:")
            print("  python pdf2zh.py          GUI 실행")
            print("  python pdf2zh.py --cli    CLI 모드")
            print("  python pdf2zh.py input.pdf  파일 직접 번역")
            print()
            print("옵션:")
            print("  -h, --help    도움말 표시")
            print("  -c, --cli     CLI 모드")
        else:
            # 파일 경로로 간주
            if os.path.exists(sys.argv[1]):
                from pdf2zh.high_level import translate
                result = translate(
                    input_path=sys.argv[1],
                    callback=print,
                )
                if result.success:
                    print(f"\n완료: {result.output_path}")
                else:
                    print(f"\n오류: {result.error}")
            else:
                print(f"파일을 찾을 수 없습니다: {sys.argv[1]}")
                sys.exit(1)
    else:
        # GUI 모드
        from pdf2zh.gui import run_gui
        run_gui()


if __name__ == "__main__":
    main()

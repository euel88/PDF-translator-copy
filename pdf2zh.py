#!/usr/bin/env python3
"""
PDF Translator - 메인 진입점
PDFMathTranslate 구조 기반
완전한 CLI, GUI, 서버 모드 지원
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

    # 명령줄 인자가 없으면 GUI 실행
    if len(sys.argv) == 1:
        try:
            from pdf2zh.gui import run_gui
            run_gui()
        except ImportError:
            print("GUI 실행 실패. CLI 모드를 사용하세요:")
            print("  python pdf2zh.py --help")
            sys.exit(1)
    else:
        # CLI 모드
        from pdf2zh.cli import main as cli_main
        sys.exit(cli_main())


if __name__ == "__main__":
    main()

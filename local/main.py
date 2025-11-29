#!/usr/bin/env python3
"""
PDF Translator Local - 로컬 PDF 번역기

Tesseract OCR + OpenAI를 사용한 PDF 이미지 번역기
"""
import sys
import subprocess
import platform
import os
from pathlib import Path

# 경로 설정
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))


# 패키지 이름 매핑 (import 이름 -> pip 패키지 이름)
PACKAGE_MAP = {
    "PyQt5": "PyQt5",
    "fitz": "PyMuPDF",
    "pytesseract": "pytesseract",
    "openai": "openai",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "numpy": "numpy",
}


def install_package(package_name: str) -> bool:
    """패키지 설치"""
    print(f"  설치 중: {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"  완료: {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"  실패: {package_name}")
        return False


def check_tesseract_installed() -> bool:
    """Tesseract OCR 엔진 설치 확인"""
    if platform.system() == "Windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return True
        return False
    else:
        # Linux/Mac: which 명령으로 확인
        try:
            subprocess.check_call(
                ["which", "tesseract"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False


def check_and_install_dependencies() -> bool:
    """의존성 확인 및 자동 설치"""
    missing = []

    # PyQt5
    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")

    # PyMuPDF
    try:
        import fitz
    except ImportError:
        missing.append("fitz")

    # pytesseract
    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")

    # OpenAI
    try:
        import openai
    except ImportError:
        missing.append("openai")

    # OpenCV
    try:
        import cv2
    except ImportError:
        missing.append("cv2")

    # Pillow
    try:
        from PIL import Image
    except ImportError:
        missing.append("PIL")

    # NumPy
    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    if missing:
        print("=" * 50)
        print("필요한 패키지를 자동 설치합니다...")
        print("=" * 50)

        failed = []
        for module_name in missing:
            package_name = PACKAGE_MAP.get(module_name, module_name)
            if not install_package(package_name):
                failed.append(package_name)

        if failed:
            print()
            print("=" * 50)
            print("다음 패키지 설치에 실패했습니다:")
            for pkg in failed:
                print(f"  - {pkg}")
            print()
            print("수동 설치:")
            print(f"  pip install {' '.join(failed)}")
            print("=" * 50)
            return False

        print()
        print("모든 Python 패키지가 설치되었습니다!")
        print("=" * 50)

    # Tesseract OCR 엔진 확인
    if not check_tesseract_installed():
        print()
        print("=" * 50)
        print("Tesseract OCR 엔진이 설치되지 않았습니다!")
        print()
        if platform.system() == "Windows":
            print("Windows 설치 방법:")
            print("  1. https://github.com/UB-Mannheim/tesseract/wiki 에서 다운로드")
            print("  2. tesseract-ocr-w64-setup-xxx.exe 실행")
            print("  3. 설치 시 'Additional language data' 에서 필요한 언어 선택")
            print("     - Korean, Japanese, Chinese 등")
        elif platform.system() == "Darwin":
            print("macOS 설치 방법:")
            print("  brew install tesseract")
            print("  brew install tesseract-lang  # 추가 언어")
        else:
            print("Linux 설치 방법:")
            print("  sudo apt install tesseract-ocr")
            print("  sudo apt install tesseract-ocr-kor  # 한국어")
            print("  sudo apt install tesseract-ocr-jpn  # 일본어")
        print("=" * 50)
        return False

    return True


def main():
    """메인 함수"""
    # 의존성 확인 및 자동 설치
    if not check_and_install_dependencies():
        input("\n아무 키나 누르면 종료합니다...")
        sys.exit(1)

    # PyQt 앱 시작
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    # High DPI 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("PDF Translator Local")
    app.setOrganizationName("PDFTranslator")

    # 스타일 설정
    app.setStyle("Fusion")

    # 메인 윈도우
    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

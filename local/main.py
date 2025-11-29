#!/usr/bin/env python3
"""
PDF Translator Local - 로컬 PDF 번역기

PaddleOCR + OpenAI를 사용한 PDF 이미지 번역기
"""
import sys
from pathlib import Path

# 경로 설정
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))


def check_dependencies():
    """의존성 확인"""
    missing = []

    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")

    try:
        import fitz
    except ImportError:
        missing.append("PyMuPDF")

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        missing.append("paddleocr")

    try:
        import openai
    except ImportError:
        missing.append("openai")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("=" * 50)
        print("다음 패키지가 설치되지 않았습니다:")
        print()
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("설치 방법:")
        print(f"  pip install {' '.join(missing)}")
        print()
        print("또는 전체 의존성 설치:")
        print(f"  pip install -r {APP_DIR / 'requirements.txt'}")
        print("=" * 50)
        return False

    return True


def main():
    """메인 함수"""
    # 의존성 확인
    if not check_dependencies():
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

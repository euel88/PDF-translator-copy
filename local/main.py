#!/usr/bin/env python3
"""
PDF Translator - Local Desktop Application
로컬 PDF 번역기 메인 엔트리포인트
"""
import sys
import subprocess
import platform


def check_and_install_dependencies():
    """필수 의존성 확인 및 설치"""
    required = {
        "PyQt5": "PyQt5>=5.15.0",
        "fitz": "PyMuPDF>=1.23.0",
        "pytesseract": "pytesseract>=0.3.10",
        "PIL": "Pillow>=10.0.0",
        "cv2": "opencv-python>=4.8.0",
        "numpy": "numpy>=1.24.0",
        "openai": "openai>=1.0.0",
    }

    missing = []

    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("=" * 50)
        print("필수 패키지 설치 중...")
        print("=" * 50)
        for pkg in missing:
            print(f"  설치: {pkg}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", pkg, "-q"
                ])
                print(f"  완료: {pkg}")
            except subprocess.CalledProcessError as e:
                print(f"  실패: {pkg} - {e}")
                return False
        print("=" * 50)
        print("모든 패키지 설치 완료!")
        print("=" * 50)

    return True


def check_tesseract():
    """Tesseract OCR 설치 확인"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def show_tesseract_guide():
    """Tesseract 설치 안내"""
    system = platform.system()

    print("\n" + "=" * 50)
    print("Tesseract OCR이 설치되어 있지 않습니다.")
    print("=" * 50)

    if system == "Windows":
        print("""
Windows 설치 방법:
1. 다음 링크에서 설치 파일 다운로드:
   https://github.com/UB-Mannheim/tesseract/wiki

2. 설치 시 언어 데이터 선택:
   - Korean (한국어)
   - Japanese (일본어)
   - Chinese (중국어) 등

3. 설치 후 PATH에 추가하거나 기본 경로 사용:
   C:\\Program Files\\Tesseract-OCR\\tesseract.exe
""")
    elif system == "Darwin":  # macOS
        print("""
macOS 설치 방법:
  brew install tesseract
  brew install tesseract-lang  # 추가 언어

또는:
  sudo port install tesseract
  sudo port install tesseract-<lang>
""")
    else:  # Linux
        print("""
Linux 설치 방법:
  sudo apt update
  sudo apt install tesseract-ocr
  sudo apt install tesseract-ocr-kor  # 한국어
  sudo apt install tesseract-ocr-jpn  # 일본어
  sudo apt install tesseract-ocr-chi-sim  # 중국어 간체
""")

    print("=" * 50)
    print("Tesseract 설치 후 프로그램을 다시 실행해주세요.")
    print("=" * 50)


def main():
    """메인 함수"""
    print("PDF Translator 시작...")

    # 의존성 확인
    if not check_and_install_dependencies():
        print("의존성 설치 실패. 수동으로 설치해주세요:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Tesseract 확인
    if not check_tesseract():
        show_tesseract_guide()
        # 경고만 표시하고 계속 진행 (GUI에서 설정 가능)
        print("\n경고: Tesseract 없이 계속 진행합니다.")
        print("OCR 기능은 Tesseract 설치 후 사용 가능합니다.\n")

    # PyQt5 앱 실행
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt

    # High DPI 지원
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 메인 윈도우
    from gui import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

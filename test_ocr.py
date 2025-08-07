"""
PDF OCR Translator - 테스트 스크립트
시스템 설정과 기능을 테스트합니다
"""

import sys
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Tuple
import logging

# 컬러 출력을 위한 ANSI 코드
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text: str, color: str = Colors.ENDC):
    """컬러 텍스트 출력"""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(title: str):
    """섹션 헤더 출력"""
    print("\n" + "="*60)
    print_colored(f"  {title}", Colors.BOLD)
    print("="*60)

def check_python_version() -> Tuple[bool, str]:
    """Python 버전 확인"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (3.10+ 필요)"

def check_tesseract() -> Tuple[bool, str]:
    """Tesseract 설치 확인"""
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
        return False, "Tesseract 실행 실패"
    except FileNotFoundError:
        return False, "Tesseract가 설치되지 않음"
    except Exception as e:
        return False, str(e)

def check_tesseract_languages() -> Dict[str, bool]:
    """Tesseract 언어 팩 확인"""
    languages = {
        'eng': 'English',
        'kor': 'Korean',
        'chi_sim': 'Chinese Simplified',
        'chi_tra': 'Chinese Traditional',
        'jpn': 'Japanese',
        'fra': 'French',
        'deu': 'German',
        'spa': 'Spanish'
    }
    
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=5
        )
        available_langs = result.stdout
        
        lang_status = {}
        for code, name in languages.items():
            lang_status[name] = code in available_langs
        
        return lang_status
    except:
        return {name: False for name in languages.values()}

def check_python_packages() -> Dict[str, bool]:
    """Python 패키지 설치 확인"""
    packages = {
        'streamlit': 'Streamlit (UI)',
        'pdf2zh': 'PDF2ZH (텍스트 번역)',
        'pytesseract': 'Pytesseract (OCR)',
        'easyocr': 'EasyOCR (고급 OCR)',
        'cv2': 'OpenCV (이미지 처리)',
        'PIL': 'Pillow (이미지 처리)',
        'fitz': 'PyMuPDF (PDF 처리)',
        'openai': 'OpenAI (GPT 번역)',
        'googletrans': 'Google Translate',
        'numpy': 'NumPy (수치 계산)',
        'requests': 'Requests (HTTP)'
    }
    
    status = {}
    for module, name in packages.items():
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            elif module == 'fitz':
                import fitz
            else:
                __import__(module)
            status[name] = True
        except ImportError:
            status[name] = False
    
    return status

def check_api_keys() -> Dict[str, bool]:
    """API 키 설정 확인"""
    keys = {
        'OPENAI_API_KEY': 'OpenAI',
        'DEEPL_API_KEY': 'DeepL',
        'PAPAGO_CLIENT_ID': 'Papago ID',
        'PAPAGO_CLIENT_SECRET': 'Papago Secret',
        'AZURE_API_KEY': 'Azure'
    }
    
    status = {}
    for env_var, name in keys.items():
        value = os.getenv(env_var)
        if value and value != '' and not value.startswith('your-'):
            status[name] = True
        else:
            status[name] = False
    
    return status

def test_ocr_simple():
    """간단한 OCR 테스트"""
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 테스트 이미지 생성
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Hello OCR Test", fill='black')
        draw.text((10, 40), "한글 테스트", fill='black')
        draw.text((10, 70), "12345", fill='black')
        
        # OCR 수행
        text = pytesseract.image_to_string(img)
        
        if "Hello" in text or "OCR" in text or "Test" in text:
            return True, "영문 인식 성공"
        else:
            return False, f"인식된 텍스트: {text[:50]}"
            
    except Exception as e:
        return False, str(e)

def test_pdf_processing():
    """PDF 처리 테스트"""
    try:
        import fitz
        
        # 테스트 PDF 생성
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test PDF Document")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            doc.save(tmp.name)
            tmp_path = tmp.name
        
        # PDF 다시 열기
        doc2 = fitz.open(tmp_path)
        text = doc2[0].get_text()
        doc2.close()
        
        # 임시 파일 삭제
        os.unlink(tmp_path)
        
        if "Test PDF" in text:
            return True, "PDF 생성 및 읽기 성공"
        else:
            return False, "PDF 텍스트 추출 실패"
            
    except Exception as e:
        return False, str(e)

def test_translation():
    """번역 테스트"""
    try:
        from googletrans import Translator
        
        translator = Translator()
        result = translator.translate("Hello", src='en', dest='ko')
        
        if result and result.text:
            return True, f"번역 성공: Hello → {result.text}"
        else:
            return False, "번역 결과 없음"
            
    except Exception as e:
        return False, f"Google 번역 테스트 실패: {e}"

def check_disk_space() -> Tuple[bool, str]:
    """디스크 공간 확인"""
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        
        if free_gb >= 2:
            return True, f"{free_gb} GB 사용 가능"
        else:
            return False, f"{free_gb} GB (최소 2GB 필요)"
            
    except Exception as e:
        return False, str(e)

def main():
    """메인 테스트 함수"""
    print_colored("\n🔍 PDF OCR Translator - 시스템 테스트", Colors.BOLD)
    print_colored("=" * 60, Colors.BLUE)
    
    all_pass = True
    
    # 1. 시스템 요구사항 확인
    print_header("시스템 요구사항")
    
    # Python 버전
    status, msg = check_python_version()
    if status:
        print_colored(f"✅ Python 버전: {msg}", Colors.GREEN)
    else:
        print_colored(f"❌ Python 버전: {msg}", Colors.RED)
        all_pass = False
    
    # 디스크 공간
    status, msg = check_disk_space()
    if status:
        print_colored(f"✅ 디스크 공간: {msg}", Colors.GREEN)
    else:
        print_colored(f"❌ 디스크 공간: {msg}", Colors.RED)
        all_pass = False
    
    # Tesseract
    status, msg = check_tesseract()
    if status:
        print_colored(f"✅ Tesseract: {msg}", Colors.GREEN)
    else:
        print_colored(f"❌ Tesseract: {msg}", Colors.RED)
        all_pass = False
    
    # 2. Tesseract 언어 팩
    print_header("Tesseract 언어 팩")
    lang_status = check_tesseract_languages()
    for lang, available in lang_status.items():
        if available:
            print_colored(f"✅ {lang}", Colors.GREEN)
        else:
            print_colored(f"⚠️  {lang} (선택사항)", Colors.YELLOW)
    
    # 3. Python 패키지
    print_header("Python 패키지")
    pkg_status = check_python_packages()
    for pkg, installed in pkg_status.items():
        if installed:
            print_colored(f"✅ {pkg}", Colors.GREEN)
        else:
            print_colored(f"❌ {pkg}", Colors.RED)
            if pkg in ['Streamlit (UI)', 'PDF2ZH (텍스트 번역)', 'Pytesseract (OCR)', 'PyMuPDF (PDF 처리)']:
                all_pass = False
    
    # 4. API 키
    print_header("API 키 설정")
    api_status = check_api_keys()
    has_any_api = False
    for api, configured in api_status.items():
        if configured:
            print_colored(f"✅ {api} 설정됨", Colors.GREEN)
            has_any_api = True
        else:
            print_colored(f"⚠️  {api} 미설정 (선택사항)", Colors.YELLOW)
    
    if not has_any_api:
        print_colored("ℹ️  Google 번역은 API 키 없이 사용 가능", Colors.BLUE)
    
    # 5. 기능 테스트
    print_header("기능 테스트")
    
    # OCR 테스트
    print("OCR 테스트 실행 중...")
    status, msg = test_ocr_simple()
    if status:
        print_colored(f"✅ OCR 테스트: {msg}", Colors.GREEN)
    else:
        print_colored(f"❌ OCR 테스트: {msg}", Colors.RED)
    
    # PDF 테스트
    print("PDF 처리 테스트 중...")
    status, msg = test_pdf_processing()
    if status:
        print_colored(f"✅ PDF 처리: {msg}", Colors.GREEN)
    else:
        print_colored(f"❌ PDF 처리: {msg}", Colors.RED)
    
    # 번역 테스트
    print("번역 테스트 중...")
    status, msg = test_translation()
    if status:
        print_colored(f"✅ 번역 테스트: {msg}", Colors.GREEN)
    else:
        print_colored(f"⚠️  번역 테스트: {msg}", Colors.YELLOW)
    
    # 6. 결과 요약
    print_header("테스트 결과")
    
    if all_pass:
        print_colored("✅ 모든 필수 구성 요소가 정상입니다!", Colors.GREEN)
        print_colored("\n다음 명령으로 애플리케이션을 실행하세요:", Colors.BLUE)
        print_colored("  streamlit run app_ocr.py", Colors.BOLD)
    else:
        print_colored("❌ 일부 필수 구성 요소가 누락되었습니다.", Colors.RED)
        print_colored("\n해결 방법:", Colors.YELLOW)
        print("1. 필수 패키지 설치: pip install -r requirements_ocr.txt")
        print("2. Tesseract 설치: https://github.com/tesseract-ocr/tesseract")
        print("3. 환경 변수 설정: .env.example을 .env로 복사 후 수정")
    
    print("\n" + "="*60)
    print_colored("테스트 완료", Colors.BOLD)


if __name__ == "__main__":
    main()

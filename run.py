#!/usr/bin/env python3
"""
PDF GPT 번역기 - 원클릭 실행 스크립트
실행: python run.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_package(package):
    """패키지 설치 확인"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def install_packages():
    """필수 패키지 자동 설치"""
    packages = {
        'streamlit': 'streamlit',
        'pdf2zh': 'pdf2zh',
        'PyPDF2': 'PyPDF2',
        'openai': 'openai'
    }
    
    print("📦 필수 패키지 확인 중...")
    
    for name, pip_name in packages.items():
        if not check_package(name):
            print(f"  ⬇️  {pip_name} 설치 중...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name, '-q'])
            print(f"  ✅ {pip_name} 설치 완료")
        else:
            print(f"  ✅ {pip_name} 이미 설치됨")

def check_api_key():
    """API 키 확인"""
    if os.path.exists('.env'):
        from pathlib import Path
        env_content = Path('.env').read_text()
        if 'sk-' in env_content:
            print("✅ API 키 설정됨")
            return True
    
    print("\n⚠️  OpenAI API 키가 필요합니다")
    print("1. https://platform.openai.com/api-keys 에서 키 발급")
    print("2. 앱 실행 후 키 입력")
    print("")
    return False

def main():
    print("="*50)
    print("   🤖 PDF GPT 번역기 시작")
    print("="*50)
    
    # 1. 패키지 설치
    install_packages()
    
    # 2. API 키 확인 (선택)
    check_api_key()
    
    # 3. Streamlit 실행
    print("\n🚀 앱을 실행합니다...")
    print("브라우저가 자동으로 열립니다")
    print("종료: Ctrl+C")
    print("="*50)
    
    # app.py가 있는지 확인
    if not os.path.exists('app.py'):
        print("❌ app.py 파일이 없습니다!")
        print("app.py 파일을 먼저 생성해주세요")
        sys.exit(1)
    
    # Streamlit 실행
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 종료합니다")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

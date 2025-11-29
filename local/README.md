# PDF Translator Local

로컬 환경에서 실행되는 PDF 이미지 번역기입니다.

## 주요 기능

- **PaddleOCR**: 고품질 OCR로 이미지에서 텍스트 추출
- **OpenAI 번역**: GPT 모델을 사용한 자연스러운 번역
- **이미지 번역**: PDF 내 이미지의 텍스트를 직접 번역하여 교체
- **PyQt GUI**: 사용하기 쉬운 데스크톱 인터페이스

## 설치

### 1. 의존성 설치

```bash
cd local
pip install -r requirements.txt
```

### 2. PaddlePaddle 설치 (GPU 사용 시)

GPU를 사용하려면 CUDA 버전에 맞는 PaddlePaddle을 설치하세요:

```bash
# CUDA 11.x
pip install paddlepaddle-gpu

# CPU만 사용
pip install paddlepaddle
```

## 실행

```bash
python main.py
```

## 사용법

1. **PDF 열기**: `파일 > PDF 열기` 또는 "PDF 열기" 버튼 클릭
2. **API 설정**: `설정 > API 설정`에서 OpenAI API 키 입력
3. **언어 선택**: 원본 언어와 번역 대상 언어 선택
4. **페이지 범위**: 번역할 페이지 범위 지정 (선택사항)
5. **번역 시작**: "번역 시작" 버튼 클릭

## 프로젝트 구조

```
local/
├── main.py              # 앱 진입점
├── config.py            # 설정 관리
├── requirements.txt     # 의존성
│
├── core/                # 핵심 모듈
│   ├── ocr_engine.py    # PaddleOCR 기반 OCR
│   ├── translator.py    # OpenAI 번역
│   ├── pdf_processor.py # PDF 처리
│   └── image_editor.py  # 이미지 편집
│
├── gui/                 # GUI 모듈
│   ├── main_window.py   # 메인 윈도우
│   └── settings_dialog.py # 설정 다이얼로그
│
└── utils/               # 유틸리티
    └── helpers.py
```

## 설정 파일

설정은 `~/.pdf_translator_local/config.json`에 저장됩니다.

## 지원 언어

- Korean (한국어)
- English
- Japanese (日本語)
- Chinese (简体中文/繁體中文)
- Spanish, French, German, Russian, 등

## 라이선스

MIT License

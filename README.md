# PDF Translator (Local Desktop)

PyQt5 기반 로컬 PDF 번역기. [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) 프로젝트를 참고하여 개발.

## 주요 기능

- **PDF 구조 직접 추출**: OCR 없이 디지털 PDF에서 정확한 텍스트 위치 추출
- **스캔 PDF 지원**: Tesseract OCR을 통한 스캔 문서 처리
- **수식 보존**: LaTeX, 그리스 문자, 수학 기호 자동 감지 및 보존
- **번역 캐싱**: 중복 번역 방지로 API 비용 절감
- **실시간 미리보기**: 번역 진행 상황 실시간 확인
- **다국어 지원**: 영어, 한국어, 일본어, 중국어 등

## 설치

### 1. Python 패키지 설치
```bash
cd local
pip install -r requirements.txt
```

### 2. Tesseract OCR 설치 (스캔 PDF용)

**Windows:**
- https://github.com/UB-Mannheim/tesseract/wiki 에서 다운로드
- 설치 시 필요한 언어 선택 (Korean, Japanese 등)

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Linux:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-kor tesseract-ocr-jpn
```

## 실행

```bash
cd local
python main.py
```

## 사용법

1. **PDF 열기**: "PDF 열기" 버튼 또는 Ctrl+O
2. **언어 설정**: 원본/번역 언어 선택
3. **페이지 범위**: 시작/끝 페이지 지정
4. **API 설정**: 설정 메뉴에서 OpenAI API 키 입력
5. **번역 시작**: "번역 시작" 버튼 클릭

## 프로젝트 구조

```
local/
├── main.py              # 엔트리포인트
├── requirements.txt     # 의존성
├── core/
│   ├── engine.py        # 번역 엔진 (통합)
│   ├── pdf_handler.py   # PDF 처리 (PyMuPDF)
│   ├── ocr.py           # OCR (Tesseract)
│   ├── translator.py    # 번역 (OpenAI)
│   └── image_processor.py # 이미지 편집
├── gui/
│   ├── main_window.py   # 메인 윈도우
│   └── dialogs.py       # 설정 다이얼로그
└── utils/
    ├── config.py        # 설정 관리
    └── logger.py        # 로깅
```

## 기술 스택

- **GUI**: PyQt5
- **PDF 처리**: PyMuPDF (fitz)
- **OCR**: Tesseract (pytesseract)
- **이미지 처리**: Pillow, OpenCV
- **번역**: OpenAI API

## 라이선스

MIT License

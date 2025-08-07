# PDF Translator with OCR Support
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    # 기본 도구
    wget \
    curl \
    git \
    # PDF 처리
    libmupdf-dev \
    mupdf-tools \
    # 이미지 처리
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # OpenCV 의존성
    libopencv-dev \
    python3-opencv \
    # 폰트
    fonts-noto-cjk \
    fonts-liberation \
    # Tesseract OCR (선택사항)
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-kor \
    tesseract-ocr-chi-sim \
    tesseract-ocr-jpn \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# EasyOCR 모델 사전 다운로드 (빌드 시간 단축)
RUN python -c "import easyocr; reader = easyocr.Reader(['en', 'ko'], gpu=False)"

# 폰트 다운로드
RUN mkdir -p /root/.cache/pdf2zh/fonts && \
    wget -O /root/.cache/pdf2zh/fonts/GoNotoKurrent-Regular.ttf \
    https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf && \
    wget -O /root/.cache/pdf2zh/fonts/NanumGothic.ttf \
    https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf

# 애플리케이션 코드 복사
COPY app.py .
COPY pdf_ocr_translator.py .
COPY image_processor.py .
COPY config.py .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV HF_HOME=/root/.cache/huggingface

# 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 실행 명령
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# PDF Translator Docker Image
# PDFMathTranslate 수준의 완전한 PDF 번역 환경

FROM python:3.11-slim

# 메타데이터
LABEL maintainer="PDF Translator"
LABEL description="PDF 문서 번역기 - 레이아웃, 수식, 차트 완벽 보존"
LABEL version="1.0.0"

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 작업 디렉토리
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF 처리
    libmupdf-dev \
    mupdf-tools \
    # OCR (Tesseract)
    tesseract-ocr \
    tesseract-ocr-kor \
    tesseract-ocr-jpn \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-deu \
    tesseract-ocr-fra \
    tesseract-ocr-spa \
    tesseract-ocr-rus \
    # 이미지 처리
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    # 폰트
    fonts-nanum \
    fonts-noto-cjk \
    fonts-dejavu-core \
    # 빌드 도구
    build-essential \
    # 기타
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 추가 의존성 (선택적)
RUN pip install --no-cache-dir \
    anthropic \
    google-generativeai \
    flask \
    celery \
    redis \
    gunicorn \
    || true

# 애플리케이션 코드 복사
COPY . .

# 캐시 디렉토리 생성
RUN mkdir -p /root/.cache/pdf2zh /root/.cache/PDFTranslator

# 레이아웃 모델 다운로드 (선택적)
# RUN python -c "from pdf2zh.doclayout import download_model; download_model()"

# 폰트 다운로드 (선택적)
# RUN python -c "from pdf2zh.fonts import download_go_noto_fonts; download_go_noto_fonts()"

# 포트 노출
EXPOSE 5000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# 기본 명령 (REST API 서버)
CMD ["python", "pdf2zh.py", "--server", "--host", "0.0.0.0", "--port", "5000"]

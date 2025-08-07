"""
PDF 번역기 - pdf2zh 기반 Streamlit Cloud 버전
수식과 레이아웃을 보존하는 고품질 PDF 번역
이미지 내 텍스트 번역 기능 포함
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path
import time
import base64
from datetime import datetime
import logging
import subprocess
import shutil
from typing import Optional, List, Dict
import json
import requests
import asyncio

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 커스텀 모듈 import - HybridPDFTranslator 사용
from pdf_ocr_translator import HybridPDFTranslator
from image_processor import ImageProcessor
from config import TranslationConfig

# 폰트 다운로드 함수
def download_font():
    """필요한 폰트 다운로드"""
    font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    
    # GoNotoKurrent 폰트 (다국어 지원)
    font_path = font_dir / "GoNotoKurrent-Regular.ttf"
    
    if not font_path.exists():
        try:
            logger.info("폰트 다운로드 중...")
            # GitHub에서 폰트 다운로드
            url = "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"폰트 다운로드 완료: {font_path}")
            else:
                logger.error(f"폰트 다운로드 실패: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"폰트 다운로드 오류: {e}")
    
    # 한글 폰트 추가 다운로드 (이미지 번역용)
    korean_font_path = font_dir / "NanumGothic.ttf"
    if not korean_font_path.exists():
        try:
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(korean_font_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"한글 폰트 다운로드 완료: {korean_font_path}")
        except Exception as e:
            logger.error(f"한글 폰트 다운로드 오류: {e}")
    
    return str(font_path) if font_path.exists() else None

# 폰트 설정
FONT_PATH = download_font()
if FONT_PATH:
    os.environ["NOTO_FONT_PATH"] = FONT_PATH
    logger.info(f"폰트 경로 설정: {FONT_PATH}")

# HuggingFace 캐시 디렉토리 설정
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

# pdf2zh import 시도
PDF2ZH_AVAILABLE = False
MODEL_INSTANCE = None

try:
    # pdf2zh 모듈 import
    import pdf2zh
    from pdf2zh import translate
    from pdf2zh.doclayout import ModelInstance, OnnxModel
    PDF2ZH_AVAILABLE = True
    logger.info("✅ pdf2zh 모듈 로드 성공")
    
    # ONNX 모델 초기화
    try:
        logger.info("ONNX 모델 로드 중...")
        ModelInstance.value = OnnxModel.from_pretrained()
        MODEL_INSTANCE = ModelInstance.value
        logger.info("✅ ONNX 모델 로드 성공")
    except Exception as e:
        logger.error(f"❌ ONNX 모델 로드 실패: {e}")
        try:
            logger.info("대체 모델 로드 시도...")
            from pdf2zh.doclayout import DocLayoutModel
            ModelInstance.value = DocLayoutModel.load_available()
            MODEL_INSTANCE = ModelInstance.value
            logger.info("✅ 대체 모델 로드 성공")
        except Exception as e2:
            logger.error(f"❌ 대체 모델도 로드 실패: {e2}")
            
except ImportError as e:
    logger.error(f"❌ pdf2zh 모듈 로드 실패: {e}")

# Python 버전 확인
python_version = sys.version_info
st.sidebar.caption(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")

# 페이지 설정
st.set_page_config(
    page_title="PDF Math Translator Pro - pdf2zh + OCR",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #10a37f, #0d8c6f);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .success-box {
        background: #d1fae5;
        border: 1px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
    }
    .error-box {
        background: #fee2e2;
        border: 1px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991b1b;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        color: #92400e;
    }
    .info-box {
        background: #e0e7ff;
        border: 1px solid #6366f1;
        padding: 1rem;
        border-radius: 8px;
        color: #312e81;
    }
    .api-key-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .ocr-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    div[data-testid="metric-container"] {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid #667eea;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []
if 'service' not in st.session_state:
    st.session_state.service = "openai"
if 'ocr_enabled' not in st.session_state:
    st.session_state.ocr_enabled = False

def check_dependencies():
    """필수 패키지 확인"""
    dependencies = {
        'pdf2zh (Module)': PDF2ZH_AVAILABLE,
        'ONNX Model': MODEL_INSTANCE is not None,
        'Font': FONT_PATH is not None,
        'openai': False,
        'OCR (EasyOCR)': False,
        'Image Processing': False,
    }
    
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        dependencies['openai'] = False
    
    try:
        import easyocr
        dependencies['OCR (EasyOCR)'] = True
    except ImportError:
        dependencies['OCR (EasyOCR)'] = False
    
    try:
        from PIL import Image
        import cv2
        dependencies['Image Processing'] = True
    except ImportError:
        dependencies['Image Processing'] = False
    
    return dependencies

def get_pdf_page_count(file_content):
    """PDF 페이지 수 확인"""
    try:
        import PyPDF2
        from io import BytesIO
        
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return len(pdf_reader.pages)
    except Exception as e:
        logger.error(f"PDF 페이지 수 확인 오류: {e}")
        return 0

def estimate_cost(pages: int, model: str, with_ocr: bool = False) -> dict:
    """OpenAI API 비용 추정"""
    tokens_per_page = 1500
    if with_ocr:
        tokens_per_page *= 1.5  # OCR 사용 시 추가 토큰
    total_tokens = pages * tokens_per_page
    
    prices = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }
    
    if model in prices:
        input_cost = (total_tokens / 1000) * prices[model]["input"]
        output_cost = (total_tokens / 1000) * prices[model]["output"]
        total_cost = input_cost + output_cost
        
        return {
            "tokens": total_tokens,
            "cost_usd": round(total_cost, 2),
            "cost_krw": round(total_cost * 1300, 0)
        }
    
    return {"tokens": total_tokens, "cost_usd": 0, "cost_krw": 0}

def translate_with_hybrid(
    input_file: str,
    output_dir: str,
    service: str,
    lang_from: str,
    lang_to: str,
    pages: Optional[List[int]] = None,
    envs: Optional[Dict] = None,
    ocr_settings: Optional[Dict] = None,
    progress_callback=None,
    threads: int = 4,
    skip_fonts: bool = True
):
    """통합 번역기를 사용한 PDF 번역"""
    try:
        # 설정 초기화
        config = TranslationConfig(
            service=service,
            lang_from=lang_from,
            lang_to=lang_to,
            api_key=envs.get("OPENAI_API_KEY") if service == "openai" else None,
            model=envs.get("OPENAI_MODEL", "gpt-4o-mini") if service == "openai" else None,
            thread_count=threads,
            skip_subset_fonts=skip_fonts,
            use_cache=True
        )
        
        # OCR 설정 적용
        if ocr_settings:
            config.ocr_settings = ocr_settings
        
        # HybridPDFTranslator 인스턴스 생성
        translator = HybridPDFTranslator(config)
        
        # 번역 실행 (비동기를 동기로 실행)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            output_files = loop.run_until_complete(
                translator.translate_pdf_async(
                    input_file,
                    output_dir,
                    pages=pages,
                    progress_callback=progress_callback
                )
            )
        finally:
            loop.close()
        
        return True, output_files['mono'], output_files['dual'], "번역 완료"
        
    except Exception as e:
        logger.error(f"통합 번역 오류: {e}", exc_info=True)
        return False, None, None, str(e)

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📐 PDF Math Translator Pro</h1>
        <p>수식과 레이아웃을 보존하는 과학 논문 번역 + 이미지 텍스트 번역</p>
        <p>Powered by pdf2zh, OCR, and OpenAI</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        deps = check_dependencies()
        working_deps = [k for k, v in deps.items() if v]
        if working_deps:
            st.success(f"✅ 사용 가능: {', '.join(working_deps)}")
        missing = [k for k, v in deps.items() if not v]
        if missing:
            st.warning(f"⚠️ 사용 불가: {', '.join(missing)}")
    with col2:
        st.metric("번역 문서", len(st.session_state.translation_history), "📚")
    with col3:
        if st.session_state.ocr_enabled:
            st.metric("OCR", "활성화 ✅", "🔍")
        else:
            st.metric("OCR", "비활성화", "🔍")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 번역 설정")
        
        # OCR 설정 섹션
        st.markdown("""
        <div class="ocr-box">
            <h3>🔍 OCR 설정 (이미지 텍스트 번역)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        enable_ocr = st.checkbox(
            "이미지 텍스트 번역 활성화",
            value=st.session_state.ocr_enabled,
            help="PDF 내 이미지에 포함된 텍스트도 번역합니다"
        )
        st.session_state.ocr_enabled = enable_ocr
        
        ocr_settings = {}
        if enable_ocr:
            st.info("📸 이미지 내 텍스트를 감지하고 번역합니다")
            
            ocr_settings['enable_ocr'] = True
            ocr_settings['ocr_languages'] = ['en']  # 기본값
            
            ocr_settings['replace_images'] = st.checkbox(
                "번역된 이미지로 교체",
                value=True,
                help="원본 이미지를 번역된 버전으로 교체"
            )
            
            ocr_settings['overlay_text'] = st.checkbox(
                "텍스트 오버레이",
                value=True,
                help="원본 이미지 위에 번역 텍스트 표시"
            )
            
            ocr_settings['preserve_layout'] = st.checkbox(
                "레이아웃 유지",
                value=True,
                help="원본 텍스트 위치 유지"
            )
            
            ocr_quality = st.select_slider(
                "OCR 품질",
                options=["빠름", "보통", "정확"],
                value="보통",
                help="높은 품질은 더 정확하지만 시간이 오래 걸립니다"
            )
            ocr_settings['quality'] = ocr_quality
        
        # 번역 서비스 선택
        st.subheader("🌐 번역 서비스")
        
        service = st.selectbox(
            "번역 엔진",
            ["openai", "google", "deepl", "azure", "ollama"],
            index=0,
            help="OpenAI GPT가 가장 정확합니다"
        )
        st.session_state.service = service
        
        # 서비스별 설정
        envs = {}
        if service == "openai":
            st.info("🤖 OpenAI GPT - 최고 품질의 번역")
            
            api_key = st.text_input(
                "OpenAI API Key *",
                type="password",
                value=st.session_state.api_key,
                placeholder="sk-...",
                help="필수: https://platform.openai.com/api-keys"
            )
            
            if api_key:
                envs["OPENAI_API_KEY"] = api_key
                st.session_state.api_key = api_key
                st.success("✅ API 키 설정됨")
            else:
                st.error("⚠️ API 키를 입력해주세요")
            
            model = st.selectbox(
                "GPT 모델",
                ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
                index=0,
                help="gpt-4o-mini: 가성비 최고 (추천)"
            )
            envs["OPENAI_MODEL"] = model
            
        elif service == "google":
            st.success("🌍 Google 번역 - 무료, API 키 불필요")
            st.info("품질은 OpenAI보다 낮지만 무료입니다")
        
        # 언어 설정
        st.subheader("🌍 언어 설정")
        
        lang_map = {
            "영어": "en",
            "한국어": "ko", 
            "중국어(간체)": "zh",
            "중국어(번체)": "zh-TW",
            "일본어": "ja",
            "독일어": "de",
            "프랑스어": "fr",
            "스페인어": "es",
            "러시아어": "ru"
        }
        
        source_lang = st.selectbox(
            "원본 언어",
            list(lang_map.keys()),
            index=0
        )
        
        target_lang = st.selectbox(
            "번역 언어",
            list(lang_map.keys()),
            index=1
        )
        
        # 고급 옵션
        with st.expander("🔧 고급 옵션"):
            pages = st.text_input(
                "페이지 범위",
                placeholder="예: 1-10, 15",
                help="비워두면 전체 번역"
            )
            
            skip_fonts = st.checkbox(
                "폰트 서브셋 건너뛰기",
                value=True,
                help="폰트 오류 시 체크"
            )
            
            threads = st.number_input(
                "병렬 처리 스레드",
                min_value=1,
                max_value=10,
                value=4,
                help="더 많은 스레드는 더 빠르지만 API 제한에 주의"
            )
    
    # 메인 영역
    tab1, tab2, tab3, tab4 = st.tabs(["📤 번역하기", "🔍 OCR 미리보기", "📖 사용법", "ℹ️ 정보"])
    
    with tab1:
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="수식과 이미지가 포함된 과학 논문에 최적화되어 있습니다"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ 파일 준비: **{uploaded_file.name}**")
                
                # 파일 정보
                file_content = uploaded_file.getvalue()
                file_size = len(file_content) / (1024 * 1024)
                pages_count = get_pdf_page_count(file_content)
                
                # 메트릭
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("파일 크기", f"{file_size:.1f} MB")
                with col_b:
                    st.metric("페이지 수", pages_count)
                with col_c:
                    if service == "openai" and "OPENAI_MODEL" in envs:
                        cost_info = estimate_cost(pages_count, envs["OPENAI_MODEL"], enable_ocr)
                        st.metric("예상 비용", f"${cost_info['cost_usd']}")
                    else:
                        st.metric("번역 엔진", service.upper())
                with col_d:
                    st.metric("OCR", "ON" if enable_ocr else "OFF")
                
                # 비용 상세 정보
                if service == "openai" and "OPENAI_MODEL" in envs:
                    with st.expander("💰 예상 비용 상세"):
                        st.info(f"""
                        - 모델: {envs["OPENAI_MODEL"]}
                        - 예상 토큰: {cost_info['tokens']:,}개
                        - USD: ${cost_info['cost_usd']}
                        - KRW: ₩{cost_info['cost_krw']:,.0f}
                        - OCR 사용: {'예' if enable_ocr else '아니오'}
                        - 무료 크레딧 사용 시: $5에서 차감
                        """)
                
                # PDF 미리보기
                with st.expander("👁️ PDF 미리보기"):
                    pdf_display = base64.b64encode(file_content).decode()
                    pdf_html = f'''
                    <iframe 
                        src="data:application/pdf;base64,{pdf_display}" 
                        width="100%" 
                        height="600"
                        type="application/pdf">
                    </iframe>
                    '''
                    st.markdown(pdf_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 번역 실행")
                
                # 설정 요약
                st.markdown(f"""
                <div class="info-box">
                <b>설정 확인</b><br>
                • 엔진: {service.upper()}<br>
                • 언어: {source_lang} → {target_lang}<br>
                • OCR: {'활성화 ✅' if enable_ocr else '비활성화'}<br>
                • 페이지: {pages if pages else '전체'}<br>
                • 폰트: {'건너뛰기' if skip_fonts else '포함'}
                </div>
                """, unsafe_allow_html=True)
                
                # API 키 체크
                can_translate = True
                if service == "openai" and "OPENAI_API_KEY" not in envs:
                    st.error("⚠️ OpenAI API 키를 입력하세요")
                    can_translate = False
                
                # 번역 버튼
                if st.button("🚀 번역 시작", type="primary", use_container_width=True, disabled=not can_translate):
                    # 진행률 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 임시 파일 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_input:
                        tmp_input.write(file_content)
                        input_path = tmp_input.name
                    
                    output_dir = tempfile.mkdtemp()
                    
                    # 진행 상태 업데이트
                    def update_progress(value, text):
                        progress_bar.progress(value)
                        status_text.text(text)
                    
                    update_progress(0.1, "📚 PDF 분석 중...")
                    
                    # 페이지 범위 파싱
                    pages_list = None
                    if pages:
                        try:
                            pages_list = []
                            for p in pages.split(','):
                                p = p.strip()
                                if '-' in p:
                                    start, end = p.split('-')
                                    pages_list.extend(range(int(start)-1, int(end)))
                                else:
                                    pages_list.append(int(p)-1)
                        except:
                            st.error("잘못된 페이지 범위입니다")
                    
                    # 번역 실행
                    start_time = time.time()
                    update_progress(0.3, "🔄 번역 중... (시간이 걸릴 수 있습니다)")
                    
                    # 통합 번역기 사용
                    logger.info("통합 번역기로 번역 시작")
                    success, mono_file, dual_file, message = translate_with_hybrid(
                        input_path,
                        output_dir,
                        service,
                        lang_map[source_lang],
                        lang_map[target_lang],
                        pages_list,
                        envs,
                        ocr_settings if enable_ocr else {'enable_ocr': False},
                        lambda p: update_progress(0.3 + p * 0.6, f"번역 중... {int(p*100)}%"),
                        threads,
                        skip_fonts
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if success:
                        st.balloons()
                        update_progress(1.0, "✅ 번역 완료!")
                        
                        st.markdown(f"""
                        <div class="success-box">
                        🎉 <b>번역 성공!</b><br>
                        ⏱️ 소요 시간: {int(elapsed)}초<br>
                        📄 출력: 2개 파일 생성됨<br>
                        {'🔍 OCR 처리: 완료' if enable_ocr else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 다운로드 버튼
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if mono_file and os.path.exists(mono_file):
                                with open(mono_file, 'rb') as f:
                                    mono_data = f.read()
                                st.download_button(
                                    label="📥 번역본 다운로드",
                                    data=mono_data,
                                    file_name=f"{uploaded_file.name.replace('.pdf', '')}-translated.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                    type="primary"
                                )
                        
                        with col2:
                            if dual_file and os.path.exists(dual_file):
                                with open(dual_file, 'rb') as f:
                                    dual_data = f.read()
                                st.download_button(
                                    label="📥 대조본 다운로드",
                                    data=dual_data,
                                    file_name=f"{uploaded_file.name.replace('.pdf', '')}-dual.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        
                        # 기록 추가
                        st.session_state.translation_history.append({
                            'filename': uploaded_file.name,
                            'pages': pages_count,
                            'service': service,
                            'ocr': enable_ocr,
                            'time': datetime.now().strftime("%H:%M")
                        })
                        
                        # 임시 파일 정리
                        try:
                            os.unlink(input_path)
                            if mono_file and os.path.exists(mono_file):
                                os.unlink(mono_file)
                            if dual_file and os.path.exists(dual_file):
                                os.unlink(dual_file)
                            shutil.rmtree(output_dir, ignore_errors=True)
                        except Exception as e:
                            logger.warning(f"임시 파일 정리 실패: {e}")
                    else:
                        st.error("❌ 번역 실패")
                        st.markdown(f"""
                        <div class="error-box">
                        <b>오류 메시지:</b><br>
                        {message}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("🔍 디버깅 정보"):
                            st.code(message)
                            st.write("**OCR 활성화:**", enable_ocr)
                            st.write("**설정:**", ocr_settings if enable_ocr else "N/A")
    
    with tab2:
        st.markdown("### 🔍 OCR 미리보기")
        
        if uploaded_file and enable_ocr:
            st.info("📸 이미지에서 감지된 텍스트를 미리 확인할 수 있습니다")
            
            if st.button("이미지 텍스트 감지"):
                with st.spinner("이미지 분석 중..."):
                    # 임시로 PDF 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        # ImageProcessor로 이미지 추출 및 텍스트 감지
                        processor = ImageProcessor()
                        images_with_text = processor.extract_images_with_text(tmp_path)
                        
                        if images_with_text:
                            st.success(f"✅ {len(images_with_text)}개 이미지에서 텍스트 감지됨")
                            
                            for idx, img_data in enumerate(images_with_text):
                                with st.expander(f"이미지 {idx + 1}"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(img_data['image'], caption="원본 이미지", use_column_width=True)
                                    with col2:
                                        st.markdown("**감지된 텍스트:**")
                                        for text in img_data['texts']:
                                            st.write(f"- {text['text']}")
                        else:
                            st.warning("텍스트가 포함된 이미지를 찾을 수 없습니다")
                            
                    except Exception as e:
                        st.error(f"오류: {e}")
                    finally:
                        os.unlink(tmp_path)
        else:
            st.info("PDF를 업로드하고 OCR을 활성화하면 이미지 텍스트를 미리 볼 수 있습니다")
    
    with tab3:
        st.markdown("""
        ### 📖 사용 가이드
        
        #### 🆕 이미지 텍스트 번역 기능
        
        **OCR (Optical Character Recognition) 기능:**
        - 📸 PDF 내 이미지에 포함된 텍스트 자동 감지
        - 🔄 감지된 텍스트를 선택한 언어로 번역
        - 🎨 번역된 텍스트를 원본 이미지 위에 오버레이
        - 📐 원본 레이아웃과 위치 유지
        
        **지원하는 이미지 유형:**
        - 그래프와 차트의 라벨
        - 스크린샷 내 텍스트
        - 다이어그램의 설명
        - 스캔된 문서
        - 사진 속 텍스트
        
        #### 🚀 빠른 시작
        
        **1단계: OCR 활성화**
        1. 왼쪽 사이드바에서 "이미지 텍스트 번역 활성화" 체크
        2. OCR 설정 조정 (선택사항)
        
        **2단계: 번역 실행**
        1. PDF 파일 업로드
        2. 번역 언어 선택
        3. "번역 시작" 클릭
        
        #### 💡 사용 팁
        
        **OCR 품질 향상:**
        - 고해상도 PDF 사용 권장
        - 선명한 이미지일수록 정확도 향상
        - "정확" 모드는 시간이 오래 걸리지만 품질 최고
        
        **최적 설정:**
        - 과학 논문: OCR + 레이아웃 유지
        - 프레젠테이션: OCR + 텍스트 오버레이
        - 스캔 문서: OCR + 이미지 교체
        
        #### ⚠️ 주의사항
        
        **OCR 한계:**
        - 손글씨는 인식률이 낮을 수 있음
        - 복잡한 수식은 텍스트로만 처리
        - 장식 폰트는 인식 오류 가능
        
        **성능 고려사항:**
        - OCR 사용 시 처리 시간 증가
        - 이미지가 많을수록 시간 소요
        - API 비용 약 1.5배 증가
        """)
    
    with tab4:
        st.markdown("""
        ### ℹ️ PDF Math Translator Pro 정보
        
        **버전**: pdf2zh 1.9.0+ with OCR Enhancement  
        **기본 엔진**: OpenAI GPT-4o-mini + EasyOCR  
        **개발**: Enhanced by Streamlit Community  
        
        #### 🔍 OCR 기술 스택
        
        **이미지 처리:**
        - **OCR 엔진**: EasyOCR (80+ 언어 지원)
        - **이미지 처리**: OpenCV + PIL
        - **레이아웃 분석**: CRAFT 텍스트 감지
        - **폰트 렌더링**: FreeType + HarfBuzz
        
        **번역 파이프라인:**
        1. PDF → 이미지 추출 (PyMuPDF)
        2. 텍스트 감지 (EasyOCR)
        3. 번역 (OpenAI/Google)
        4. 이미지 재생성 (PIL + OpenCV)
        5. PDF 재구성 (PyMuPDF)
        
        #### 📊 성능 비교
        
        | 기능 | 기본 모드 | OCR 모드 |
        |------|----------|----------|
        | 텍스트 레이어 | ✅ | ✅ |
        | 수식 | ✅ | ✅ |
        | 이미지 텍스트 | ❌ | ✅ |
        | 그래프 라벨 | ❌ | ✅ |
        | 처리 속도 | 빠름 | 보통 |
        | API 비용 | 기본 | 1.5x |
        
        #### 🛠️ 주요 기능
        
        **pdf2zh 기본 기능:**
        - 📐 수식 완벽 보존
        - 📑 레이아웃 유지
        - 🔤 폰트 보존
        - 📊 도표 위치 유지
        
        **OCR 추가 기능:**
        - 🖼️ 이미지 텍스트 번역
        - 📈 그래프 라벨 번역
        - 📸 스크린샷 번역
        - 🎨 텍스트 오버레이
        
        #### 🔗 관련 링크
        - [GitHub: PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
        - [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
        - [OpenAI Platform](https://platform.openai.com)
        - [온라인 데모](https://pdf2zh.com)
        
        #### 📝 라이선스
        - pdf2zh: AGPL-3.0
        - EasyOCR: Apache 2.0
        - OpenAI API: 상용 라이선스
        - 번역 결과물: 사용자 소유
        
        #### 🙏 감사의 말
        pdf2zh 개발팀, EasyOCR 팀, OpenAI에 감사드립니다.
        오픈소스 커뮤니티의 기여로 발전하고 있습니다.
        """)

if __name__ == "__main__":
    main()

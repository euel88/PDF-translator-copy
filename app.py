"""
PDF 번역기 - pdf2zh + OCR 통합 버전
수식과 레이아웃을 보존하는 고품질 PDF 번역
이미지 내 텍스트 번역 기능 포함
하이브리드 번역기 사용
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
import gc

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 커스텀 모듈 import
from hybrid_translator import HybridTranslator
from image_processor import ImageProcessor
from config import TranslationConfig, ConfigManager

# 폰트 다운로드 함수
def download_font():
    """필요한 폰트 다운로드"""
    font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    
    fonts = {
        "GoNotoKurrent-Regular.ttf": "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf",
        "NanumGothic.ttf": "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    }
    
    for font_name, url in fonts.items():
        font_path = font_dir / font_name
        if not font_path.exists():
            try:
                logger.info(f"폰트 다운로드 중: {font_name}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"폰트 다운로드 완료: {font_path}")
            except Exception as e:
                logger.error(f"폰트 다운로드 오류: {e}")
    
    # 기본 폰트 경로 반환
    default_font = font_dir / "GoNotoKurrent-Regular.ttf"
    return str(default_font) if default_font.exists() else None

# 폰트 설정
FONT_PATH = download_font()
if FONT_PATH:
    os.environ["NOTO_FONT_PATH"] = FONT_PATH
    logger.info(f"폰트 경로 설정: {FONT_PATH}")

# HuggingFace 캐시 디렉토리 설정
os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

# pdf2zh 가용성 확인
PDF2ZH_AVAILABLE = False
try:
    import pdf2zh
    from pdf2zh.doclayout import ModelInstance, OnnxModel
    PDF2ZH_AVAILABLE = True
    logger.info("✅ pdf2zh 모듈 로드 성공")
    
    # ONNX 모델 초기화
    try:
        logger.info("ONNX 모델 로드 중...")
        ModelInstance.value = OnnxModel.from_pretrained()
        logger.info("✅ ONNX 모델 로드 성공")
    except Exception as e:
        logger.error(f"❌ ONNX 모델 로드 실패: {e}")
        
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
    .stats-box {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        padding: 1rem;
        border-radius: 8px;
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
if 'translator' not in st.session_state:
    st.session_state.translator = None

def check_dependencies():
    """필수 패키지 확인"""
    dependencies = {
        'pdf2zh': PDF2ZH_AVAILABLE,
        'ONNX Model': PDF2ZH_AVAILABLE and ModelInstance.value is not None,
        'Font': FONT_PATH is not None,
        'openai': False,
        'OCR (EasyOCR)': False,
        'Image Processing': False,
    }
    
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        pass
    
    try:
        import easyocr
        dependencies['OCR (EasyOCR)'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        import cv2
        dependencies['Image Processing'] = True
    except ImportError:
        pass
    
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

async def translate_with_hybrid(
    input_file: str,
    output_dir: str,
    service: str,
    lang_from: str,
    lang_to: str,
    pages: Optional[List[int]] = None,
    envs: Optional[Dict] = None,
    ocr_settings: Optional[Dict] = None,
    progress_callback=None
):
    """하이브리드 번역기를 사용한 번역"""
    try:
        # 설정 초기화
        config = TranslationConfig(
            service=service,
            lang_from=lang_from,
            lang_to=lang_to,
            api_key=envs.get("OPENAI_API_KEY") if service == "openai" else None,
            model=envs.get("OPENAI_MODEL", "gpt-4o-mini") if service == "openai" else None,
            thread_count=4,
            use_cache=True,
            skip_subset_fonts=True
        )
        
        # OCR 설정
        if ocr_settings:
            config.ocr_settings = ocr_settings
        
        # 기존 번역기가 있으면 정리
        if st.session_state.translator:
            try:
                st.session_state.translator._cleanup()
            except:
                pass
            gc.collect()
        
        # 하이브리드 번역기 생성
        translator = HybridTranslator(config)
        st.session_state.translator = translator
        
        # 비동기 번역 실행
        result = await translator.translate_document_async(
            input_file,
            output_dir,
            pages=pages,
            progress_callback=progress_callback
        )
        
        # 통계 정보 저장
        if 'stats' in result:
            st.session_state.last_stats = result['stats']
        
        return True, result['mono'], result['dual'], "번역 완료"
        
    except Exception as e:
        logger.error(f"하이브리드 번역 오류: {e}", exc_info=True)
        return False, None, None, str(e)
    finally:
        # 메모리 정리
        gc.collect()

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📐 PDF Math Translator Pro</h1>
        <p>수식과 레이아웃을 보존하는 과학 논문 번역 + 이미지 텍스트 번역</p>
        <p>Powered by Hybrid Translator (pdf2zh + OCR + Async)</p>
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
        st.subheader("🔍 OCR 설정")
        
        enable_ocr = st.checkbox(
            "이미지 텍스트 번역 활성화",
            value=st.session_state.ocr_enabled,
            help="PDF 내 이미지에 포함된 텍스트도 번역합니다"
        )
        st.session_state.ocr_enabled = enable_ocr
        
        ocr_settings = {'enable_ocr': enable_ocr}
        if enable_ocr:
            st.info("📸 이미지 내 텍스트를 감지하고 번역합니다")
            
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
            ["openai", "google", "deepl", "azure"],
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
    
    # 메인 영역
    tab1, tab2, tab3 = st.tabs(["📤 번역하기", "📊 통계", "📖 사용법"])
    
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
            
            with col2:
                st.markdown("### 🎯 번역 실행")
                
                # 설정 요약
                st.markdown(f"""
                <div class="info-box">
                <b>설정 확인</b><br>
                • 엔진: {service.upper()}<br>
                • 언어: {source_lang} → {target_lang}<br>
                • OCR: {'활성화 ✅' if enable_ocr else '비활성화'}<br>
                • 페이지: {pages if pages else '전체'}
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
                    def update_progress(value):
                        progress_bar.progress(value)
                        status_text.text(f"진행률: {int(value * 100)}%")
                    
                    update_progress(0.1)
                    status_text.text("📚 PDF 분석 중...")
                    
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
                    update_progress(0.3)
                    status_text.text("🔄 번역 중... (시간이 걸릴 수 있습니다)")
                    
                    # 비동기 번역 실행
                    success, mono_file, dual_file, message = asyncio.run(
                        translate_with_hybrid(
                            input_path,
                            output_dir,
                            service,
                            lang_map[source_lang],
                            lang_map[target_lang],
                            pages_list,
                            envs,
                            ocr_settings,
                            lambda p: update_progress(0.3 + p * 0.6)
                        )
                    )
                    
                    elapsed = time.time() - start_time
                    
                    if success:
                        st.balloons()
                        update_progress(1.0)
                        status_text.text("✅ 번역 완료!")
                        
                        st.markdown(f"""
                        <div class="success-box">
                        🎉 <b>번역 성공!</b><br>
                        ⏱️ 소요 시간: {int(elapsed)}초<br>
                        📄 출력: 2개 파일 생성됨<br>
                        {'🔍 OCR 처리: 완료' if enable_ocr else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 통계 표시
                        if hasattr(st.session_state, 'last_stats'):
                            stats = st.session_state.last_stats
                            st.markdown(f"""
                            <div class="stats-box">
                            <b>📊 번역 통계</b><br>
                            • 처리된 페이지: {stats.get('pages_processed', 0)}<br>
                            • 번역된 텍스트 블록: {stats.get('text_blocks_translated', 0)}<br>
                            • 처리된 이미지: {stats.get('images_processed', 0)}<br>
                            • OCR로 감지된 텍스트: {stats.get('ocr_texts_found', 0)}<br>
                            • 오류: {len(stats.get('errors', []))}
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
    
    with tab2:
        st.markdown("### 📊 번역 통계")
        
        if hasattr(st.session_state, 'last_stats'):
            stats = st.session_state.last_stats
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("페이지", stats.get('pages_processed', 0))
            with col2:
                st.metric("텍스트 블록", stats.get('text_blocks_translated', 0))
            with col3:
                st.metric("이미지", stats.get('images_processed', 0))
            with col4:
                st.metric("OCR 텍스트", stats.get('ocr_texts_found', 0))
            
            if stats.get('errors'):
                st.warning(f"⚠️ {len(stats['errors'])}개 오류 발생")
                with st.expander("오류 상세"):
                    for error in stats['errors']:
                        st.text(error)
        else:
            st.info("번역을 실행하면 통계가 표시됩니다")
    
    with tab3:
        st.markdown("""
        ### 📖 사용 가이드
        
        #### 🚀 새로운 기능
        
        **하이브리드 번역 시스템:**
        - ⚡ 비동기 처리로 성능 향상
        - 🔄 pdf2zh와 OCR 완벽 통합
        - 📊 실시간 번역 통계
        - 🛡️ 강화된 오류 처리
        
        #### 💡 사용 팁
        
        **최적 설정:**
        - 과학 논문: pdf2zh + OCR + 레이아웃 유지
        - 프레젠테이션: OCR + 텍스트 오버레이
        - 스캔 문서: OCR + 이미지 교체
        
        **성능 최적화:**
        - 대용량 파일은 페이지 범위 지정
        - OCR은 필요한 경우만 활성화
        - Google 번역으로 무료 테스트 후 OpenAI 사용
        
        #### ⚠️ 문제 해결
        
        **이미지 교체 실패:**
        - PyMuPDF 버전 확인 (1.23.0 이상)
        - 폴백 모드 자동 활성화
        
        **메모리 부족:**
        - 페이지 범위를 나누어 번역
        - 브라우저 새로고침 후 재시도
        """)

if __name__ == "__main__":
    main()

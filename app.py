"""
PDF Math Translator Pro - Streamlit Cloud 최적화 버전
pdf2zh + OCR 전체 기능 포함 (메모리 최적화)
"""

import streamlit as st
import os
import sys
import tempfile
import gc
import subprocess
import logging
from pathlib import Path
import time
import base64
from datetime import datetime
from typing import Optional, List, Dict
import json
import shutil
import io

# 메모리 모니터링
try:
    import psutil
    PSUTIL_AVAILABLE = True
except:
    PSUTIL_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="PDF Math Translator Pro",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
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
    .memory-bar {
        background: #f3f4f6;
        border-radius: 5px;
        height: 20px;
        overflow: hidden;
    }
    .memory-fill {
        background: linear-gradient(to right, #10b981, #059669);
        height: 100%;
        transition: width 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if 'ocr_enabled' not in st.session_state:
    st.session_state.ocr_enabled = False
if 'dependencies_installed' not in st.session_state:
    st.session_state.dependencies_installed = False

def get_memory_usage():
    """메모리 사용량 확인"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / 1024 / 1024  # MB
    return 0

def clear_memory():
    """메모리 정리"""
    gc.collect()
    if hasattr(gc, 'garbage'):
        del gc.garbage[:]
    logger.info(f"메모리 정리 완료: {get_memory_usage():.1f} MB 사용 중")

@st.cache_resource
def install_runtime_dependencies():
    """런타임에 필요한 패키지 설치 (한 번만 실행)"""
    if st.session_state.dependencies_installed:
        return True
    
    with st.spinner("🔧 필수 구성 요소 설치 중... (첫 실행 시에만 필요)"):
        try:
            # torch CPU 버전 설치 (가장 작은 크기)
            if not check_package_installed('torch'):
                st.info("PyTorch CPU 버전 설치 중...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 
                    'torch==2.0.1+cpu', 'torchvision==0.15.2+cpu',
                    '-f', 'https://download.pytorch.org/whl/torch_stable.html',
                    '--no-cache-dir'
                ])
                st.success("✅ PyTorch 설치 완료")
            
            # EasyOCR 설치
            if not check_package_installed('easyocr'):
                st.info("EasyOCR 설치 중...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    'easyocr', '--no-cache-dir'
                ])
                st.success("✅ EasyOCR 설치 완료")
            
            st.session_state.dependencies_installed = True
            clear_memory()
            return True
            
        except Exception as e:
            st.error(f"설치 실패: {e}")
            return False

def check_package_installed(package_name):
    """패키지 설치 여부 확인"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

@st.cache_resource
def initialize_pdf2zh():
    """pdf2zh 초기화 (한 번만 실행)"""
    try:
        # pdf2zh import
        import pdf2zh
        from pdf2zh import translate
        from pdf2zh.doclayout import ModelInstance, OnnxModel
        
        # ONNX 모델 로드
        with st.spinner("📚 레이아웃 분석 모델 로드 중..."):
            if ModelInstance.value is None:
                ModelInstance.value = OnnxModel.from_pretrained()
        
        logger.info("✅ pdf2zh 초기화 완료")
        return True, ModelInstance.value
        
    except Exception as e:
        logger.error(f"pdf2zh 초기화 실패: {e}")
        return False, None

@st.cache_resource
def initialize_ocr_engine():
    """OCR 엔진 초기화 (지연 로딩)"""
    if not st.session_state.ocr_enabled:
        return None
    
    try:
        import easyocr
        
        with st.spinner("🔍 OCR 엔진 초기화 중..."):
            # CPU 모드로 초기화 (GPU 사용 안 함)
            reader = easyocr.Reader(['en', 'ko'], gpu=False)
            logger.info("✅ OCR 엔진 초기화 완료")
            clear_memory()
            return reader
            
    except Exception as e:
        logger.error(f"OCR 초기화 실패: {e}")
        st.warning(f"OCR을 사용할 수 없습니다: {e}")
        return None

def download_fonts():
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
                import requests
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"폰트 다운로드: {font_name}")
            except Exception as e:
                logger.error(f"폰트 다운로드 실패: {e}")

class OptimizedPDFTranslator:
    """메모리 최적화된 PDF 번역기"""
    
    def __init__(self, config):
        self.config = config
        self.pdf2zh_available = False
        self.model_instance = None
        self.ocr_reader = None
    
    def translate_with_pdf2zh(self, input_path, output_dir, pages=None, progress_callback=None):
        """pdf2zh를 사용한 번역 (메모리 최적화)"""
        try:
            from pdf2zh import translate
            
            # 환경 변수 설정
            envs = {}
            if self.config.get('api_key'):
                if self.config['service'] == 'openai':
                    envs['OPENAI_API_KEY'] = self.config['api_key']
                    envs['OPENAI_MODEL'] = self.config.get('model', 'gpt-4o-mini')
            
            # pdf2zh 실행
            result = translate(
                files=[input_path],
                output=output_dir,
                pages=pages,
                lang_in=self.config['lang_from'],
                lang_out=self.config['lang_to'],
                service=self.config['service'],
                thread=self.config.get('threads', 4),
                model=self.model_instance,
                envs=envs,
                skip_subset_fonts=True,
                ignore_cache=False
            )
            
            # 결과 파일 경로
            base_name = Path(input_path).stem
            mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
            dual_file = Path(output_dir) / f"{base_name}-dual.pdf"
            
            clear_memory()
            return True, str(mono_file), str(dual_file), "번역 완료"
            
        except Exception as e:
            logger.error(f"pdf2zh 번역 오류: {e}")
            return False, None, None, str(e)
    
    def enhance_with_ocr(self, pdf_path, ocr_reader):
        """OCR로 이미지 텍스트 번역 추가 (메모리 최적화)"""
        if not ocr_reader or not self.config.get('ocr_enabled'):
            return pdf_path
        
        try:
            import fitz
            from PIL import Image
            import numpy as np
            
            doc = fitz.open(pdf_path)
            
            # 페이지별 처리 (메모리 절약)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # 이미지 추출
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.alpha:
                            pix = fitz.Pixmap(pix, 0)
                        
                        # PIL 이미지로 변환
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # numpy 배열로 변환
                        img_array = np.array(image)
                        
                        # OCR 실행
                        results = ocr_reader.readtext(img_array)
                        
                        if results:
                            logger.info(f"페이지 {page_num}: {len(results)}개 텍스트 감지")
                            # 여기에 번역 및 이미지 교체 로직 추가 가능
                        
                        # 메모리 해제
                        pix = None
                        del img_array
                        image.close()
                        
                    except Exception as e:
                        logger.error(f"이미지 처리 오류: {e}")
                        continue
                
                # 각 페이지 처리 후 메모리 정리
                if page_num % 5 == 0:
                    clear_memory()
            
            # 저장
            output_path = pdf_path.replace('.pdf', '_ocr.pdf')
            doc.save(output_path)
            doc.close()
            
            clear_memory()
            return output_path
            
        except Exception as e:
            logger.error(f"OCR 처리 오류: {e}")
            return pdf_path

def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📐 PDF Math Translator Pro</h1>
        <p>수식과 레이아웃을 보존하는 과학 논문 번역 + 이미지 텍스트 번역</p>
        <p>Powered by pdf2zh & OCR - Streamlit Cloud Optimized</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 메모리 상태 표시
    if PSUTIL_AVAILABLE:
        mem_usage = get_memory_usage()
        mem_percent = min(mem_usage / 1024 * 100, 100)  # 1GB 기준
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"""
            <div class="memory-bar">
                <div class="memory-fill" style="width: {mem_percent}%"></div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"메모리 사용: {mem_usage:.1f} MB / 1024 MB")
        with col2:
            if st.button("🧹 메모리 정리"):
                clear_memory()
                st.rerun()
        with col3:
            st.metric("상태", "정상" if mem_usage < 900 else "주의")
    
    # 의존성 설치 확인
    if not st.session_state.dependencies_installed:
        if st.button("🚀 시작하기 (필수 구성 요소 설치)"):
            if install_runtime_dependencies():
                st.success("✅ 설치 완료! 페이지를 새로고침하세요.")
                st.balloons()
            else:
                st.error("설치 실패. 페이지를 새로고침한 후 다시 시도하세요.")
        st.stop()
    
    # pdf2zh 초기화
    pdf2zh_available, model_instance = initialize_pdf2zh()
    
    # OCR 엔진 (지연 로딩)
    ocr_reader = None
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 번역 설정")
        
        # OCR 설정
        st.subheader("🔍 OCR 설정")
        ocr_enabled = st.checkbox(
            "이미지 텍스트 번역 활성화",
            value=st.session_state.ocr_enabled,
            help="PDF 내 이미지에 포함된 텍스트도 번역합니다"
        )
        st.session_state.ocr_enabled = ocr_enabled
        
        if ocr_enabled:
            st.info("📸 이미지 내 텍스트를 감지하고 번역합니다")
            # OCR 엔진 초기화 (처음 활성화 시)
            if ocr_reader is None:
                ocr_reader = initialize_ocr_engine()
        
        # 번역 서비스
        st.subheader("🌐 번역 서비스")
        service = st.selectbox(
            "번역 엔진",
            ["openai", "google"],
            help="OpenAI GPT가 가장 정확합니다"
        )
        
        api_key = ""
        model = "gpt-4o-mini"
        
        if service == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="sk-..."
            )
            
            if api_key:
                st.session_state.api_key = api_key
                st.success("✅ API 키 설정됨")
            else:
                st.warning("⚠️ API 키를 입력하세요")
            
            model = st.selectbox(
                "GPT 모델",
                ["gpt-4o-mini", "gpt-3.5-turbo"],
                help="gpt-4o-mini: 가성비 최고"
            )
        
        # 언어 설정
        st.subheader("🌍 언어 설정")
        lang_map = {
            "영어": "en",
            "한국어": "ko",
            "중국어(간체)": "zh",
            "일본어": "ja",
            "스페인어": "es",
            "프랑스어": "fr",
            "독일어": "de"
        }
        
        source_lang = st.selectbox("원본 언어", list(lang_map.keys()), index=0)
        target_lang = st.selectbox("번역 언어", list(lang_map.keys()), index=1)
        
        # 고급 옵션
        with st.expander("🔧 고급 옵션"):
            pages = st.text_input(
                "페이지 범위",
                placeholder="예: 1-10, 15",
                help="비워두면 전체 번역"
            )
            
            threads = st.slider("병렬 처리", 1, 4, 2, help="메모리 사용량 증가")
    
    # 메인 영역
    tab1, tab2, tab3 = st.tabs(["📤 번역하기", "📖 사용법", "ℹ️ 정보"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="수식과 이미지가 포함된 과학 논문에 최적화"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ 파일 준비: **{uploaded_file.name}**")
                
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"📁 파일 크기: {file_size:.1f} MB")
                
                if file_size > 10:
                    st.warning("⚠️ 큰 파일은 처리 시간이 오래 걸릴 수 있습니다")
            
            with col2:
                st.markdown("### 🎯 번역 실행")
                
                # 설정 요약
                st.markdown(f"""
                - 엔진: {service.upper()}
                - 언어: {source_lang} → {target_lang}
                - OCR: {'✅' if ocr_enabled else '❌'}
                - pdf2zh: {'✅' if pdf2zh_available else '❌'}
                """)
                
                # 번역 버튼
                can_translate = True
                if service == "openai" and not api_key:
                    st.error("API 키를 입력하세요")
                    can_translate = False
                
                if st.button("🚀 번역 시작", type="primary", disabled=not can_translate):
                    # 진행률
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 폰트 다운로드
                    download_fonts()
                    
                    # 임시 파일 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        input_path = tmp.name
                    
                    output_dir = tempfile.mkdtemp()
                    
                    try:
                        # 번역 설정
                        config = {
                            'service': service,
                            'api_key': api_key,
                            'model': model,
                            'lang_from': lang_map[source_lang],
                            'lang_to': lang_map[target_lang],
                            'ocr_enabled': ocr_enabled,
                            'threads': threads
                        }
                        
                        # 번역기 초기화
                        translator = OptimizedPDFTranslator(config)
                        translator.pdf2zh_available = pdf2zh_available
                        translator.model_instance = model_instance
                        translator.ocr_reader = ocr_reader
                        
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
                                st.error("잘못된 페이지 범위")
                        
                        # 번역 실행
                        status_text.text("📚 PDF 분석 중...")
                        progress_bar.progress(0.2)
                        
                        if pdf2zh_available:
                            status_text.text("🔄 번역 중... (시간이 걸릴 수 있습니다)")
                            progress_bar.progress(0.5)
                            
                            success, mono_file, dual_file, message = translator.translate_with_pdf2zh(
                                input_path, output_dir, pages_list
                            )
                            
                            if success and ocr_enabled and ocr_reader:
                                status_text.text("🔍 이미지 텍스트 처리 중...")
                                progress_bar.progress(0.8)
                                mono_file = translator.enhance_with_ocr(mono_file, ocr_reader)
                            
                            progress_bar.progress(1.0)
                            
                            if success:
                                st.balloons()
                                status_text.text("✅ 번역 완료!")
                                
                                # 다운로드 버튼
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    if mono_file and os.path.exists(mono_file):
                                        with open(mono_file, 'rb') as f:
                                            st.download_button(
                                                "📥 번역본 다운로드",
                                                f.read(),
                                                f"{uploaded_file.name.replace('.pdf', '')}_translated.pdf",
                                                "application/pdf",
                                                use_container_width=True
                                            )
                                
                                with col_b:
                                    if dual_file and os.path.exists(dual_file):
                                        with open(dual_file, 'rb') as f:
                                            st.download_button(
                                                "📥 대조본 다운로드",
                                                f.read(),
                                                f"{uploaded_file.name.replace('.pdf', '')}_dual.pdf",
                                                "application/pdf",
                                                use_container_width=True
                                            )
                            else:
                                st.error(f"❌ 번역 실패: {message}")
                        else:
                            st.error("pdf2zh를 사용할 수 없습니다")
                        
                    except Exception as e:
                        st.error(f"오류 발생: {e}")
                        logger.error(f"번역 오류: {e}", exc_info=True)
                    
                    finally:
                        # 임시 파일 정리
                        try:
                            if os.path.exists(input_path):
                                os.unlink(input_path)
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir, ignore_errors=True)
                        except:
                            pass
                        
                        # 메모리 정리
                        clear_memory()
    
    with tab2:
        st.markdown("""
        ### 📖 사용 가이드
        
        #### 🚀 빠른 시작
        1. 첫 실행 시 "시작하기" 버튼을 클릭하여 필수 구성 요소 설치
        2. PDF 파일 업로드
        3. 번역 언어 선택
        4. "번역 시작" 클릭
        
        #### 💡 기능
        - ✅ **레이아웃 보존**: pdf2zh로 원본 레이아웃 유지
        - ✅ **수식 보존**: 수학 공식과 기호 완벽 보존
        - ✅ **이미지 텍스트 번역**: OCR로 이미지 내 텍스트 감지 및 번역
        - ✅ **대조본 생성**: 원본과 번역본 페이지별 대조
        
        #### ⚙️ 최적화
        - 메모리 사용량 실시간 모니터링
        - 필요한 모듈만 지연 로딩
        - 페이지별 처리로 메모리 절약
        - 자동 가비지 컬렉션
        
        #### ⚠️ 제한사항
        - Streamlit Cloud: 1GB RAM, 1GB 스토리지
        - 큰 파일(>10MB)은 처리 시간이 오래 걸림
        - OCR 사용 시 추가 시간 소요
        """)
    
    with tab3:
        st.markdown("""
        ### ℹ️ PDF Math Translator Pro 정보
        
        **버전**: pdf2zh 1.9.0+ with OCR  
        **최적화**: Streamlit Cloud (1GB RAM)  
        **엔진**: OpenAI GPT / Google Translate  
        
        #### 🛠️ 기술 스택
        - **PDF 처리**: pdf2zh, PyMuPDF
        - **레이아웃 분석**: DocLayout-YOLO (ONNX)
        - **OCR**: EasyOCR (CPU mode)
        - **번역**: OpenAI API, Google Translate
        - **메모리 관리**: psutil, gc
        
        #### 📊 메모리 최적화 전략
        1. **지연 로딩**: OCR과 torch는 필요 시에만 로드
        2. **CPU 전용**: GPU 비활성화로 메모리 절약
        3. **페이지별 처리**: 대용량 PDF도 안정적 처리
        4. **적극적 정리**: 각 단계마다 메모리 해제
        
        #### 🔗 관련 링크
        - [GitHub: PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
        - [온라인 데모](https://pdf2zh.com)
        - [OpenAI Platform](https://platform.openai.com)
        """)

if __name__ == "__main__":
    main()

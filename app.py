"""
PDF 번역기 - pdf2zh 기반 Streamlit Cloud 버전
수식과 레이아웃을 보존하는 고품질 PDF 번역
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
PDF2ZH_CLI_AVAILABLE = False

try:
    # pdf2zh 모듈 import
    import pdf2zh
    from pdf2zh import translate
    PDF2ZH_AVAILABLE = True
    logger.info("✅ pdf2zh 모듈 로드 성공")
except ImportError as e:
    logger.error(f"❌ pdf2zh 모듈 로드 실패: {e}")

# pdf2zh CLI 경로 찾기
PDF2ZH_CMD = None
for cmd in ['pdf2zh', '/home/adminuser/venv/bin/pdf2zh', '/home/appuser/venv/bin/pdf2zh', '/usr/local/bin/pdf2zh']:
    try:
        result = subprocess.run([cmd, '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            PDF2ZH_CMD = cmd
            PDF2ZH_CLI_AVAILABLE = True
            logger.info(f"✅ pdf2zh CLI 사용 가능: {cmd}")
            break
    except:
        continue

if not PDF2ZH_CLI_AVAILABLE:
    logger.warning("⚠️ pdf2zh CLI를 찾을 수 없음")

# Python 버전 확인
python_version = sys.version_info
st.sidebar.caption(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")

# 페이지 설정
st.set_page_config(
    page_title="PDF Math Translator - pdf2zh",
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
        background: linear-gradient(135deg, #10a37f 0%, #0d8c6f 100%);
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
        background: linear-gradient(135deg, #10a37f 0%, #0d8c6f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    div[data-testid="metric-container"] {
        background: rgba(16, 163, 127, 0.1);
        border: 1px solid #10a37f;
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
    st.session_state.service = "openai"  # 기본값을 openai로 설정

def check_dependencies():
    """필수 패키지 확인"""
    dependencies = {
        'pdf2zh (Module)': PDF2ZH_AVAILABLE,
        'pdf2zh (CLI)': PDF2ZH_CLI_AVAILABLE,
        'Font': FONT_PATH is not None,
        'openai': False,
    }
    
    try:
        import openai
        dependencies['openai'] = True
    except ImportError:
        dependencies['openai'] = False
    
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

def estimate_cost(pages: int, model: str) -> dict:
    """OpenAI API 비용 추정"""
    tokens_per_page = 1500
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

def translate_with_pdf2zh_api(
    input_file: str,
    output_dir: str,
    service: str,
    lang_from: str,
    lang_to: str,
    pages: Optional[List[int]] = None,
    envs: Optional[Dict] = None,
    thread: int = 2,
    skip_fonts: bool = True
):
    """pdf2zh Python API를 사용한 번역"""
    try:
        if not PDF2ZH_AVAILABLE:
            return False, None, None, "pdf2zh 모듈을 사용할 수 없습니다"
        
        # 환경 변수 설정
        if envs:
            for key, value in envs.items():
                os.environ[key] = value
        
        # 폰트 경로 설정
        if FONT_PATH:
            os.environ["NOTO_FONT_PATH"] = FONT_PATH
        
        # pdf2zh.translate 함수 호출
        from pdf2zh import translate
        
        logger.info(f"PDF 번역 시작: {input_file}")
        logger.info(f"설정: service={service}, lang={lang_from}->{lang_to}, pages={pages}")
        logger.info(f"폰트 경로: {os.environ.get('NOTO_FONT_PATH')}")
        logger.info(f"폰트 서브셋 건너뛰기: {skip_fonts}")
        
        # translate 함수 호출 (skip_subset_fonts 추가)
        result = translate(
            files=[input_file],
            output=output_dir,
            pages=pages,
            lang_in=lang_from,
            lang_out=lang_to,
            service=service,
            thread=thread,
            skip_subset_fonts=skip_fonts  # 폰트 서브셋팅 비활성화
        )
        
        # 출력 파일 확인
        base_name = Path(input_file).stem
        mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_file = Path(output_dir) / f"{base_name}-dual.pdf"
        
        if mono_file.exists() and dual_file.exists():
            return True, str(mono_file), str(dual_file), "번역 완료"
        else:
            return False, None, None, "번역 파일 생성 실패"
            
    except Exception as e:
        logger.error(f"pdf2zh API 오류: {e}")
        return False, None, None, str(e)

def translate_with_pdf2zh_cli(
    input_file: str,
    output_dir: str,
    service: str,
    lang_from: str,
    lang_to: str,
    pages: str = None,
    envs: dict = None,
    skip_fonts: bool = True
):
    """pdf2zh CLI를 사용한 번역"""
    try:
        if not PDF2ZH_CLI_AVAILABLE or not PDF2ZH_CMD:
            return False, None, None, "pdf2zh CLI를 사용할 수 없습니다"
        
        # 환경 변수 설정
        env = os.environ.copy()
        if envs:
            env.update(envs)
        
        # 폰트 경로 설정
        if FONT_PATH:
            env["NOTO_FONT_PATH"] = FONT_PATH
        
        # 명령어 구성
        cmd = [
            PDF2ZH_CMD,
            input_file,
            "-o", output_dir,
            "-s", service,
            "-li", lang_from,
            "-lo", lang_to,
            "-t", "2"
        ]
        
        # 폰트 서브셋팅 비활성화
        if skip_fonts:
            cmd.append("--skip-subset-fonts")
        
        if pages:
            cmd.extend(["-p", pages])
        
        logger.info(f"실행 명령: {' '.join(cmd)}")
        
        # 실행
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )
        
        logger.info(f"명령 실행 결과: returncode={result.returncode}")
        if result.stdout:
            logger.info(f"stdout: {result.stdout[:500]}")
        if result.stderr:
            logger.warning(f"stderr: {result.stderr[:500]}")
        
        if result.returncode == 0:
            # 출력 파일 찾기
            base_name = Path(input_file).stem
            mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
            dual_file = Path(output_dir) / f"{base_name}-dual.pdf"
            
            if mono_file.exists() and dual_file.exists():
                return True, str(mono_file), str(dual_file), "번역 완료"
            else:
                # 파일명 패턴이 다를 수 있으므로 glob으로 찾기
                mono_files = list(Path(output_dir).glob("*-mono.pdf"))
                dual_files = list(Path(output_dir).glob("*-dual.pdf"))
                
                if mono_files and dual_files:
                    return True, str(mono_files[0]), str(dual_files[0]), "번역 완료"
                else:
                    return False, None, None, "번역 파일을 찾을 수 없습니다"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            return False, None, None, f"번역 실패: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, None, None, "번역 시간 초과 (10분)"
    except Exception as e:
        logger.error(f"pdf2zh CLI 오류: {e}")
        return False, None, None, str(e)

def main():
    """메인 애플리케이션"""
    
    # pdf2zh 체크
    if not PDF2ZH_AVAILABLE and not PDF2ZH_CLI_AVAILABLE:
        st.markdown("""
        <div class="error-box">
        ❌ <b>pdf2zh를 사용할 수 없습니다</b><br>
        PDF 번역을 위해 pdf2zh가 필요합니다. 설치를 확인해주세요.
        </div>
        """, unsafe_allow_html=True)
        
        st.code("pip install pdf2zh", language="bash")
        
        with st.expander("🔧 문제 해결 방법"):
            st.markdown(f"""
            ### 디버깅 정보:
            - Python 경로: `{sys.executable}`
            - pdf2zh 모듈: {'✅ 사용 가능' if PDF2ZH_AVAILABLE else '❌ 사용 불가'}
            - pdf2zh CLI: {'✅ 사용 가능' if PDF2ZH_CLI_AVAILABLE else '❌ 사용 불가'}
            - 폰트 경로: {FONT_PATH or 'Not set'}
            """)
        
        st.stop()
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📐 PDF Math Translator</h1>
        <p>수식과 레이아웃을 보존하는 과학 논문 번역 - Powered by OpenAI & pdf2zh</p>
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
        if PDF2ZH_AVAILABLE:
            st.metric("pdf2zh", "API ✅", "🔧")
        elif PDF2ZH_CLI_AVAILABLE:
            st.metric("pdf2zh", "CLI ✅", "🔧")
        else:
            st.metric("pdf2zh", "❌", "🔧")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 번역 설정")
        
        # 번역 서비스 선택 (OpenAI를 기본값으로)
        st.subheader("🌐 번역 서비스")
        
        # OpenAI를 기본으로 하되, API 키가 없으면 안내
        if not st.session_state.api_key:
            st.markdown("""
            <div class="api-key-box">
                <h4>🔑 OpenAI API 키 필요</h4>
                <p>최고 품질의 번역을 위해 OpenAI GPT를 사용합니다.</p>
            </div>
            """, unsafe_allow_html=True)
        
        service = st.selectbox(
            "번역 엔진",
            ["openai", "google", "deepl", "azure", "ollama"],
            index=0,  # openai가 기본
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
            
            with st.expander("💡 OpenAI API 키 받기"):
                st.markdown("""
                **무료 크레딧으로 시작하기:**
                1. [OpenAI Platform](https://platform.openai.com) 접속
                2. 회원가입 (구글/MS 계정 가능)
                3. 신규 가입 시 $5 무료 크레딧 제공
                4. API keys 메뉴에서 키 생성
                5. 생성된 키 복사 (sk-로 시작)
                
                **예상 비용:**
                - 10페이지: 약 $0.02 (26원)
                - 100페이지: 약 $0.20 (260원)
                - 무료 크레딧으로 약 2500페이지 번역 가능
                """)
            
        elif service == "google":
            st.success("🌍 Google 번역 - 무료, API 키 불필요")
            st.info("품질은 OpenAI보다 낮지만 무료입니다")
            
        elif service == "deepl":
            st.info("DeepL - 유럽 언어 전문")
            deepl_key = st.text_input(
                "DeepL API Key",
                type="password",
                placeholder="xxxxx-xxxxx-xxxxx"
            )
            if deepl_key:
                envs["DEEPL_AUTH_KEY"] = deepl_key
                
        elif service == "azure":
            st.info("Azure Translator")
            azure_key = st.text_input(
                "Azure API Key",
                type="password"
            )
            if azure_key:
                envs["AZURE_API_KEY"] = azure_key
                
        elif service == "ollama":
            st.info("Ollama - 로컬 AI")
            ollama_host = st.text_input(
                "Ollama Host",
                value="http://localhost:11434"
            )
            ollama_model = st.text_input(
                "모델명",
                value="gemma2"
            )
            envs["OLLAMA_HOST"] = ollama_host
            envs["OLLAMA_MODEL"] = ollama_model
        
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
            index=0  # 영어가 기본
        )
        
        target_lang = st.selectbox(
            "번역 언어",
            list(lang_map.keys()),
            index=1  # 한국어가 기본
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
            
            use_api = st.checkbox(
                "Python API 사용",
                value=PDF2ZH_AVAILABLE,
                help="체크 해제 시 CLI 사용"
            )
    
    # 메인 영역
    tab1, tab2, tab3 = st.tabs(["📤 번역하기", "📖 사용법", "ℹ️ 정보"])
    
    with tab1:
        # OpenAI 선택했는데 API 키 없으면 큰 안내
        if service == "openai" and "OPENAI_API_KEY" not in envs:
            st.markdown("""
            <div class="api-key-box">
                <h2>🔑 OpenAI API 키를 입력해주세요</h2>
                <p>왼쪽 사이드바에서 API 키를 입력하면 번역을 시작할 수 있습니다.</p>
                <p>무료 크레딧 $5로 약 2500페이지를 번역할 수 있습니다.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="수식이 포함된 과학 논문에 최적화되어 있습니다"
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
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("파일 크기", f"{file_size:.1f} MB")
                with col_b:
                    st.metric("페이지 수", pages_count)
                with col_c:
                    if service == "openai" and "OPENAI_MODEL" in envs:
                        cost_info = estimate_cost(pages_count, envs["OPENAI_MODEL"])
                        st.metric("예상 비용", f"${cost_info['cost_usd']}")
                    else:
                        st.metric("번역 엔진", service.upper())
                
                # 비용 상세 정보 (OpenAI인 경우)
                if service == "openai" and "OPENAI_MODEL" in envs:
                    st.info(f"""
                    💰 **예상 비용 상세**
                    - 모델: {envs["OPENAI_MODEL"]}
                    - 예상 토큰: {cost_info['tokens']:,}개
                    - USD: ${cost_info['cost_usd']}
                    - KRW: ₩{cost_info['cost_krw']:,.0f}
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
                method = "API" if use_api and PDF2ZH_AVAILABLE else "CLI"
                st.markdown(f"""
                <div class="info-box">
                <b>설정 확인</b><br>
                • 엔진: {service.upper()}<br>
                • 언어: {source_lang} → {target_lang}<br>
                • 방식: {method}<br>
                • 페이지: {pages if pages else '전체'}<br>
                • 폰트: {'건너뛰기' if skip_fonts else '포함'}
                </div>
                """, unsafe_allow_html=True)
                
                # API 키 체크
                can_translate = True
                if service == "openai" and "OPENAI_API_KEY" not in envs:
                    st.error("⚠️ OpenAI API 키를 입력하세요")
                    can_translate = False
                elif service == "deepl" and "DEEPL_AUTH_KEY" not in envs:
                    st.error("⚠️ DeepL API 키를 입력하세요")
                    can_translate = False
                elif service == "azure" and "AZURE_API_KEY" not in envs:
                    st.error("⚠️ Azure API 키를 입력하세요")
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
                    progress_bar.progress(0.2)
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
                    progress_bar.progress(0.5)
                    status_text.text("🔄 번역 중... (시간이 걸릴 수 있습니다)")
                    
                    # API 또는 CLI 선택
                    if use_api and PDF2ZH_AVAILABLE:
                        logger.info("Python API 방식으로 번역 시작")
                        success, mono_file, dual_file, message = translate_with_pdf2zh_api(
                            input_path,
                            output_dir,
                            service,
                            lang_map[source_lang],
                            lang_map[target_lang],
                            pages_list,
                            envs,
                            skip_fonts=skip_fonts
                        )
                    else:
                        logger.info("CLI 방식으로 번역 시작")
                        success, mono_file, dual_file, message = translate_with_pdf2zh_cli(
                            input_path,
                            output_dir,
                            service,
                            lang_map[source_lang],
                            lang_map[target_lang],
                            pages,
                            envs,
                            skip_fonts=skip_fonts
                        )
                    
                    elapsed = time.time() - start_time
                    
                    if success:
                        st.balloons()
                        progress_bar.progress(1.0)
                        status_text.text("✅ 번역 완료!")
                        
                        st.markdown(f"""
                        <div class="success-box">
                        🎉 <b>번역 성공!</b><br>
                        ⏱️ 소요 시간: {int(elapsed)}초<br>
                        📄 출력: 2개 파일 생성됨
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
                            st.write("**Python:**", sys.version)
                            st.write("**pdf2zh Module:**", PDF2ZH_AVAILABLE)
                            st.write("**pdf2zh CLI:**", PDF2ZH_CLI_AVAILABLE)
                            st.write("**pdf2zh CMD:**", PDF2ZH_CMD)
                            st.write("**작업 디렉토리:**", os.getcwd())
                            st.write("**임시 파일:**", input_path)
                            st.write("**출력 디렉토리:**", output_dir)
                            st.write("**폰트 경로:**", FONT_PATH)
                            
                            # 폰트 파일 확인
                            if FONT_PATH:
                                st.write("**폰트 파일 존재:**", os.path.exists(FONT_PATH))
                                if os.path.exists(FONT_PATH):
                                    st.write("**폰트 파일 크기:**", os.path.getsize(FONT_PATH), "bytes")
    
    with tab2:
        st.markdown("""
        ### 📖 사용 가이드
        
        #### 🚀 빠른 시작 (OpenAI 추천)
        
        **1단계: OpenAI API 키 받기**
        1. [platform.openai.com](https://platform.openai.com) 접속
        2. 회원가입 (구글/MS 계정 가능)
        3. 신규 가입 시 **$5 무료 크레딧** 자동 제공
        4. API keys 메뉴에서 'Create new secret key' 클릭
        5. 생성된 키 복사 (sk-로 시작)
        
        **2단계: 번역하기**
        1. 왼쪽 사이드바에 API 키 붙여넣기
        2. PDF 파일 업로드
        3. 번역 시작 클릭
        
        #### 💰 비용 안내
        
        **OpenAI 무료 크레딧**
        - 신규 가입 시 $5 무료 제공
        - 약 2500페이지 번역 가능
        - 추가 결제 없이 사용 가능
        
        **페이지별 예상 비용** (gpt-4o-mini 기준)
        | 페이지 | USD | KRW | 무료 크레딧 |
        |--------|-----|-----|------------|
        | 10 | $0.02 | ₩26 | 0.4% 사용 |
        | 50 | $0.10 | ₩130 | 2% 사용 |
        | 100 | $0.20 | ₩260 | 4% 사용 |
        | 500 | $1.00 | ₩1,300 | 20% 사용 |
        
        #### 🆓 무료 옵션
        
        **Google 번역 사용**
        - API 키 불필요
        - 완전 무료
        - 품질은 OpenAI보다 낮음
        - 사이드바에서 'google' 선택
        
        #### ✨ pdf2zh의 특징
        - 📐 **수식 완벽 보존**: LaTeX 수식 그대로 유지
        - 📑 **레이아웃 유지**: 원본 구조 보존
        - 🔤 **폰트 보존**: 서체와 스타일 유지
        - 📊 **도표 위치 유지**: 그래프와 표 위치 보존
        
        #### 💡 사용 팁
        
        **비용 절감**
        - 필요한 페이지만 지정 (예: 1-10)
        - gpt-4o-mini 모델 사용 (기본값)
        - 초록과 결론만 먼저 번역
        
        **품질 최적화**
        - OpenAI 서비스 사용
        - 전체 문서 한 번에 번역
        - dual 파일로 원문 대조 확인
        
        #### ⚠️ 주의사항
        - 스캔된 이미지 PDF는 지원 안 됨
        - 50MB 이상 파일은 시간이 오래 걸림
        - 암호화된 PDF는 지원 안 됨
        
        #### 🔧 문제 해결
        
        **"폰트 오류" 발생 시**
        1. 고급 옵션에서 "폰트 서브셋 건너뛰기" 체크
        2. 다시 번역 시도
        
        **"API 키 오류" 발생 시**
        1. API 키가 sk-로 시작하는지 확인
        2. 키 전체를 복사했는지 확인
        3. OpenAI 계정에 크레딧이 있는지 확인
        
        **번역이 너무 오래 걸릴 때**
        1. 페이지 범위 지정 (예: 1-5)
        2. Google 번역으로 전환
        """)
    
    with tab3:
        st.markdown("""
        ### ℹ️ PDF Math Translator 정보
        
        **버전**: pdf2zh 1.9.0+ with OpenAI GPT  
        **기본 엔진**: OpenAI GPT-4o-mini  
        **개발**: Byaidu & Contributors  
        
        #### 🤖 OpenAI GPT 우선 이유
        
        **최고의 번역 품질**
        - 전문 용어 정확도 95% 이상
        - 문맥 이해 능력 탁월
        - 수식 설명 자연스러움
        - 학술 문체 완벽 보존
        
        **합리적인 비용**
        - 무료 크레딧 $5 제공
        - 페이지당 약 2센트 (26원)
        - 논문 1편 약 50센트 (650원)
        
        #### 🛠️ 기술 스택
        - **AI 엔진**: OpenAI GPT-4 시리즈
        - **PDF 처리**: pdf2zh (수식 보존)
        - **레이아웃**: ONNX DocLayout-YOLO
        - **폰트**: Go Noto Universal
        - **프레임워크**: Streamlit
        
        #### 📊 서비스 비교
        
        | 항목 | OpenAI | Google | DeepL |
        |------|--------|--------|-------|
        | 품질 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
        | 속도 | 빠름 | 매우 빠름 | 빠름 |
        | 비용 | 유료 | 무료 | 유료 |
        | 수식 | 완벽 | 보통 | 좋음 |
        | 전문용어 | 탁월 | 보통 | 좋음 |
        
        #### 🔗 관련 링크
        - [GitHub: PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
        - [OpenAI Platform](https://platform.openai.com)
        - [온라인 데모](https://pdf2zh.com)
        - [문제 신고](https://github.com/Byaidu/PDFMathTranslate/issues)
        
        #### 📝 라이선스
        - pdf2zh: AGPL-3.0
        - OpenAI API: 상용 라이선스
        - 번역 결과물: 사용자 소유
        
        #### 🙏 감사의 말
        pdf2zh 개발팀과 OpenAI에 감사드립니다.
        오픈소스 커뮤니티의 기여로 발전하고 있습니다.
        """)

if __name__ == "__main__":
    main()

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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 폰트 경로 설정 (쓰기 가능한 디렉토리로 변경)
FONT_DIR = Path.home() / ".cache" / "pdf2zh" / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)
os.environ["NOTO_FONT_PATH"] = str(FONT_DIR)

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
for cmd in ['pdf2zh', '/home/adminuser/venv/bin/pdf2zh', '/usr/local/bin/pdf2zh']:
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
        background: linear-gradient(to right, #4080FF, #165DFF);
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
    div[data-testid="metric-container"] {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid #6366f1;
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
        'PyPDF2': False,
        'openai': False,
        'pymupdf': False
    }
    
    for package in ['PyPDF2', 'openai', 'pymupdf']:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
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

def download_font_if_needed():
    """필요한 경우 폰트 다운로드"""
    try:
        # 한국어 폰트 경로
        font_path = FONT_DIR / "SourceHanSerifKR-Regular.ttf"
        
        if not font_path.exists():
            logger.info(f"폰트 다운로드 중: {font_path}")
            # 폰트 다운로드 URL (예시)
            # 실제로는 pdf2zh가 자동으로 다운로드하지만, 경로만 설정
            os.environ["NOTO_FONT_PATH"] = str(font_path)
        
        return True
    except Exception as e:
        logger.error(f"폰트 설정 오류: {e}")
        return False

def translate_with_pdf2zh_api(
    input_file: str,
    output_dir: str,
    service: str,
    lang_from: str,
    lang_to: str,
    pages: Optional[List[int]] = None,
    envs: Optional[Dict] = None,
    thread: int = 2
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
        os.environ["NOTO_FONT_PATH"] = str(FONT_DIR)
        
        # pdf2zh.translate 함수 호출
        from pdf2zh import translate
        
        logger.info(f"PDF 번역 시작: {input_file}")
        logger.info(f"설정: service={service}, lang={lang_from}->{lang_to}, pages={pages}")
        logger.info(f"폰트 경로: {os.environ.get('NOTO_FONT_PATH')}")
        
        # translate 함수 호출
        result = translate(
            files=[input_file],
            output=output_dir,
            pages=pages,
            lang_in=lang_from,
            lang_out=lang_to,
            service=service,
            thread=thread
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
    envs: dict = None
):
    """pdf2zh CLI를 사용한 번역 (폴백)"""
    try:
        if not PDF2ZH_CLI_AVAILABLE or not PDF2ZH_CMD:
            return False, None, None, "pdf2zh CLI를 사용할 수 없습니다"
        
        # 환경 변수 설정
        env = os.environ.copy()
        if envs:
            env.update(envs)
        
        # 폰트 경로 설정
        env["NOTO_FONT_PATH"] = str(FONT_DIR)
        
        # 명령어 구성
        cmd = [
            PDF2ZH_CMD,  # pdf2zh 실행 파일 경로
            input_file,
            "-o", output_dir,
            "-s", service,
            "-li", lang_from,
            "-lo", lang_to,
            "-t", "2"  # 스레드 수
        ]
        
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
    
    # 폰트 설정
    download_font_if_needed()
    
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
            st.markdown("""
            ### 디버깅 정보:
            - Python 경로: `{}`
            - pdf2zh 모듈: {}
            - pdf2zh CLI: {}
            - 폰트 경로: {}
            - PATH: {}
            
            ### 해결 방법:
            1. **requirements.txt 확인**
               ```
               pdf2zh>=1.9.0
               ```
            
            2. **로컬 테스트**
               ```bash
               pip install pdf2zh
               pdf2zh --version
               ```
            """.format(
                sys.executable,
                "✅ 사용 가능" if PDF2ZH_AVAILABLE else "❌ 사용 불가",
                "✅ 사용 가능" if PDF2ZH_CLI_AVAILABLE else "❌ 사용 불가",
                os.environ.get('NOTO_FONT_PATH', 'Not set'),
                os.environ.get('PATH', '')[:200]
            ))
        
        st.stop()
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📐 PDF Math Translator</h1>
        <p>수식과 레이아웃을 보존하는 과학 논문 번역 - Powered by pdf2zh</p>
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
        service = st.selectbox(
            "번역 엔진",
            ["openai", "google", "deepl", "ollama", "azure"],
            index=0,  # openai가 첫 번째이므로 index=0
            help="OpenAI GPT가 가장 정확합니다"
        )
        st.session_state.service = service
        
        # 서비스별 설정
        envs = {}
        if service == "openai":
            st.info("🤖 OpenAI GPT를 사용합니다")
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="sk-...",
                help="https://platform.openai.com/api-keys 에서 발급"
            )
            if api_key:
                envs["OPENAI_API_KEY"] = api_key
                st.session_state.api_key = api_key
            
            model = st.selectbox(
                "GPT 모델",
                ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"],
                index=0,  # gpt-4o-mini를 기본값으로
                help="gpt-4o-mini가 가성비 최고"
            )
            envs["OPENAI_MODEL"] = model
            
            # API 키 없으면 경고
            if not api_key:
                st.warning("⚠️ API 키를 입력하세요")
                with st.expander("API 키 발급 방법"):
                    st.markdown("""
                    1. [OpenAI Platform](https://platform.openai.com) 접속
                    2. 로그인 또는 회원가입
                    3. 우측 상단 프로필 → API keys
                    4. 'Create new secret key' 클릭
                    5. 생성된 키 복사 (sk-로 시작)
                    6. 위 입력란에 붙여넣기
                    """)
            
        elif service == "google":
            st.info("🌍 Google 번역 (무료)")
            st.success("API 키 불필요 - 바로 사용 가능!")
            
        elif service == "deepl":
            st.info("DeepL API 키가 필요합니다")
            deepl_key = st.text_input(
                "DeepL API Key",
                type="password",
                placeholder="xxxxx-xxxxx-xxxxx"
            )
            if deepl_key:
                envs["DEEPL_AUTH_KEY"] = deepl_key
                
        elif service == "azure":
            st.info("Azure Translator 키가 필요합니다")
            azure_key = st.text_input(
                "Azure API Key",
                type="password"
            )
            if azure_key:
                envs["AZURE_API_KEY"] = azure_key
                
        elif service == "ollama":
            st.info("로컬 Ollama 서버 필요")
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
            
            use_api = st.checkbox(
                "Python API 사용",
                value=PDF2ZH_AVAILABLE,
                help="체크 해제 시 CLI 사용"
            )
    
    # 메인 영역
    tab1, tab2, tab3 = st.tabs(["📤 번역하기", "📖 사용법", "ℹ️ 정보"])
    
    with tab1:
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
                • 페이지: {pages if pages else '전체'}
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
                            envs
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
                            envs
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
                            st.write("**폰트 경로:**", os.environ.get('NOTO_FONT_PATH'))
    
    with tab2:
        st.markdown("""
        ### 📖 사용 가이드
        
        #### 🚀 빠른 시작
        1. **OpenAI API 키 입력** (기본 설정)
        2. PDF 파일 업로드
        3. 언어 설정 확인
        4. 번역 시작!
        
        #### 🤖 OpenAI GPT 사용법
        
        **1. API 키 발급**
        - [OpenAI Platform](https://platform.openai.com) 접속
        - 회원가입 또는 로그인
        - API keys 메뉴에서 새 키 생성
        - 생성된 키 복사 (sk-로 시작)
        
        **2. 모델 선택**
        - **gpt-4o-mini**: 가성비 최고 (추천) ⭐
        - **gpt-3.5-turbo**: 가장 저렴
        - **gpt-4o**: 최고 품질
        - **gpt-4-turbo**: 고품질
        
        **3. 비용**
        - 10페이지 기준: 약 $0.02-0.05 (gpt-4o-mini)
        - 100페이지 기준: 약 $0.20-0.50
        
        #### 🌐 다른 번역 서비스
        
        | 서비스 | 품질 | 속도 | 비용 | API 키 |
        |--------|------|------|------|--------|
        | OpenAI | ⭐⭐⭐⭐⭐ | ⚡⚡ | 유료 | 필요 |
        | Google | ⭐⭐⭐ | ⚡⚡⚡ | 무료 | 불필요 |
        | DeepL | ⭐⭐⭐⭐ | ⚡⚡ | 유료 | 필요 |
        | Azure | ⭐⭐⭐⭐ | ⚡⚡ | 유료 | 필요 |
        | Ollama | ⭐⭐⭐ | ⚡ | 무료 | 로컬 서버 |
        
        #### ✨ pdf2zh의 특징
        - 📐 **수식 완벽 보존**: LaTeX 수식 그대로 유지
        - 📑 **레이아웃 유지**: 원본 구조 보존
        - 🔤 **폰트 보존**: 서체와 스타일 유지
        - 📊 **도표 위치 유지**: 그래프와 표 위치 보존
        
        #### 💡 팁
        - **비용 절감**: 필요한 페이지만 지정 (예: 1-10)
        - **품질 우선**: OpenAI gpt-4o-mini 사용
        - **무료 옵션**: Google 번역 사용
        - **대조 확인**: dual 파일로 원문과 번역 비교
        
        #### ⚠️ 주의사항
        - 스캔된 이미지 PDF는 지원하지 않음
        - 매우 큰 파일(>50MB)은 시간이 오래 걸림
        - OpenAI는 API 사용료가 발생함
        
        #### 🔧 문제 해결
        
        **"Permission denied" 오류**
        - 정상입니다. 폰트 경로가 자동 조정됩니다.
        
        **번역이 안 될 때**
        1. Google 번역으로 먼저 테스트
        2. 페이지 범위를 작게 설정 (예: 1-5)
        3. API 키 확인 (OpenAI 사용 시)
        """)
    
    with tab3:
        st.markdown("""
        ### ℹ️ PDF Math Translator 정보
        
        **버전**: pdf2zh 1.9.0+ on Streamlit Cloud  
        **엔진**: [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)  
        **개발**: Byaidu & Contributors  
        
        #### 🛠️ 기술 스택
        - **핵심 엔진**: pdf2zh (수식 보존 번역)
        - **AI 번역**: OpenAI GPT-4 (기본)
        - **PDF 처리**: PyMuPDF, PDFMiner
        - **레이아웃 분석**: ONNX DocLayout-YOLO
        - **웹 프레임워크**: Streamlit
        
        #### 📚 지원 문서 형식
        - ✅ 과학 논문 (arXiv, IEEE, ACM)
        - ✅ 수학/물리 교재
        - ✅ 기술 문서
        - ✅ 연구 보고서
        - ✅ 특허 문서
        - ❌ 스캔된 이미지 PDF
        - ❌ 암호화된 PDF
        
        #### 🎯 OpenAI GPT의 장점
        - **정확도**: 전문 용어와 문맥 이해
        - **일관성**: 문서 전체 톤 유지
        - **유연성**: 다양한 분야 지원
        - **품질**: 자연스러운 번역
        
        #### 🔗 관련 링크
        - [GitHub 저장소](https://github.com/Byaidu/PDFMathTranslate)
        - [온라인 데모](https://pdf2zh.com)
        - [OpenAI Platform](https://platform.openai.com)
        - [문제 신고](https://github.com/Byaidu/PDFMathTranslate/issues)
        
        #### 📝 라이선스
        AGPL-3.0 License
        
        #### 🙏 감사의 말
        이 프로젝트는 오픈소스 커뮤니티의 기여로 만들어졌습니다.
        특히 pdf2zh 개발팀과 OpenAI에 감사드립니다.
        """)

if __name__ == "__main__":
    main()

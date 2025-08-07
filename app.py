"""
PDF 번역기 - Streamlit Cloud 최적화 버전
OpenAI GPT를 활용한 PDF 번역 (pdf2zh 선택적 사용)
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pdf2zh 설치 시도
PDF2ZH_AVAILABLE = False
try:
    # pdf2zh를 런타임에 설치 시도
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "pdf2zh"], 
                          capture_output=True, timeout=60)
    import pdf2zh
    PDF2ZH_AVAILABLE = True
    logger.info("pdf2zh 모듈 로드 성공")
except Exception as e:
    logger.warning(f"pdf2zh 사용 불가: {e}")
    logger.info("OpenAI Direct 모드로 실행됩니다")

# Python 버전 확인
python_version = sys.version_info
st.sidebar.caption(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")

# 페이지 설정
st.set_page_config(
    page_title="PDF AI 번역기 - GPT Powered",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #10a37f, #0d8c6f);
    }
    .api-key-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
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

def check_dependencies():
    """필수 패키지 확인"""
    dependencies = {
        'PyPDF2': False,
        'openai': False,
        'pypdf': False
    }
    
    for package in dependencies.keys():
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies

def get_pdf_page_count(file_content):
    """PDF 페이지 수 확인"""
    try:
        import pypdf
        from io import BytesIO
        
        pdf_file = BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        return len(pdf_reader.pages)
    except ImportError:
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"PDF 페이지 수 확인 오류: {e}")
            return 0

def extract_text_from_pdf(file_content):
    """PDF에서 텍스트 추출"""
    try:
        import pypdf
        from io import BytesIO
        
        pdf_file = BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        text_by_page = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_by_page.append(text)
        
        return text_by_page
    except ImportError:
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_by_page = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_by_page.append(text)
            
            return text_by_page
        except Exception as e:
            logger.error(f"텍스트 추출 오류: {e}")
            return []

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

def translate_with_openai_direct(
    file_content,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    pages: list = None,
    progress_callback = None
):
    """OpenAI API를 직접 호출하여 번역"""
    try:
        import openai
        
        # OpenAI 클라이언트 설정
        client = openai.OpenAI(api_key=api_key)
        
        # PDF에서 텍스트 추출
        text_pages = extract_text_from_pdf(file_content)
        
        if not text_pages:
            return False, None, "PDF에서 텍스트를 추출할 수 없습니다"
        
        # 선택된 페이지만 처리
        if pages:
            selected_pages = []
            for p in pages:
                if 0 <= p < len(text_pages):
                    selected_pages.append(text_pages[p])
            text_pages = selected_pages
        
        if not text_pages:
            return False, None, "선택된 페이지가 없습니다"
        
        total_pages = len(text_pages)
        translated_pages = []
        
        for i, page_text in enumerate(text_pages):
            if progress_callback:
                progress_callback((i + 1) / total_pages, i + 1, total_pages)
            
            if page_text.strip():
                # OpenAI API로 번역
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system", 
                                "content": f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Preserve all formatting and technical terms."
                            },
                            {"role": "user", "content": page_text}
                        ],
                        temperature=0.3
                    )
                    
                    translated_text = response.choices[0].message.content
                    translated_pages.append(translated_text)
                except Exception as e:
                    logger.error(f"페이지 {i+1} 번역 오류: {e}")
                    translated_pages.append(f"[번역 오류: {str(e)}]\n\n{page_text}")
            else:
                translated_pages.append("")
        
        # 번역된 텍스트를 파일로 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for i, text in enumerate(translated_pages):
                f.write(f"=== 페이지 {i+1} ===\n\n")
                f.write(text)
                f.write("\n\n")
            output_file = f.name
        
        return True, output_file, "번역 완료"
        
    except Exception as e:
        logger.error(f"Direct OpenAI 번역 오류: {e}")
        return False, None, str(e)

def translate_with_pdf2zh_cli(
    input_file: str,
    output_dir: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    pages: str = None,
    progress_callback = None
):
    """pdf2zh CLI를 사용한 번역 (폴백)"""
    try:
        # 환경 변수 설정
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_MODEL"] = model
        
        # 명령어 구성
        cmd = [
            sys.executable, "-m", "pdf2zh",
            input_file,
            "-o", output_dir,
            "-s", "openai",
            "-li", source_lang,
            "-lo", target_lang,
            "-t", "1"  # 단일 스레드
        ]
        
        if pages:
            cmd.extend(["-p", pages])
        
        # 실행
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        
        if result.returncode == 0:
            # 출력 파일 찾기
            output_files = list(Path(output_dir).glob("*-mono.pdf"))
            if output_files:
                return True, str(output_files[0]), "번역 완료"
            else:
                return True, None, "번역 완료 (PDF 생성 실패, 텍스트만 제공)"
        else:
            return False, None, f"번역 실패: {result.stderr}"
            
    except Exception as e:
        logger.error(f"pdf2zh CLI 오류: {e}")
        return False, None, str(e)

def main():
    """메인 애플리케이션"""
    
    # 헤더
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🤖 AI PDF 번역기")
        st.caption("ChatGPT로 PDF 문서를 번역 - Streamlit Cloud Edition")
    with col2:
        st.metric("번역 문서", len(st.session_state.translation_history), "📚")
    with col3:
        # 의존성 체크
        deps = check_dependencies()
        all_ok = all(deps.values())
        st.metric("상태", "✅ 정상" if all_ok else "⚠️ 제한", "🔧")
    
    # pdf2zh 상태 알림
    if not PDF2ZH_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>제한된 모드로 실행 중</b><br>
        pdf2zh 모듈을 사용할 수 없어 기본 텍스트 추출 방식으로 작동합니다.<br>
        레이아웃 보존이 제한될 수 있습니다.
        </div>
        """, unsafe_allow_html=True)
    
    # API 키 설정
    if not st.session_state.api_key:
        st.markdown("""
        <div class="api-key-box">
            <h3>🔑 OpenAI API 키 설정</h3>
            <p>ChatGPT API를 사용하려면 API 키가 필요합니다</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key_input = st.text_input(
                "OpenAI API 키 입력",
                type="password",
                placeholder="sk-...",
                help="https://platform.openai.com/api-keys 에서 발급"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("저장", type="primary", use_container_width=True):
                if api_key_input and api_key_input.startswith("sk-"):
                    st.session_state.api_key = api_key_input
                    st.success("✅ API 키 저장됨!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("올바른 API 키를 입력하세요")
        
        with st.expander("❓ API 키 발급 방법"):
            st.markdown("""
            1. [OpenAI 플랫폼](https://platform.openai.com) 접속
            2. 우측 상단 계정 → API keys
            3. 'Create new secret key' 클릭
            4. 키 복사 (sk-로 시작)
            5. 위 입력란에 붙여넣기
            """)
        
        st.stop()
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 번역 설정")
        
        # API 키 표시
        st.success(f"✅ API 키: {st.session_state.api_key[:15]}...")
        if st.button("🔄 API 키 변경"):
            st.session_state.api_key = ""
            st.rerun()
        
        st.divider()
        
        # 번역 모드
        st.subheader("🔧 번역 엔진")
        
        if PDF2ZH_AVAILABLE:
            translation_mode = st.radio(
                "번역 방식",
                ["pdf2zh (권장)", "OpenAI Direct"],
                help="pdf2zh는 레이아웃을 보존합니다"
            )
        else:
            translation_mode = "OpenAI Direct"
            st.info("OpenAI Direct 모드로 고정")
        
        # GPT 모델 선택
        st.subheader("🧠 AI 모델")
        model = st.selectbox(
            "GPT 모델",
            ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            index=1,
            help="gpt-4o-mini가 비용 대비 성능이 좋습니다"
        )
        
        # 언어 설정
        st.subheader("🌍 언어 설정")
        
        lang_map = {
            "한국어": "ko",
            "영어": "en",
            "중국어": "zh",
            "일본어": "ja",
            "독일어": "de",
            "프랑스어": "fr",
            "스페인어": "es"
        }
        
        source_lang = st.selectbox(
            "원본 언어",
            list(lang_map.keys()),
            index=1
        )
        
        target_lang = st.selectbox(
            "번역 언어",
            list(lang_map.keys()),
            index=0
        )
        
        # 페이지 선택
        with st.expander("🔧 고급 옵션"):
            page_range = st.text_input(
                "페이지 범위",
                placeholder="예: 1-5, 10",
                help="비어있으면 전체 번역"
            )
    
    # 메인 영역
    tab1, tab2, tab3 = st.tabs(["📤 번역하기", "💡 도움말", "ℹ️ 정보"])
    
    with tab1:
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="최대 200MB"
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
                    cost_info = estimate_cost(pages_count, model)
                    st.metric("예상 비용", f"${cost_info['cost_usd']}")
                
                # 비용 정보
                st.info(f"""
                **📊 예상 비용**
                - 모델: {model}
                - 토큰: {cost_info['tokens']:,}개
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
                        height="600">
                    </iframe>
                    '''
                    st.markdown(pdf_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 번역 실행")
                
                # 설정 요약
                st.markdown(f"""
                **설정 확인**
                - 🧠 {model}
                - 🌐 {source_lang} → {target_lang}
                - 🔧 {translation_mode if PDF2ZH_AVAILABLE else 'Direct'}
                - 📄 {page_range if page_range else '전체'}
                """)
                
                # 번역 버튼
                if st.button("🚀 번역 시작", type="primary", use_container_width=True):
                    # 진행률 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 진행률 콜백
                    def update_progress(progress, current, total):
                        progress_bar.progress(progress)
                        status_text.text(f"🤖 번역 중... 페이지 {current}/{total}")
                    
                    # 페이지 범위 파싱
                    selected_pages = []
                    if page_range:
                        try:
                            for p in page_range.split(","):
                                p = p.strip()
                                if "-" in p:
                                    start, end = p.split("-")
                                    selected_pages.extend(range(int(start)-1, int(end)))
                                else:
                                    selected_pages.append(int(p)-1)
                        except:
                            st.error("잘못된 페이지 범위입니다")
                            st.stop()
                    
                    # 번역 실행
                    start_time = time.time()
                    
                    if PDF2ZH_AVAILABLE and "pdf2zh" in translation_mode:
                        # pdf2zh 사용
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_input:
                            tmp_input.write(file_content)
                            input_path = tmp_input.name
                        
                        output_dir = tempfile.mkdtemp()
                        
                        success, output_file, message = translate_with_pdf2zh_cli(
                            input_path,
                            output_dir,
                            st.session_state.api_key,
                            model,
                            lang_map[source_lang],
                            lang_map[target_lang],
                            page_range,
                            update_progress
                        )
                        
                        # 임시 파일 정리
                        os.unlink(input_path)
                    else:
                        # OpenAI Direct 사용
                        success, output_file, message = translate_with_openai_direct(
                            file_content,
                            st.session_state.api_key,
                            model,
                            source_lang,
                            target_lang,
                            selected_pages,
                            update_progress
                        )
                    
                    elapsed = time.time() - start_time
                    
                    if success:
                        st.balloons()
                        progress_bar.progress(1.0)
                        status_text.text("✅ 번역 완료!")
                        
                        st.markdown(f"""
                        <div class="success-box">
                        🎉 <b>번역 성공!</b><br>
                        ⏱️ 소요 시간: {int(elapsed)}초
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if output_file and os.path.exists(output_file):
                            # 다운로드
                            with open(output_file, 'rb') as f:
                                result_data = f.read()
                            
                            # 파일 확장자 결정
                            if output_file.endswith('.pdf'):
                                mime_type = "application/pdf"
                                file_ext = "pdf"
                            else:
                                mime_type = "text/plain"
                                file_ext = "txt"
                            
                            st.download_button(
                                label=f"📥 번역 결과 다운로드 (.{file_ext})",
                                data=result_data,
                                file_name=f"translated_{uploaded_file.name.replace('.pdf', f'.{file_ext}')}",
                                mime=mime_type,
                                use_container_width=True,
                                type="primary"
                            )
                            
                            # 텍스트 파일인 경우 미리보기
                            if file_ext == "txt":
                                with st.expander("📝 번역 결과 미리보기"):
                                    st.text(result_data.decode('utf-8', errors='ignore')[:2000] + "...")
                            
                            # 임시 파일 정리
                            try:
                                os.unlink(output_file)
                            except:
                                pass
                        else:
                            st.warning("번역은 완료되었으나 파일 생성에 실패했습니다")
                        
                        # 기록
                        st.session_state.translation_history.append({
                            'filename': uploaded_file.name,
                            'pages': pages_count,
                            'model': model,
                            'time': datetime.now().strftime("%H:%M")
                        })
                    else:
                        st.error(f"❌ 번역 실패")
                        st.markdown(f"""
                        <div class="error-box">
                        <b>오류 메시지:</b><br>
                        {message}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("🔍 디버깅 정보"):
                            st.code(message)
                            st.write("**Python:**", sys.version)
                            st.write("**pdf2zh:**", "사용 가능" if PDF2ZH_AVAILABLE else "사용 불가")
    
    with tab2:
        st.markdown("""
        ### 💡 사용 가이드
        
        #### 🚀 빠른 시작
        1. OpenAI API 키 입력 (첫 실행 시)
        2. PDF 파일 업로드
        3. 언어 설정 확인
        4. 번역 시작 클릭
        
        #### ⚠️ 제한사항 (Streamlit Cloud)
        - 파일 크기: 최대 200MB
        - 실행 시간: 최대 10분
        - pdf2zh가 작동하지 않을 경우 텍스트만 추출됩니다
        
        #### 💰 비용 절감 팁
        - `gpt-4o-mini` 모델 사용 (성능 대비 저렴)
        - 필요한 페이지만 선택하여 번역
        - 긴 문서는 나눠서 번역
        
        #### 📝 페이지 범위 지정
        - 전체: 비워두기
        - 특정 페이지: `1, 3, 5`
        - 범위: `1-10`
        - 혼합: `1-5, 10, 15-20`
        
        #### 🔧 문제 해결
        **"pdf2zh 사용 불가" 메시지**
        - 정상입니다. OpenAI Direct 모드로 작동합니다
        - 텍스트 추출만 가능하며 레이아웃은 보존되지 않습니다
        
        **번역이 느린 경우**
        - 페이지 수가 많으면 시간이 오래 걸립니다
        - 필요한 페이지만 선택하세요
        
        **오류 발생 시**
        - API 키 확인
        - 파일 크기 확인 (200MB 이하)
        - 다른 모델로 시도
        """)
    
    with tab3:
        st.markdown("""
        ### 🤖 AI PDF 번역기 정보
        
        **버전**: 2.2.0 (Streamlit Cloud Edition)  
        **엔진**: OpenAI GPT + pdf2zh (선택적)  
        **환경**: Streamlit Cloud  
        
        #### ✨ 특징
        - 🤖 ChatGPT 기반 고품질 번역
        - ☁️ Streamlit Cloud 최적화
        - 🔧 자동 폴백 메커니즘
        - 📊 비용 사전 계산
        
        #### 📊 모델 비교
        
        | 모델 | 속도 | 품질 | 비용 |
        |------|------|------|------|
        | gpt-3.5-turbo | ⚡⚡⚡ | ⭐⭐⭐ | 💰 |
        | gpt-4o-mini | ⚡⚡ | ⭐⭐⭐⭐ | 💰 |
        | gpt-4o | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰 |
        | gpt-4-turbo | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 |
        
        #### 🔗 관련 링크
        - [OpenAI API](https://platform.openai.com)
        - [PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
        - [Streamlit](https://streamlit.io)
        
        #### 📝 라이선스
        이 앱은 오픈소스 프로젝트들을 활용합니다.
        - PDFMathTranslate: AGPL-3.0
        - Streamlit: Apache 2.0
        """)
        
        # 번역 기록
        if st.session_state.translation_history:
            st.divider()
            st.subheader("📚 번역 기록")
            for item in st.session_state.translation_history[-5:]:
                st.text(f"• {item['time']} - {item['filename']} ({item['pages']}페이지, {item['model']})")

if __name__ == "__main__":
    main()

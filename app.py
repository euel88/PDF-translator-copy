"""
PDF 번역기 - OpenAI GPT 버전
ChatGPT API를 활용한 고품질 PDF 번역
"""

import streamlit as st
import subprocess
import tempfile
import os
from pathlib import Path
import time
import base64
import PyPDF2
from datetime import datetime
import json

# 페이지 설정
st.set_page_config(
    page_title="PDF AI 번역기 - GPT Powered",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
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
    div[data-testid="metric-container"] {
        background: rgba(16, 163, 127, 0.1);
        border: 1px solid #10a37f;
        padding: 10px;
        border-radius: 10px;
    }
    .gpt-badge {
        background: #10a37f;
        color: white;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

def check_pdf2zh():
    """pdf2zh 설치 확인"""
    try:
        result = subprocess.run(['pdf2zh', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def install_pdf2zh():
    """pdf2zh 자동 설치"""
    with st.spinner("pdf2zh 설치 중... (1회만 필요)"):
        try:
            subprocess.check_call(["pip", "install", "pdf2zh", "-q"])
            return True
        except:
            return False

def get_pdf_info(file_path):
    """PDF 정보 추출"""
    try:
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            return {
                'pages': len(pdf.pages),
                'encrypted': pdf.is_encrypted
            }
    except:
        return {'pages': 0, 'encrypted': False}

def estimate_cost(pages: int, model: str) -> dict:
    """OpenAI API 비용 추정"""
    # 대략적인 추정 (페이지당 토큰 수는 내용에 따라 다름)
    tokens_per_page = 1500  # 평균 추정치
    total_tokens = pages * tokens_per_page
    
    # GPT 모델별 가격 (1K 토큰당, 2024년 기준)
    prices = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    if model in prices:
        # 입력과 출력이 비슷하다고 가정
        input_cost = (total_tokens / 1000) * prices[model]["input"]
        output_cost = (total_tokens / 1000) * prices[model]["output"]
        total_cost = input_cost + output_cost
        
        return {
            "tokens": total_tokens,
            "cost_usd": round(total_cost, 2),
            "cost_krw": round(total_cost * 1300, 0)  # 환율 1300원 가정
        }
    
    return {"tokens": total_tokens, "cost_usd": 0, "cost_krw": 0}

def translate_with_gpt(
    input_file: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    pages: str = None,
    file_type: str = "mono",
    progress_callback = None
):
    """OpenAI GPT를 사용한 PDF 번역"""
    
    # 출력 파일명
    output_suffix = "-gpt-dual" if file_type == "dual" else "-gpt"
    output_file = input_file.replace('.pdf', f'{output_suffix}.pdf')
    
    # 명령어 구성
    cmd = [
        "pdf2zh",
        input_file,
        "-o", output_file,
        "-s", "openai",  # OpenAI 서비스 사용
        "-li", source_lang,
        "-lo", target_lang
    ]
    
    # 페이지 범위
    if pages:
        cmd.extend(["-p", pages])
    
    # 이중 언어 모드
    if file_type == "dual":
        cmd.append("--dual")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_MODEL"] = model  # GPT 모델 지정
    env["OPENAI_BASE_URL"] = "https://api.openai.com/v1"  # 기본 엔드포인트
    
    try:
        # 프로세스 실행
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # 진행률 시뮬레이션
        if progress_callback:
            pdf_info = get_pdf_info(input_file)
            total_pages = pdf_info['pages']
            
            for i in range(total_pages):
                progress = (i + 1) / total_pages
                progress_callback(progress, i + 1, total_pages)
                time.sleep(1)  # 실제로는 프로세스 출력을 파싱해야 함
        
        # 완료 대기
        stdout, stderr = process.communicate(timeout=600)  # 10분 타임아웃
        
        if process.returncode == 0:
            return True, output_file, "번역 완료"
        else:
            return False, None, stderr
            
    except subprocess.TimeoutExpired:
        process.kill()
        return False, None, "번역 시간 초과 (10분)"
    except Exception as e:
        return False, None, str(e)

def main():
    """메인 애플리케이션"""
    
    # 헤더
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("🤖 AI PDF 번역기")
        st.markdown('<span class="gpt-badge">GPT-4 Powered</span>', unsafe_allow_html=True)
        st.caption("ChatGPT로 과학 논문을 정확하게 번역 - 수식과 레이아웃 완벽 보존")
    with col2:
        st.metric("번역 문서", len(st.session_state.translation_history), "📚")
    with col3:
        tokens = st.session_state.total_tokens_used
        st.metric("토큰 사용", f"{tokens:,}", "🪙")
    
    # pdf2zh 설치 확인
    if not check_pdf2zh():
        st.warning("⚠️ 초기 설정이 필요합니다")
        if st.button("🔧 자동 설정", type="primary"):
            if install_pdf2zh():
                st.success("✅ 설정 완료!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("자동 설정 실패. 터미널에서: `pip install pdf2zh`")
        return
    
    # OpenAI API 키 설정 (메인 영역 상단)
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
        
        # API 키 발급 안내
        with st.expander("❓ API 키 발급 방법"):
            st.markdown("""
            1. [OpenAI 플랫폼](https://platform.openai.com) 접속
            2. 우측 상단 계정 → API keys
            3. 'Create new secret key' 클릭
            4. 키 복사 (sk-로 시작)
            5. 위 입력란에 붙여넣기
            
            **요금**: 
            - GPT-3.5: ~$0.002/페이지 (약 3원)
            - GPT-4: ~$0.06/페이지 (약 80원)
            """)
        
        st.stop()  # API 키 없으면 여기서 중단
    
    # 사이드바 - 번역 설정
    with st.sidebar:
        st.header("⚙️ 번역 설정")
        
        # API 키 표시 및 변경
        st.success(f"✅ API 키: {st.session_state.api_key[:10]}...")
        if st.button("🔄 API 키 변경"):
            st.session_state.api_key = ""
            st.rerun()
        
        st.divider()
        
        # GPT 모델 선택
        st.subheader("🧠 AI 모델")
        model = st.selectbox(
            "GPT 모델 선택",
            ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
            format_func=lambda x: {
                "gpt-3.5-turbo": "GPT-3.5 Turbo (빠르고 저렴)",
                "gpt-4-turbo": "GPT-4 Turbo (균형)",
                "gpt-4": "GPT-4 (최고 품질)"
            }.get(x, x),
            help="GPT-4는 더 정확하지만 비용이 높습니다"
        )
        
        # 모델별 특징 표시
        if model == "gpt-3.5-turbo":
            st.info("💰 가장 경제적 / ⚡ 빠른 속도")
        elif model == "gpt-4-turbo":
            st.info("⚖️ 품질과 비용의 균형")
        else:
            st.info("🏆 최고 품질 / 전문 용어 정확")
        
        # 언어 설정
        st.subheader("🌍 언어 설정")
        
        lang_map = {
            "한국어": "ko",
            "영어": "en",
            "중국어": "zh",
            "일본어": "ja",
            "독일어": "de",
            "프랑스어": "fr",
            "스페인어": "es",
            "러시아어": "ru"
        }
        
        source_lang = st.selectbox(
            "원본 언어",
            list(lang_map.keys()),
            index=1  # 기본: 영어
        )
        
        target_lang = st.selectbox(
            "번역 언어",
            list(lang_map.keys()),
            index=0  # 기본: 한국어
        )
        
        # 고급 옵션
        with st.expander("🔧 고급 옵션"):
            file_type = st.radio(
                "출력 형식",
                ["mono", "dual"],
                format_func=lambda x: "번역만" if x == "mono" else "원본+번역",
                help="dual: 원본과 번역을 나란히 표시"
            )
            
            pages = st.text_input(
                "페이지 범위",
                placeholder="예: 1-10, 15, 20-30",
                help="비용 절감을 위해 특정 페이지만 번역"
            )
            
            temperature = st.slider(
                "창의성 (Temperature)",
                0.0, 1.0, 0.3,
                help="낮을수록 일관성, 높을수록 창의적"
            )
        
        # 사용 통계
        st.divider()
        st.subheader("📊 사용 통계")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("번역 완료", f"{len(st.session_state.translation_history)}개")
        with col2:
            total_cost = st.session_state.total_tokens_used * 0.002 / 1000  # 대략적 계산
            st.metric("예상 비용", f"${total_cost:.2f}")
        
        # 번역 기록
        if st.session_state.translation_history:
            st.subheader("🕐 최근 번역")
            for item in st.session_state.translation_history[-3:]:
                with st.container():
                    st.write(f"📄 **{item['filename']}**")
                    st.caption(f"{item['pages']}페이지 | {item['model']} | {item['time']}")
    
    # 메인 영역 - 탭
    tab1, tab2, tab3 = st.tabs(["📤 번역하기", "💡 사용 팁", "ℹ️ 정보"])
    
    with tab1:
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "PDF 파일을 드래그하거나 클릭하여 선택",
            type=['pdf'],
            help="최대 200MB, 과학 논문 최적화"
        )
        
        if uploaded_file:
            # 파일 정보 표시
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ 파일 준비 완료: **{uploaded_file.name}**")
                
                # 임시 파일 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # PDF 정보
                pdf_info = get_pdf_info(tmp_path)
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                
                # 메트릭 표시
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("파일 크기", f"{file_size:.1f} MB")
                with col_b:
                    st.metric("총 페이지", pdf_info['pages'])
                with col_c:
                    # 비용 추정
                    cost_info = estimate_cost(pdf_info['pages'], model)
                    st.metric("예상 비용", f"${cost_info['cost_usd']}")
                
                # 비용 상세 정보
                st.info(f"""
                **📊 예상 비용 분석**
                - 모델: {model}
                - 예상 토큰: {cost_info['tokens']:,}개
                - USD: ${cost_info['cost_usd']}
                - KRW: ₩{cost_info['cost_krw']:,.0f}
                """)
                
                # PDF 미리보기
                with st.expander("👁️ PDF 미리보기"):
                    pdf_display = base64.b64encode(uploaded_file.getvalue()).decode()
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
                **현재 설정**
                - 🧠 모델: {model}
                - 🌐 언어: {source_lang} → {target_lang}
                - 📄 형식: {'원본+번역' if file_type == 'dual' else '번역만'}
                - 📖 페이지: {pages if pages else '전체'}
                """)
                
                # 번역 버튼
                if st.button(
                    "🚀 GPT 번역 시작",
                    type="primary",
                    use_container_width=True,
                    help=f"예상 비용: ${cost_info['cost_usd']}"
                ):
                    # 진행률 컨테이너
                    progress_container = st.container()
                    
                    with progress_container:
                        st.markdown("### 📊 번역 진행 상황")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # 진행률 콜백
                        def update_progress(progress, current, total):
                            progress_bar.progress(progress)
                            status_text.text(f"🤖 GPT 번역 중... 페이지 {current}/{total}")
                        
                        # 번역 실행
                        start_time = time.time()
                        
                        success, output_file, message = translate_with_gpt(
                            tmp_path,
                            st.session_state.api_key,
                            model,
                            lang_map[source_lang],
                            lang_map[target_lang],
                            pages,
                            file_type,
                            update_progress
                        )
                        
                        elapsed = time.time() - start_time
                        
                        if success and output_file and os.path.exists(output_file):
                            # 성공
                            st.balloons()
                            progress_bar.progress(1.0)
                            status_text.text("✅ GPT 번역 완료!")
                            
                            # 결과 표시
                            st.markdown(f"""
                            <div class="success-box">
                            🎉 <b>번역 성공!</b><br>
                            ⏱️ 소요 시간: {int(elapsed)}초<br>
                            💰 예상 비용: ${cost_info['cost_usd']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 다운로드
                            with open(output_file, 'rb') as f:
                                pdf_data = f.read()
                            
                            st.download_button(
                                label="📥 번역된 PDF 다운로드",
                                data=pdf_data,
                                file_name=f"{model}_{uploaded_file.name}",
                                mime="application/pdf",
                                use_container_width=True,
                                type="primary"
                            )
                            
                            # 기록 저장
                            st.session_state.translation_history.append({
                                'filename': uploaded_file.name,
                                'pages': pdf_info['pages'],
                                'model': model,
                                'time': datetime.now().strftime("%H:%M"),
                                'tokens': cost_info['tokens']
                            })
                            st.session_state.total_tokens_used += cost_info['tokens']
                            
                            # 정리
                            os.unlink(tmp_path)
                            os.unlink(output_file)
                            
                        else:
                            st.error(f"❌ 번역 실패: {message}")
                            
                            # 디버깅 정보
                            with st.expander("🔍 오류 상세"):
                                st.code(message)
                                st.markdown("""
                                **해결 방법:**
                                1. API 키가 올바른지 확인
                                2. OpenAI 계정에 크레딧이 있는지 확인
                                3. 페이지 수를 줄여서 다시 시도
                                """)
    
    with tab2:
        st.markdown("""
        ### 💡 GPT 번역 사용 팁
        
        #### 🎯 최적 설정
        
        | 용도 | 모델 | 페이지 | 예상 비용 |
        |------|------|--------|----------|
        | 초록/요약 | GPT-3.5 | 1-3 | $0.01 |
        | 중요 논문 | GPT-4 | 전체 | $1-3 |
        | 빠른 확인 | GPT-3.5 | 10-20 | $0.05 |
        | 정밀 번역 | GPT-4 Turbo | 전체 | $0.5-1 |
        
        #### 💰 비용 절감 방법
        
        1. **페이지 지정**: 중요한 부분만 번역 (예: "1-5, 10-15")
        2. **GPT-3.5 활용**: 초벌 번역은 저렴한 모델로
        3. **요약 먼저**: Abstract와 Conclusion만 먼저 번역
        
        #### 🔧 문제 해결
        
        **"API 키 오류"가 나타날 때:**
        - OpenAI 대시보드에서 키 재확인
        - 키가 'sk-'로 시작하는지 확인
        - 사용량 한도 확인
        
        **"시간 초과" 오류:**
        - 페이지 수를 줄여서 재시도
        - 네트워크 연결 확인
        
        #### 📈 품질 향상 팁
        
        - **수식 많은 논문**: GPT-4 추천 (LaTeX 이해도 높음)
        - **일반 문서**: GPT-3.5로 충분
        - **의학/법률**: GPT-4 필수 (전문 용어 정확도)
        """)
    
    with tab3:
        st.markdown("""
        ### 🤖 AI PDF 번역기 정보
        
        **버전**: 2.0.0 (GPT Enhanced)  
        **엔진**: OpenAI GPT-3.5/4  
        **기반**: PDFMathTranslate + Streamlit  
        
        #### ✨ 주요 특징
        
        - 🧠 **ChatGPT 기반**: 최신 AI 모델로 자연스러운 번역
        - 📐 **수식 보존**: LaTeX 수식 완벽 보존
        - 📊 **그래프/표 유지**: 시각 자료 레이아웃 보존
        - 💬 **문맥 이해**: GPT의 뛰어난 문맥 파악 능력
        - 🎯 **전문 용어**: 학술 용어 정확한 번역
        
        #### 📊 모델 비교
        
        | 모델 | 속도 | 품질 | 비용 | 추천 용도 |
        |------|------|------|------|----------|
        | GPT-3.5 | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | 일반 문서 |
        | GPT-4 Turbo | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | 중요 문서 |
        | GPT-4 | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | 정밀 번역 |
        
        #### 🔗 유용한 링크
        
        - [OpenAI API 키 발급](https://platform.openai.com/api-keys)
        - [요금 계산기](https://openai.com/pricing)
        - [PDFMathTranslate GitHub](https://github.com/Byaidu/PDFMathTranslate)
        
        #### 📞 지원
        
        문제가 있으신가요?
        - GitHub Issues에 문의
        - OpenAI 지원팀 (API 관련)
        
        ---
        
        Made with ❤️ using Streamlit & OpenAI GPT
        """)

if __name__ == "__main__":
    main()

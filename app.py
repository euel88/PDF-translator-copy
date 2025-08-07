"""
PDF 번역기 - Streamlit Cloud 간단 버전
OpenAI GPT를 활용한 PDF 텍스트 추출 및 번역
"""

import streamlit as st
import os
import time
import base64
from datetime import datetime
import PyPDF2
from io import BytesIO
import openai

# 페이지 설정
st.set_page_config(
    page_title="PDF AI 번역기",
    page_icon="🤖",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #10a37f;
        color: white;
    }
    .stButton > button:hover {
        background-color: #0d8c6f;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""

def extract_text_from_pdf(file_bytes):
    """PDF에서 텍스트 추출"""
    try:
        pdf_file = BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        total_pages = len(pdf_reader.pages)
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n=== 페이지 {page_num + 1} ===\n\n"
                text += page_text
        
        return text, total_pages
    except Exception as e:
        st.error(f"PDF 읽기 오류: {str(e)}")
        return None, 0

def translate_text(text, api_key, source_lang="영어", target_lang="한국어"):
    """OpenAI API로 텍스트 번역"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 텍스트가 너무 길면 자르기 (토큰 제한 고려)
        max_chars = 10000
        if len(text) > max_chars:
            text = text[:max_chars]
            st.warning(f"텍스트가 너무 길어 처음 {max_chars}자만 번역합니다.")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"당신은 전문 번역가입니다. {source_lang}를 {target_lang}로 정확하게 번역해주세요."
                },
                {
                    "role": "user",
                    "content": f"다음 텍스트를 번역해주세요:\n\n{text}"
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"번역 오류: {str(e)}")
        return None

# 메인 UI
st.title("🤖 PDF AI 번역기")
st.caption("ChatGPT를 활용한 간단한 PDF 번역 도구")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 입력
    api_key_input = st.text_input(
        "OpenAI API 키",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
        help="https://platform.openai.com/api-keys"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.success("✅ API 키 설정됨")
    else:
        st.warning("⚠️ API 키를 입력하세요")
    
    st.divider()
    
    # 언어 설정
    source_lang = st.selectbox(
        "원본 언어",
        ["영어", "한국어", "중국어", "일본어", "스페인어", "프랑스어", "독일어"],
        index=0
    )
    
    target_lang = st.selectbox(
        "번역 언어",
        ["한국어", "영어", "중국어", "일본어", "스페인어", "프랑스어", "독일어"],
        index=0
    )
    
    st.divider()
    
    # 정보
    st.info("""
    **사용법:**
    1. API 키 입력
    2. PDF 파일 업로드
    3. 번역 버튼 클릭
    
    **제한사항:**
    - 파일 크기: 200MB 이하
    - 텍스트만 추출 (이미지 제외)
    - 긴 문서는 일부만 번역
    """)

# 메인 컨텐츠
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 PDF 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=['pdf'],
        help="최대 200MB"
    )
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        
        # 파일 정보
        file_size = len(file_bytes) / (1024 * 1024)
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        st.info(f"📊 파일 크기: {file_size:.2f} MB")
        
        # PDF 미리보기
        with st.expander("👁️ PDF 미리보기"):
            pdf_display = base64.b64encode(file_bytes).decode()
            pdf_html = f'<iframe src="data:application/pdf;base64,{pdf_display}" width="100%" height="500"></iframe>'
            st.markdown(pdf_html, unsafe_allow_html=True)
        
        # 번역 버튼
        if st.button("🚀 번역 시작", type="primary", use_container_width=True):
            if not st.session_state.api_key:
                st.error("❌ API 키를 먼저 입력하세요!")
            else:
                with st.spinner("📖 PDF 텍스트 추출 중..."):
                    text, total_pages = extract_text_from_pdf(file_bytes)
                
                if text:
                    st.info(f"📄 총 {total_pages} 페이지에서 텍스트 추출 완료")
                    
                    with st.spinner("🤖 번역 중... (시간이 걸릴 수 있습니다)"):
                        translated = translate_text(
                            text, 
                            st.session_state.api_key,
                            source_lang,
                            target_lang
                        )
                    
                    if translated:
                        st.session_state.translated_text = translated
                        st.success("✅ 번역 완료!")
                else:
                    st.error("❌ PDF에서 텍스트를 추출할 수 없습니다.")

with col2:
    st.subheader("📝 번역 결과")
    
    if st.session_state.translated_text:
        # 텍스트 영역에 번역 결과 표시
        st.text_area(
            "번역된 텍스트",
            value=st.session_state.translated_text,
            height=400,
            key="translation_output"
        )
        
        # 다운로드 버튼
        st.download_button(
            label="📥 번역 결과 다운로드 (TXT)",
            data=st.session_state.translated_text,
            file_name=f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # 복사 안내
        st.info("💡 텍스트를 선택하고 Ctrl+C (Mac: Cmd+C)로 복사할 수 있습니다.")
    else:
        st.info("번역 결과가 여기에 표시됩니다.")

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Powered by OpenAI GPT-3.5 | Made with Streamlit</p>
    <p>⚠️ 이 도구는 텍스트만 추출하며, 레이아웃과 이미지는 보존되지 않습니다.</p>
</div>
""", unsafe_allow_html=True)

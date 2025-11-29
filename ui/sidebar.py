"""
Sidebar component for the Streamlit app.
사이드바 컴포넌트 (한글 UI)
"""
import streamlit as st
from typing import Dict, Any, Optional

from config import APP_CONFIG, LANGUAGES, TranslationService, SERVICE_NAMES


def render_sidebar() -> Dict[str, Any]:
    """
    번역 설정이 포함된 사이드바 렌더링

    Returns:
        선택된 설정이 담긴 딕셔너리
    """
    settings = {}

    with st.sidebar:
        # 앱 타이틀
        st.markdown("## 설정")
        st.divider()

        # 번역 서비스
        st.markdown("### 번역 서비스")

        service_options = {
            "google": "Google 번역 (무료)",
            "openai": "OpenAI GPT (ChatGPT)",
            "deepl": "DeepL",
            "ollama": "Ollama (로컬)",
            "deepseek": "DeepSeek",
            "gemini": "Gemini",
        }

        settings["service"] = st.selectbox(
            "서비스 선택",
            options=list(service_options.keys()),
            format_func=lambda x: service_options.get(x, x),
            key="sidebar_service",
            label_visibility="collapsed"
        )

        # 서비스별 설정
        _render_service_settings(settings)

        st.divider()

        # 언어 설정
        st.markdown("### 언어")

        lang_options = list(LANGUAGES.keys())

        settings["source_lang"] = st.selectbox(
            "원본 언어",
            options=lang_options,
            format_func=lambda x: LANGUAGES.get(x, x),
            index=lang_options.index("en") if "en" in lang_options else 0,
            key="sidebar_source_lang"
        )

        settings["target_lang"] = st.selectbox(
            "번역 언어",
            options=lang_options,
            format_func=lambda x: LANGUAGES.get(x, x),
            index=lang_options.index("ko") if "ko" in lang_options else 0,
            key="sidebar_target_lang"
        )

        st.divider()

        # 출력 설정
        st.markdown("### 출력 형식")

        settings["output_format"] = st.radio(
            "형식",
            options=["dual", "mono", "both"],
            format_func=lambda x: {
                "dual": "이중언어 (원문+번역)",
                "mono": "번역본만",
                "both": "둘 다"
            }.get(x, x),
            key="sidebar_output_format",
            label_visibility="collapsed"
        )

        st.divider()

        # 다운로드 경로 설정
        st.markdown("### 저장 경로")

        use_custom_path = st.checkbox(
            "사용자 지정 경로 사용",
            value=False,
            key="sidebar_use_custom_path",
            help="체크하면 지정한 경로에 파일이 저장됩니다"
        )

        if use_custom_path:
            settings["download_path"] = st.text_input(
                "저장 경로",
                value="./output",
                key="sidebar_download_path",
                help="번역된 PDF가 저장될 경로"
            )

        st.divider()

        # 고급 설정
        with st.expander("고급 설정", expanded=False):
            settings["thread_count"] = st.slider(
                "스레드 수",
                min_value=1,
                max_value=8,
                value=4,
                key="sidebar_threads",
                help="번역 병렬 처리 수"
            )

            settings["use_cache"] = st.checkbox(
                "캐시 사용",
                value=True,
                key="sidebar_cache",
                help="번역 캐시를 사용하여 중복 번역 방지"
            )

        st.divider()

        # 정보 섹션
        st.markdown("### 정보")
        st.caption(f"**{APP_CONFIG.app_name}** v{APP_CONFIG.version}")
        st.caption("[PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) 기반")

        # 도움말 링크
        st.markdown("""
        ---
        [문서](https://github.com/PDFMathTranslate/PDFMathTranslate) |
        [이슈 보고](https://github.com/PDFMathTranslate/PDFMathTranslate/issues)
        """)

    return settings


def _render_service_settings(settings: Dict[str, Any]):
    """서비스별 설정 렌더링"""
    service = settings.get("service", "google")

    if service == "openai":
        st.markdown("#### ChatGPT API 설정")

        settings["openai_api_key"] = st.text_input(
            "OpenAI API 키",
            type="password",
            key="sidebar_openai_key",
            help="OpenAI Platform에서 발급받은 API 키를 입력하세요"
        )

        settings["openai_model"] = st.selectbox(
            "모델",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            key="sidebar_openai_model",
            help="gpt-4o-mini: 빠르고 저렴 / gpt-4o: 최고 품질"
        )

        # 고급 OpenAI 설정
        with st.expander("OpenAI 고급 설정", expanded=False):
            settings["openai_base_url"] = st.text_input(
                "API Base URL (선택)",
                placeholder="https://api.openai.com/v1",
                key="sidebar_openai_base_url",
                help="커스텀 엔드포인트 사용 시 입력 (Azure OpenAI 등)"
            )

        # API 키 상태 표시
        if settings.get("openai_api_key"):
            st.success("API 키 입력됨")
        else:
            st.info("API 키를 입력하세요")
            st.caption("[OpenAI Platform](https://platform.openai.com)에서 API 키 발급")

    elif service == "deepl":
        settings["deepl_api_key"] = st.text_input(
            "DeepL API 키",
            type="password",
            key="sidebar_deepl_key",
            help="DeepL에서 발급받은 API 키"
        )

        if settings.get("deepl_api_key"):
            st.success("API 키 입력됨")
        else:
            st.info("API 키를 입력하세요")

    elif service == "ollama":
        st.markdown("#### Ollama 설정 (로컬)")

        settings["ollama_host"] = st.text_input(
            "Ollama 호스트",
            value="http://localhost:11434",
            key="sidebar_ollama_host",
            help="Ollama 서버 주소"
        )
        settings["ollama_model"] = st.text_input(
            "모델",
            value="gemma2",
            key="sidebar_ollama_model",
            help="예: gemma2, llama3, mistral"
        )

        st.caption("Ollama가 로컬에 설치되어 있어야 합니다")

    elif service == "deepseek":
        settings["deepseek_api_key"] = st.text_input(
            "DeepSeek API 키",
            type="password",
            key="sidebar_deepseek_key",
            help="DeepSeek에서 발급받은 API 키"
        )

        if settings.get("deepseek_api_key"):
            st.success("API 키 입력됨")
        else:
            st.info("API 키를 입력하세요")

    elif service == "gemini":
        settings["gemini_api_key"] = st.text_input(
            "Gemini API 키",
            type="password",
            key="sidebar_gemini_key",
            help="Google AI Studio에서 발급받은 API 키"
        )

        if settings.get("gemini_api_key"):
            st.success("API 키 입력됨")
        else:
            st.info("API 키를 입력하세요")

    elif service == "google":
        st.info("Google 번역은 API 키가 필요 없습니다")
        st.caption("무료로 사용 가능하지만, 품질은 유료 서비스보다 낮을 수 있습니다")


def render_minimal_sidebar() -> Dict[str, Any]:
    """
    간단한 사용을 위한 최소 사이드바 렌더링

    Returns:
        선택된 설정이 담긴 딕셔너리
    """
    settings = {}

    with st.sidebar:
        st.markdown("## 설정")

        # 간단한 언어 선택
        settings["source_lang"] = st.selectbox(
            "원본 언어",
            options=["en", "zh", "ja", "ko", "de", "fr", "es"],
            format_func=lambda x: LANGUAGES.get(x, x),
            key="mini_source"
        )

        settings["target_lang"] = st.selectbox(
            "번역 언어",
            options=["ko", "en", "zh", "ja", "de", "fr", "es"],
            format_func=lambda x: LANGUAGES.get(x, x),
            key="mini_target"
        )

        settings["service"] = "google"  # 기본으로 무료 서비스 사용

    return settings

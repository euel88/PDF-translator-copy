"""
PDFMathTranslate - Streamlit Application
Based on: https://github.com/PDFMathTranslate/PDFMathTranslate

수학 공식을 보존하며 과학 PDF를 번역하는 인터페이스
"""

import streamlit as st
import os
import sys
import tempfile
import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from config import APP_CONFIG, LANGUAGES, TranslationService, OutputFormat
from ui.styles import apply_custom_styles
from ui.sidebar import render_sidebar
from core.pdf_processor import PDFProcessor, PDFInfo
from core.translator import PDFTranslator, TranslationStatus, TranslationProgress

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_page():
    """Streamlit 페이지 설정 초기화"""
    st.set_page_config(
        page_title="PDF 수학 번역기",
        page_icon="📚",
        layout=APP_CONFIG.layout,
        initial_sidebar_state="expanded"
    )
    apply_custom_styles()


def init_session_state():
    """세션 상태 변수 초기화"""
    defaults = {
        "pdf2zh_available": False,
        "translation_in_progress": False,
        "current_file": None,
        "translation_result": None,
        "error_message": None,
        "openai_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_pdf2zh() -> bool:
    """pdf2zh 사용 가능 여부 확인 및 초기화"""
    try:
        import pdf2zh
        from pdf2zh.doclayout import ModelInstance, OnnxModel

        if ModelInstance.value is None:
            with st.spinner("레이아웃 감지 모델 로딩 중..."):
                ModelInstance.value = OnnxModel.from_pretrained()

        st.session_state.pdf2zh_available = True
        return True

    except ImportError:
        st.session_state.pdf2zh_available = False
        return False
    except Exception as e:
        logger.error(f"pdf2zh 초기화 실패: {e}")
        st.session_state.pdf2zh_available = False
        return False


def render_header():
    """애플리케이션 헤더 렌더링"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #1f1f1f; margin-bottom: 0.5rem;">
            PDF 수학 번역기
        </h1>
        <p style="color: #666; font-size: 1.1rem;">
            수학 공식과 레이아웃을 보존하며 과학 PDF를 번역합니다
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar():
    """시스템 상태 바 렌더링"""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.pdf2zh_available:
            st.success("pdf2zh: 준비됨")
        else:
            st.warning("pdf2zh: 초기화 필요")

    with col2:
        if st.button("pdf2zh 초기화", use_container_width=True):
            if check_pdf2zh():
                st.success("초기화 완료!")
                st.rerun()
            else:
                st.error("초기화 실패")

    with col3:
        st.info(f"버전: {APP_CONFIG.version}")


def render_file_upload() -> Optional[Any]:
    """파일 업로드 섹션 렌더링"""
    st.markdown("### PDF 업로드")

    uploaded_file = st.file_uploader(
        "번역할 PDF 파일을 선택하세요",
        type=["pdf"],
        help="최대 파일 크기: 200MB. 과학 논문에 최적화되어 있습니다.",
        label_visibility="collapsed"
    )

    if uploaded_file:
        # 파일 정보
        file_size_mb = uploaded_file.size / (1024 * 1024)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"**{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        with col2:
            if file_size_mb > 50:
                st.warning("대용량 파일")

        # 세션 상태에 저장
        st.session_state.current_file = uploaded_file

    return uploaded_file


def render_translation_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """메인 영역에서 추가 번역 설정 렌더링"""
    with st.expander("고급 설정", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # 페이지 범위
            use_all_pages = st.checkbox("모든 페이지 번역", value=True)
            if not use_all_pages:
                settings["pages"] = st.text_input(
                    "페이지 범위",
                    placeholder="예: 1-5, 8, 10-12",
                    help="번역할 페이지를 지정하세요"
                )

        with col2:
            # 커스텀 프롬프트 (LLM 기반 서비스용)
            if settings.get("service") in ["openai", "deepseek", "gemini", "ollama"]:
                settings["custom_prompt"] = st.text_area(
                    "사용자 정의 프롬프트 (선택사항)",
                    placeholder="번역을 위한 추가 지시사항...",
                    height=80
                )

    return settings


def translate_pdf(
    file_path: str,
    settings: Dict[str, Any],
    progress_placeholder
) -> Optional[Dict[str, Any]]:
    """
    PDF 번역 실행

    Returns dict with output file paths or None on failure.
    """
    try:
        from pdf2zh import translate
        from pdf2zh.doclayout import ModelInstance

        # 매개변수 준비
        envs = {}
        service = settings.get("service", "google")

        # 서비스별 환경 변수
        if service == "openai" and settings.get("openai_api_key"):
            envs["OPENAI_API_KEY"] = settings["openai_api_key"]
            if settings.get("openai_base_url"):
                envs["OPENAI_BASE_URL"] = settings["openai_base_url"]
        elif service == "deepl" and settings.get("deepl_api_key"):
            envs["DEEPL_API_KEY"] = settings["deepl_api_key"]
        elif service == "deepseek" and settings.get("deepseek_api_key"):
            envs["DEEPSEEK_API_KEY"] = settings["deepseek_api_key"]
        elif service == "gemini" and settings.get("gemini_api_key"):
            envs["GEMINI_API_KEY"] = settings["gemini_api_key"]

        # 페이지 파싱
        pages = None
        if settings.get("pages"):
            try:
                pages = []
                for part in settings["pages"].replace(" ", "").split(","):
                    if "-" in part:
                        start, end = part.split("-")
                        pages.extend(range(int(start) - 1, int(end)))
                    else:
                        pages.append(int(part) - 1)
            except ValueError:
                logger.warning("잘못된 페이지 범위, 전체 페이지 사용")
                pages = None

        # 출력 디렉토리 생성
        output_dir = settings.get("download_path", tempfile.mkdtemp())
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 진행 상황 업데이트
        progress_placeholder.info("번역 시작 중...")

        # 번역 매개변수 구성
        translate_params = {
            "files": [file_path],
            "output": output_dir,
            "lang_in": settings.get("source_lang", "en"),
            "lang_out": settings.get("target_lang", "ko"),
            "service": service,
            "thread": settings.get("thread_count", 4),
            "model": ModelInstance.value,
            "envs": envs if envs else None,
            "skip_subset_fonts": True,
        }

        if pages:
            translate_params["pages"] = pages

        if settings.get("openai_model") and service == "openai":
            translate_params["model_name"] = settings["openai_model"]

        if settings.get("ollama_model") and service == "ollama":
            translate_params["model_name"] = settings["ollama_model"]
            if settings.get("ollama_host"):
                if not envs:
                    envs = {}
                envs["OLLAMA_HOST"] = settings["ollama_host"]
                translate_params["envs"] = envs

        # 번역 실행
        progress_placeholder.info("번역 중... 몇 분 정도 소요될 수 있습니다.")

        result = translate(**translate_params)

        # 출력 파일 가져오기
        base_name = Path(file_path).stem
        mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_file = Path(output_dir) / f"{base_name}-dual.pdf"

        return {
            "success": True,
            "mono_path": str(mono_file) if mono_file.exists() else None,
            "dual_path": str(dual_file) if dual_file.exists() else None,
            "output_dir": output_dir,
        }

    except ImportError as e:
        logger.error(f"Import 오류: {e}")
        return {"success": False, "error": "pdf2zh를 사용할 수 없습니다. 먼저 초기화해주세요."}

    except Exception as e:
        logger.error(f"번역 오류: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def render_translation_button(uploaded_file, settings: Dict[str, Any]):
    """번역 버튼 렌더링 및 번역 처리"""

    # 설정 유효성 검사
    service = settings.get("service", "google")
    can_translate = True
    warning_message = None

    if service == "openai" and not settings.get("openai_api_key"):
        can_translate = False
        warning_message = "OpenAI API 키가 필요합니다"
    elif service == "deepl" and not settings.get("deepl_api_key"):
        can_translate = False
        warning_message = "DeepL API 키가 필요합니다"
    elif service == "deepseek" and not settings.get("deepseek_api_key"):
        can_translate = False
        warning_message = "DeepSeek API 키가 필요합니다"
    elif service == "gemini" and not settings.get("gemini_api_key"):
        can_translate = False
        warning_message = "Gemini API 키가 필요합니다"

    if not st.session_state.pdf2zh_available and service != "google":
        can_translate = False
        warning_message = "먼저 pdf2zh를 초기화해주세요"

    if warning_message:
        st.warning(warning_message)

    # 번역 버튼
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        translate_clicked = st.button(
            "PDF 번역하기",
            type="primary",
            disabled=not can_translate or not uploaded_file,
            use_container_width=True
        )

    if translate_clicked and uploaded_file:
        # 진행 상황 플레이스홀더
        progress_placeholder = st.empty()
        progress_placeholder.info("번역 준비 중...")

        # 업로드된 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            input_path = tmp.name

        try:
            # 번역 실행
            result = translate_pdf(input_path, settings, progress_placeholder)

            if result and result.get("success"):
                progress_placeholder.success("번역 완료!")
                st.balloons()

                # 다운로드 버튼
                st.markdown("### 결과 다운로드")

                col1, col2 = st.columns(2)

                output_format = settings.get("output_format", "dual")

                if output_format in ["mono", "both"] and result.get("mono_path"):
                    if os.path.exists(result["mono_path"]):
                        with col1:
                            with open(result["mono_path"], "rb") as f:
                                st.download_button(
                                    "번역본만 다운로드",
                                    f.read(),
                                    f"{uploaded_file.name.replace('.pdf', '')}_번역본.pdf",
                                    "application/pdf",
                                    use_container_width=True
                                )

                if output_format in ["dual", "both"] and result.get("dual_path"):
                    if os.path.exists(result["dual_path"]):
                        with col2:
                            with open(result["dual_path"], "rb") as f:
                                st.download_button(
                                    "이중언어 PDF 다운로드",
                                    f.read(),
                                    f"{uploaded_file.name.replace('.pdf', '')}_이중언어.pdf",
                                    "application/pdf",
                                    use_container_width=True
                                )

                # 저장 경로 표시
                if settings.get("download_path"):
                    st.info(f"파일 저장 위치: {result.get('output_dir', settings['download_path'])}")

            else:
                error_msg = result.get("error", "알 수 없는 오류") if result else "번역 실패"
                progress_placeholder.error(f"번역 실패: {error_msg}")

        except Exception as e:
            st.error(f"오류: {e}")
            logger.error(f"번역 오류: {e}", exc_info=True)

        finally:
            # 입력 파일 정리
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except Exception:
                pass

            # 메모리 정리
            gc.collect()


def render_info_tab():
    """정보 탭 렌더링"""
    st.markdown("""
    ### PDF 수학 번역기 소개

    이 애플리케이션은 과학 PDF 문서를 번역하면서 다음을 보존합니다:
    - 수학 공식 및 방정식
    - 문서 레이아웃 및 구조
    - 표와 그림
    - 참조 및 인용

    ### 지원 서비스

    | 서비스 | API 키 | 품질 | 속도 |
    |--------|--------|------|------|
    | Google 번역 | 불필요 | 양호 | 빠름 |
    | OpenAI GPT (ChatGPT) | 필요 | 우수 | 보통 |
    | DeepL | 필요 | 우수 | 빠름 |
    | Ollama | 불필요 (로컬) | 양호 | 환경에 따라 다름 |
    | DeepSeek | 필요 | 매우 좋음 | 보통 |
    | Gemini | 필요 | 매우 좋음 | 보통 |

    ### ChatGPT API 사용 방법

    1. [OpenAI Platform](https://platform.openai.com)에서 API 키 발급
    2. 사이드바에서 "OpenAI GPT (ChatGPT)" 선택
    3. API 키 입력
    4. 모델 선택 (gpt-4o-mini 권장)
    5. PDF 업로드 후 번역 실행

    ### 최상의 결과를 위한 팁

    1. **pdf2zh 사용**: 더 나은 레이아웃 보존을 위해 pdf2zh 초기화
    2. **학술 논문**: 학술 PDF에 최적화되어 있습니다
    3. **페이지 선택**: 대용량 문서는 먼저 특정 페이지만 번역해보세요
    4. **서비스 선택**: Google은 무료; OpenAI/DeepL은 더 좋은 품질 제공

    ### 제한 사항

    - 최대 파일 크기: 200MB
    - 클라우드 메모리 제한: ~1GB
    - 스캔된 PDF는 OCR이 필요할 수 있습니다

    ### 링크

    - [GitHub - PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate)
    - [문서](https://github.com/PDFMathTranslate/PDFMathTranslate/blob/main/docs/ADVANCED.md)
    """)


def main():
    """메인 애플리케이션 진입점"""
    # 초기화
    init_page()
    init_session_state()

    # 헤더
    render_header()

    # 상태 바
    render_status_bar()

    st.divider()

    # 사이드바 설정
    settings = render_sidebar()

    # 메인 콘텐츠 탭
    tab_translate, tab_info = st.tabs(["번역", "정보"])

    with tab_translate:
        # 파일 업로드
        uploaded_file = render_file_upload()

        if uploaded_file:
            st.divider()

            # 추가 설정
            settings = render_translation_settings(settings)

            st.divider()

            # 번역 버튼
            render_translation_button(uploaded_file, settings)

    with tab_info:
        render_info_tab()

    # 푸터
    st.divider()
    st.caption(
        f"**{APP_CONFIG.app_name}** v{APP_CONFIG.version} | "
        "[PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate) 기반"
    )


if __name__ == "__main__":
    main()

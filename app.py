"""
PDFMathTranslate - Streamlit Application
Based on: https://github.com/PDFMathTranslate/PDFMathTranslate

A streamlined interface for translating scientific PDFs with math formula preservation.
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
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title=APP_CONFIG.page_title,
        page_icon=APP_CONFIG.page_icon,
        layout=APP_CONFIG.layout,
        initial_sidebar_state="expanded"
    )
    apply_custom_styles()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "pdf2zh_available": False,
        "translation_in_progress": False,
        "current_file": None,
        "translation_result": None,
        "error_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_pdf2zh() -> bool:
    """Check if pdf2zh is available and initialize it."""
    try:
        import pdf2zh
        from pdf2zh.doclayout import ModelInstance, OnnxModel

        if ModelInstance.value is None:
            with st.spinner("Loading layout detection model..."):
                ModelInstance.value = OnnxModel.from_pretrained()

        st.session_state.pdf2zh_available = True
        return True

    except ImportError:
        st.session_state.pdf2zh_available = False
        return False
    except Exception as e:
        logger.error(f"pdf2zh initialization failed: {e}")
        st.session_state.pdf2zh_available = False
        return False


def render_header():
    """Render the application header."""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #1f1f1f; margin-bottom: 0.5rem;">
            PDF Math Translator
        </h1>
        <p style="color: #666; font-size: 1.1rem;">
            Translate scientific PDFs while preserving mathematical formulas and layouts
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_status_bar():
    """Render the system status bar."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.pdf2zh_available:
            st.success("pdf2zh: Ready")
        else:
            st.warning("pdf2zh: Not initialized")

    with col2:
        if st.button("Initialize pdf2zh", use_container_width=True):
            if check_pdf2zh():
                st.success("Initialized!")
                st.rerun()
            else:
                st.error("Failed to initialize")

    with col3:
        st.info(f"Version: {APP_CONFIG.version}")


def render_file_upload() -> Optional[Any]:
    """Render file upload section."""
    st.markdown("### Upload PDF")

    uploaded_file = st.file_uploader(
        "Select a PDF file to translate",
        type=["pdf"],
        help="Maximum file size: 200MB. Works best with scientific papers.",
        label_visibility="collapsed"
    )

    if uploaded_file:
        # File info
        file_size_mb = uploaded_file.size / (1024 * 1024)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"**{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        with col2:
            if file_size_mb > 50:
                st.warning("Large file")

        # Save to session state
        st.session_state.current_file = uploaded_file

    return uploaded_file


def render_translation_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Render additional translation settings in the main area."""
    with st.expander("Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Page range
            use_all_pages = st.checkbox("Translate all pages", value=True)
            if not use_all_pages:
                settings["pages"] = st.text_input(
                    "Page range",
                    placeholder="e.g., 1-5, 8, 10-12",
                    help="Specify pages to translate"
                )

        with col2:
            # Custom prompt (for LLM-based services)
            if settings.get("service") in ["openai", "deepseek", "gemini", "ollama"]:
                settings["custom_prompt"] = st.text_area(
                    "Custom prompt (optional)",
                    placeholder="Additional instructions for translation...",
                    height=80
                )

    return settings


def translate_pdf(
    file_path: str,
    settings: Dict[str, Any],
    progress_placeholder
) -> Optional[Dict[str, Any]]:
    """
    Execute PDF translation.

    Returns dict with output file paths or None on failure.
    """
    try:
        from pdf2zh import translate
        from pdf2zh.doclayout import ModelInstance

        # Prepare parameters
        envs = {}
        service = settings.get("service", "google")

        # Service-specific environment variables
        if service == "openai" and settings.get("openai_api_key"):
            envs["OPENAI_API_KEY"] = settings["openai_api_key"]
        elif service == "deepl" and settings.get("deepl_api_key"):
            envs["DEEPL_API_KEY"] = settings["deepl_api_key"]
        elif service == "deepseek" and settings.get("deepseek_api_key"):
            envs["DEEPSEEK_API_KEY"] = settings["deepseek_api_key"]
        elif service == "gemini" and settings.get("gemini_api_key"):
            envs["GEMINI_API_KEY"] = settings["gemini_api_key"]

        # Parse pages
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
                logger.warning("Invalid page range, using all pages")
                pages = None

        # Create output directory
        output_dir = tempfile.mkdtemp()

        # Update progress
        progress_placeholder.info("Starting translation...")

        # Build translation parameters
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

        # Execute translation
        progress_placeholder.info("Translating... This may take a few minutes.")

        result = translate(**translate_params)

        # Get output files
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
        logger.error(f"Import error: {e}")
        return {"success": False, "error": "pdf2zh not available. Please initialize it first."}

    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def render_translation_button(uploaded_file, settings: Dict[str, Any]):
    """Render translation button and handle translation."""

    # Validate settings
    service = settings.get("service", "google")
    can_translate = True
    warning_message = None

    if service == "openai" and not settings.get("openai_api_key"):
        can_translate = False
        warning_message = "OpenAI API key required"
    elif service == "deepl" and not settings.get("deepl_api_key"):
        can_translate = False
        warning_message = "DeepL API key required"
    elif service == "deepseek" and not settings.get("deepseek_api_key"):
        can_translate = False
        warning_message = "DeepSeek API key required"
    elif service == "gemini" and not settings.get("gemini_api_key"):
        can_translate = False
        warning_message = "Gemini API key required"

    if not st.session_state.pdf2zh_available and service != "google":
        can_translate = False
        warning_message = "Please initialize pdf2zh first"

    if warning_message:
        st.warning(warning_message)

    # Translation button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        translate_clicked = st.button(
            "Translate PDF",
            type="primary",
            disabled=not can_translate or not uploaded_file,
            use_container_width=True
        )

    if translate_clicked and uploaded_file:
        # Progress placeholder
        progress_placeholder = st.empty()
        progress_placeholder.info("Preparing translation...")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            input_path = tmp.name

        try:
            # Execute translation
            result = translate_pdf(input_path, settings, progress_placeholder)

            if result and result.get("success"):
                progress_placeholder.success("Translation completed!")
                st.balloons()

                # Download buttons
                st.markdown("### Download Results")

                col1, col2 = st.columns(2)

                output_format = settings.get("output_format", "dual")

                if output_format in ["mono", "both"] and result.get("mono_path"):
                    if os.path.exists(result["mono_path"]):
                        with col1:
                            with open(result["mono_path"], "rb") as f:
                                st.download_button(
                                    "Download Translation Only",
                                    f.read(),
                                    f"{uploaded_file.name.replace('.pdf', '')}_translated.pdf",
                                    "application/pdf",
                                    use_container_width=True
                                )

                if output_format in ["dual", "both"] and result.get("dual_path"):
                    if os.path.exists(result["dual_path"]):
                        with col2:
                            with open(result["dual_path"], "rb") as f:
                                st.download_button(
                                    "Download Bilingual PDF",
                                    f.read(),
                                    f"{uploaded_file.name.replace('.pdf', '')}_bilingual.pdf",
                                    "application/pdf",
                                    use_container_width=True
                                )

            else:
                error_msg = result.get("error", "Unknown error") if result else "Translation failed"
                progress_placeholder.error(f"Translation failed: {error_msg}")

        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Translation error: {e}", exc_info=True)

        finally:
            # Cleanup input file
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
            except Exception:
                pass

            # Memory cleanup
            gc.collect()


def render_info_tab():
    """Render the information tab."""
    st.markdown("""
    ### About PDF Math Translator

    This application translates scientific PDF documents while preserving:
    - Mathematical formulas and equations
    - Document layout and structure
    - Tables and figures
    - References and citations

    ### Supported Services

    | Service | API Key | Quality | Speed |
    |---------|---------|---------|-------|
    | Google Translate | Not required | Good | Fast |
    | OpenAI GPT | Required | Excellent | Medium |
    | DeepL | Required | Excellent | Fast |
    | Ollama | Not required (local) | Good | Depends |
    | DeepSeek | Required | Very Good | Medium |
    | Gemini | Required | Very Good | Medium |

    ### Tips for Best Results

    1. **Use pdf2zh**: Initialize pdf2zh for better layout preservation
    2. **Scientific papers**: Works best with academic PDFs
    3. **Page selection**: For large documents, translate specific pages first
    4. **Service selection**: Google is free; OpenAI/DeepL give better quality

    ### Limitations

    - Maximum file size: 200MB
    - Memory limit on cloud: ~1GB
    - Scanned PDFs may require OCR (not available in lite version)

    ### Links

    - [PDFMathTranslate on GitHub](https://github.com/PDFMathTranslate/PDFMathTranslate)
    - [Documentation](https://github.com/PDFMathTranslate/PDFMathTranslate/blob/main/docs/ADVANCED.md)
    """)


def main():
    """Main application entry point."""
    # Initialize
    init_page()
    init_session_state()

    # Header
    render_header()

    # Status bar
    render_status_bar()

    st.divider()

    # Sidebar settings
    settings = render_sidebar()

    # Main content tabs
    tab_translate, tab_info = st.tabs(["Translate", "Information"])

    with tab_translate:
        # File upload
        uploaded_file = render_file_upload()

        if uploaded_file:
            st.divider()

            # Additional settings
            settings = render_translation_settings(settings)

            st.divider()

            # Translation button
            render_translation_button(uploaded_file, settings)

    with tab_info:
        render_info_tab()

    # Footer
    st.divider()
    st.caption(
        f"**{APP_CONFIG.app_name}** v{APP_CONFIG.version} | "
        "Based on [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate)"
    )


if __name__ == "__main__":
    main()

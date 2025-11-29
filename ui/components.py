"""
Reusable Streamlit UI components.
"""
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
import base64

from utils.languages import LANGUAGE_CODES, get_source_languages, get_target_languages
from services.providers import SUPPORTED_SERVICES, get_translation_service
from config import TranslationService, SERVICE_NAMES


def file_uploader(
    key: str = "pdf_uploader",
    label: str = "Upload PDF",
    help_text: str = "Select a PDF file to translate (max 200MB)"
) -> Optional[Any]:
    """
    Render a styled file uploader for PDF files.

    Args:
        key: Unique key for the uploader
        label: Label text
        help_text: Help text shown below

    Returns:
        Uploaded file object or None
    """
    uploaded_file = st.file_uploader(
        label,
        type=["pdf"],
        key=key,
        help=help_text,
        accept_multiple_files=False
    )

    if uploaded_file:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.caption(f"File: {uploaded_file.name} ({file_size_mb:.2f} MB)")

    return uploaded_file


def language_selector(
    key_prefix: str = "lang",
    default_source: str = "en",
    default_target: str = "ko",
    show_swap: bool = True
) -> Tuple[str, str]:
    """
    Render source and target language selectors.

    Args:
        key_prefix: Prefix for session state keys
        default_source: Default source language code
        default_target: Default target language code
        show_swap: Whether to show swap button

    Returns:
        Tuple of (source_lang, target_lang) codes
    """
    # Get language options
    all_langs = LANGUAGE_CODES

    # Create columns
    col1, col2, col3 = st.columns([5, 1, 5]) if show_swap else st.columns([1, 1])

    with col1:
        source_options = list(all_langs.keys())
        source_idx = source_options.index(default_source) if default_source in source_options else 0

        source_lang = st.selectbox(
            "Source Language",
            options=source_options,
            format_func=lambda x: all_langs.get(x, x),
            index=source_idx,
            key=f"{key_prefix}_source"
        )

    if show_swap:
        with col2:
            st.write("")  # Spacing
            st.write("")
            if st.button("⇄", key=f"{key_prefix}_swap", help="Swap languages"):
                # Swap logic handled by session state
                pass

    with col3 if show_swap else col2:
        target_options = list(all_langs.keys())
        target_idx = target_options.index(default_target) if default_target in target_options else 0

        target_lang = st.selectbox(
            "Target Language",
            options=target_options,
            format_func=lambda x: all_langs.get(x, x),
            index=target_idx,
            key=f"{key_prefix}_target"
        )

    return source_lang, target_lang


def service_selector(
    key: str = "service_selector",
    default_service: str = "google"
) -> Tuple[str, Dict[str, Any]]:
    """
    Render translation service selector with configuration.

    Args:
        key: Unique key for the selector
        default_service: Default service ID

    Returns:
        Tuple of (service_id, service_config)
    """
    # Service options
    service_options = {
        "google": "Google Translate (Free)",
        "openai": "OpenAI GPT (API Key Required)",
        "deepl": "DeepL (API Key Required)",
        "ollama": "Ollama (Local)",
        "deepseek": "DeepSeek (API Key Required)",
        "gemini": "Google Gemini (API Key Required)",
    }

    service_id = st.selectbox(
        "Translation Service",
        options=list(service_options.keys()),
        format_func=lambda x: service_options.get(x, x),
        index=list(service_options.keys()).index(default_service) if default_service in service_options else 0,
        key=key
    )

    # Service-specific configuration
    config = {}

    if service_id == "openai":
        with st.expander("OpenAI Settings", expanded=True):
            config["api_key"] = st.text_input(
                "API Key",
                type="password",
                key=f"{key}_openai_key",
                help="Enter your OpenAI API key"
            )
            config["model"] = st.selectbox(
                "Model",
                options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                key=f"{key}_openai_model"
            )

    elif service_id == "deepl":
        with st.expander("DeepL Settings", expanded=True):
            config["api_key"] = st.text_input(
                "API Key",
                type="password",
                key=f"{key}_deepl_key",
                help="Enter your DeepL API key"
            )

    elif service_id == "ollama":
        with st.expander("Ollama Settings", expanded=True):
            config["host"] = st.text_input(
                "Host URL",
                value="http://localhost:11434",
                key=f"{key}_ollama_host"
            )
            config["model"] = st.text_input(
                "Model Name",
                value="gemma2",
                key=f"{key}_ollama_model",
                help="e.g., gemma2, llama3, mistral"
            )

    elif service_id == "deepseek":
        with st.expander("DeepSeek Settings", expanded=True):
            config["api_key"] = st.text_input(
                "API Key",
                type="password",
                key=f"{key}_deepseek_key"
            )

    elif service_id == "gemini":
        with st.expander("Gemini Settings", expanded=True):
            config["api_key"] = st.text_input(
                "API Key",
                type="password",
                key=f"{key}_gemini_key"
            )

    return service_id, config


def progress_display(
    progress: float,
    message: str = "",
    status: str = "in_progress"
):
    """
    Display translation progress.

    Args:
        progress: Progress value (0-100)
        message: Status message
        status: Status type ("in_progress", "completed", "failed")
    """
    # Progress bar
    progress_bar = st.progress(int(progress) / 100)

    # Status indicator
    if status == "completed":
        st.success(f"✅ {message}" if message else "✅ Translation completed!")
    elif status == "failed":
        st.error(f"❌ {message}" if message else "❌ Translation failed")
    else:
        st.info(f"⏳ {message}" if message else f"⏳ Translating... {int(progress)}%")


def download_button(
    file_path: str,
    filename: str,
    label: str = "Download",
    mime_type: str = "application/pdf"
) -> bool:
    """
    Render a download button for a file.

    Args:
        file_path: Path to the file
        filename: Download filename
        label: Button label
        mime_type: File MIME type

    Returns:
        True if button was clicked
    """
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()

        return st.download_button(
            label=label,
            data=file_data,
            file_name=filename,
            mime=mime_type
        )
    except Exception as e:
        st.error(f"Error preparing download: {e}")
        return False


def download_button_from_bytes(
    data: bytes,
    filename: str,
    label: str = "Download",
    mime_type: str = "application/pdf"
) -> bool:
    """
    Render a download button for byte data.

    Args:
        data: File data as bytes
        filename: Download filename
        label: Button label
        mime_type: File MIME type

    Returns:
        True if button was clicked
    """
    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )


def pdf_preview(
    file_path: Optional[str] = None,
    file_data: Optional[bytes] = None,
    page_num: int = 0,
    width: int = 400
):
    """
    Display a PDF page preview.

    Args:
        file_path: Path to PDF file
        file_data: Or PDF data as bytes
        page_num: Page number to preview (0-indexed)
        width: Image width
    """
    try:
        import fitz

        if file_path:
            doc = fitz.open(file_path)
        elif file_data:
            doc = fitz.open(stream=file_data, filetype="pdf")
        else:
            st.warning("No PDF provided for preview")
            return

        if page_num >= len(doc):
            page_num = 0

        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")

        # Display image
        st.image(img_data, width=width, caption=f"Page {page_num + 1} of {len(doc)}")

        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page_num > 0:
                if st.button("← Previous"):
                    st.session_state.preview_page = page_num - 1
        with col3:
            if page_num < len(doc) - 1:
                if st.button("Next →"):
                    st.session_state.preview_page = page_num + 1

        doc.close()

    except ImportError:
        st.info("PDF preview requires PyMuPDF (fitz)")
    except Exception as e:
        st.error(f"Error displaying preview: {e}")


def error_display(message: str, details: Optional[str] = None):
    """Display an error message."""
    st.error(f"❌ {message}")
    if details:
        with st.expander("Error Details"):
            st.code(details)


def success_display(message: str, details: Optional[Dict] = None):
    """Display a success message."""
    st.success(f"✅ {message}")
    if details:
        with st.expander("Details"):
            for key, value in details.items():
                st.write(f"**{key}:** {value}")


def page_range_input(
    max_pages: int,
    key: str = "page_range"
) -> Optional[str]:
    """
    Input for selecting page range.

    Args:
        max_pages: Maximum number of pages
        key: Unique key

    Returns:
        Page range string or None for all pages
    """
    col1, col2 = st.columns([1, 2])

    with col1:
        all_pages = st.checkbox("All pages", value=True, key=f"{key}_all")

    with col2:
        if not all_pages:
            page_range = st.text_input(
                "Page range",
                placeholder="e.g., 1-5,8,10-12",
                key=f"{key}_range",
                help=f"Enter page numbers or ranges (1-{max_pages})"
            )
            return page_range if page_range else None

    return None


def output_format_selector(key: str = "output_format") -> str:
    """
    Select output format.

    Args:
        key: Unique key

    Returns:
        Output format ("dual", "mono", or "both")
    """
    format_options = {
        "dual": "Bilingual (Original + Translation)",
        "mono": "Translation Only",
        "both": "Generate Both"
    }

    return st.selectbox(
        "Output Format",
        options=list(format_options.keys()),
        format_func=lambda x: format_options[x],
        key=key
    )

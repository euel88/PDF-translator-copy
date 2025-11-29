"""
Sidebar component for the Streamlit app.
"""
import streamlit as st
from typing import Dict, Any, Optional

from config import APP_CONFIG, LANGUAGES, TranslationService, SERVICE_NAMES


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar with translation settings.

    Returns:
        Dictionary of selected settings
    """
    settings = {}

    with st.sidebar:
        # App title
        st.markdown("## ⚙️ Settings")
        st.divider()

        # Translation Service
        st.markdown("### Translation Service")

        service_options = {
            "google": "🌐 Google (Free)",
            "openai": "🤖 OpenAI GPT",
            "deepl": "📝 DeepL",
            "ollama": "💻 Ollama (Local)",
            "deepseek": "🔮 DeepSeek",
            "gemini": "✨ Gemini",
        }

        settings["service"] = st.selectbox(
            "Select Service",
            options=list(service_options.keys()),
            format_func=lambda x: service_options.get(x, x),
            key="sidebar_service",
            label_visibility="collapsed"
        )

        # Service-specific settings
        _render_service_settings(settings)

        st.divider()

        # Language Settings
        st.markdown("### Languages")

        lang_options = list(LANGUAGES.keys())

        settings["source_lang"] = st.selectbox(
            "Source Language",
            options=lang_options,
            format_func=lambda x: LANGUAGES.get(x, x),
            index=lang_options.index("en") if "en" in lang_options else 0,
            key="sidebar_source_lang"
        )

        settings["target_lang"] = st.selectbox(
            "Target Language",
            options=lang_options,
            format_func=lambda x: LANGUAGES.get(x, x),
            index=lang_options.index("ko") if "ko" in lang_options else 0,
            key="sidebar_target_lang"
        )

        st.divider()

        # Output Settings
        st.markdown("### Output")

        settings["output_format"] = st.radio(
            "Format",
            options=["dual", "mono", "both"],
            format_func=lambda x: {
                "dual": "📄 Bilingual",
                "mono": "📃 Translation Only",
                "both": "📚 Both Versions"
            }.get(x, x),
            key="sidebar_output_format",
            label_visibility="collapsed"
        )

        st.divider()

        # Advanced Settings
        with st.expander("⚡ Advanced", expanded=False):
            settings["thread_count"] = st.slider(
                "Threads",
                min_value=1,
                max_value=8,
                value=4,
                key="sidebar_threads"
            )

            settings["use_cache"] = st.checkbox(
                "Enable Cache",
                value=True,
                key="sidebar_cache"
            )

        st.divider()

        # Info section
        st.markdown("### About")
        st.caption(f"**{APP_CONFIG.app_name}** v{APP_CONFIG.version}")
        st.caption("Based on [PDFMathTranslate](https://github.com/PDFMathTranslate/PDFMathTranslate)")

        # Help link
        st.markdown("""
        ---
        📖 [Documentation](https://github.com/PDFMathTranslate/PDFMathTranslate)
        🐛 [Report Issue](https://github.com/PDFMathTranslate/PDFMathTranslate/issues)
        """)

    return settings


def _render_service_settings(settings: Dict[str, Any]):
    """Render service-specific settings."""
    service = settings.get("service", "google")

    if service == "openai":
        settings["openai_api_key"] = st.text_input(
            "OpenAI API Key",
            type="password",
            key="sidebar_openai_key",
            help="Required for OpenAI translation"
        )
        settings["openai_model"] = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            key="sidebar_openai_model"
        )

    elif service == "deepl":
        settings["deepl_api_key"] = st.text_input(
            "DeepL API Key",
            type="password",
            key="sidebar_deepl_key",
            help="Required for DeepL translation"
        )

    elif service == "ollama":
        settings["ollama_host"] = st.text_input(
            "Ollama Host",
            value="http://localhost:11434",
            key="sidebar_ollama_host"
        )
        settings["ollama_model"] = st.text_input(
            "Model",
            value="gemma2",
            key="sidebar_ollama_model",
            help="e.g., gemma2, llama3, mistral"
        )

    elif service == "deepseek":
        settings["deepseek_api_key"] = st.text_input(
            "DeepSeek API Key",
            type="password",
            key="sidebar_deepseek_key"
        )

    elif service == "gemini":
        settings["gemini_api_key"] = st.text_input(
            "Gemini API Key",
            type="password",
            key="sidebar_gemini_key"
        )


def render_minimal_sidebar() -> Dict[str, Any]:
    """
    Render a minimal sidebar for simple use cases.

    Returns:
        Dictionary of selected settings
    """
    settings = {}

    with st.sidebar:
        st.markdown("## Settings")

        # Simple language selection
        settings["source_lang"] = st.selectbox(
            "From",
            options=["en", "zh", "ja", "ko", "de", "fr", "es"],
            format_func=lambda x: LANGUAGES.get(x, x),
            key="mini_source"
        )

        settings["target_lang"] = st.selectbox(
            "To",
            options=["ko", "en", "zh", "ja", "de", "fr", "es"],
            format_func=lambda x: LANGUAGES.get(x, x),
            key="mini_target"
        )

        settings["service"] = "google"  # Use free service by default

    return settings

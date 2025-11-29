"""
Custom CSS styles for the Streamlit app.
"""
import streamlit as st


def get_custom_css() -> str:
    """Get custom CSS for the app."""
    return """
    <style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .app-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }

    .app-header h1 {
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .app-header p {
        color: #666;
        font-size: 1.1rem;
    }

    /* Card styling */
    .stcard {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e6e6e6;
        margin-bottom: 1rem;
    }

    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #4f8bf9;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8faff;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f8bf9 0%, #3b7dda 100%);
        color: white;
        border: none;
    }

    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4f8bf9 0%, #6c5ce7 100%);
        border-radius: 10px;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8faff 0%, #ffffff 100%);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.3rem;
        color: #1f1f1f;
    }

    /* Info box styling */
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #4f8bf9;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Success box styling */
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Error box styling */
    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* PDF preview container */
    .pdf-preview {
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        overflow: hidden;
        background: #f5f5f5;
    }

    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }

    .stats-card h3 {
        font-size: 2rem;
        margin: 0;
    }

    .stats-card p {
        margin: 0;
        opacity: 0.9;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-header h1 {
            font-size: 1.8rem;
        }
    }
    </style>
    """


def apply_custom_styles():
    """Apply custom CSS styles to the app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def create_card(content: str, title: str = "") -> str:
    """Create a styled card HTML."""
    title_html = f"<h4>{title}</h4>" if title else ""
    return f"""
    <div class="stcard">
        {title_html}
        {content}
    </div>
    """


def create_info_box(message: str) -> str:
    """Create an info box HTML."""
    return f'<div class="info-box">{message}</div>'


def create_success_box(message: str) -> str:
    """Create a success box HTML."""
    return f'<div class="success-box">{message}</div>'


def create_error_box(message: str) -> str:
    """Create an error box HTML."""
    return f'<div class="error-box">{message}</div>'


def create_stats_card(value: str, label: str) -> str:
    """Create a stats card HTML."""
    return f"""
    <div class="stats-card">
        <h3>{value}</h3>
        <p>{label}</p>
    </div>
    """

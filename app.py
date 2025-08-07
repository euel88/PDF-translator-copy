"""
PDF Math Translator Pro - Streamlit Cloud Stable Version
Full features with memory optimization
"""

import streamlit as st
import os
import sys
import tempfile
import gc
import subprocess
import logging
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, List, Dict
import shutil
import io

# Basic imports
import requests
import html
import json

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF Math Translator Pro",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
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
    .info-box {
        background: #e0e7ff;
        border: 1px solid #6366f1;
        padding: 1rem;
        border-radius: 8px;
        color: #312e81;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
if 'ocr_enabled' not in st.session_state:
    st.session_state.ocr_enabled = False
if 'pdf2zh_available' not in st.session_state:
    st.session_state.pdf2zh_available = False

def check_dependencies():
    """Check available dependencies"""
    deps = {
        'streamlit': False,
        'pdf2zh': False,
        'PyPDF2': False,
        'pymupdf': False,
        'openai': False,
        'PIL': False
    }
    
    for module in deps.keys():
        try:
            if module == 'pymupdf':
                __import__('fitz')
            elif module == 'PIL':
                __import__('PIL')
            else:
                __import__(module)
            deps[module] = True
        except ImportError:
            deps[module] = False
    
    return deps

def initialize_pdf2zh():
    """Initialize pdf2zh if available"""
    try:
        import pdf2zh
        from pdf2zh import translate
        from pdf2zh.doclayout import ModelInstance, OnnxModel
        
        if ModelInstance.value is None:
            with st.spinner("Loading layout detection model..."):
                ModelInstance.value = OnnxModel.from_pretrained()
        
        st.session_state.pdf2zh_available = True
        logger.info("pdf2zh initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"pdf2zh initialization failed: {e}")
        st.session_state.pdf2zh_available = False
        return False

def download_fonts():
    """Download required fonts"""
    font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)
    
    fonts = {
        "GoNotoKurrent-Regular.ttf": "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf",
        "NanumGothic.ttf": "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    }
    
    for font_name, url in fonts.items():
        font_path = font_dir / font_name
        if not font_path.exists():
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(font_path, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded font: {font_name}")
            except Exception as e:
                logger.error(f"Font download failed: {e}")

class SimplePDFTranslator:
    """Simple PDF translator with optional pdf2zh support"""
    
    def __init__(self, config):
        self.config = config
        self.use_pdf2zh = st.session_state.pdf2zh_available and config.get('use_pdf2zh', True)
    
    def translate_text_openai(self, text: str) -> str:
        """Translate using OpenAI API"""
        if not self.config.get('api_key'):
            return text
        
        try:
            import openai
            from openai import OpenAI
            
            client = OpenAI(api_key=self.config['api_key'])
            
            # Limit text length
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            response = client.chat.completions.create(
                model=self.config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate from {self.config['lang_from']} to {self.config['lang_to']}. Only return translation."
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=2000
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return text
    
    def translate_text_google(self, text: str) -> str:
        """Translate using Google Translate"""
        try:
            # Limit text length
            max_chars = 5000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            url = "https://translate.google.com/m"
            params = {
                'sl': self.config['lang_from'],
                'tl': self.config['lang_to'],
                'q': text
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                import re
                result = re.findall(r'class="(?:t0|result-container)">(.*?)<', response.text)
                if result:
                    return html.unescape(result[0])
            
            return text
            
        except Exception as e:
            logger.error(f"Google translation error: {e}")
            return text
    
    def translate_with_pdf2zh(self, input_path: str, output_dir: str, 
                             pages: Optional[List[int]] = None) -> Dict:
        """Translate using pdf2zh"""
        try:
            from pdf2zh import translate
            from pdf2zh.doclayout import ModelInstance
            
            # Set environment variables
            envs = {}
            if self.config.get('api_key'):
                if self.config['service'] == 'openai':
                    envs['OPENAI_API_KEY'] = self.config['api_key']
                    envs['OPENAI_MODEL'] = self.config.get('model', 'gpt-3.5-turbo')
            
            # Run pdf2zh
            result = translate(
                files=[input_path],
                output=output_dir,
                pages=pages,
                lang_in=self.config['lang_from'],
                lang_out=self.config['lang_to'],
                service=self.config['service'],
                thread=2,  # Limit threads
                model=ModelInstance.value,
                envs=envs,
                skip_subset_fonts=True,
                ignore_cache=False
            )
            
            # Get output files
            base_name = Path(input_path).stem
            mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
            dual_file = Path(output_dir) / f"{base_name}-dual.pdf"
            
            return {
                'success': True,
                'mono': str(mono_file),
                'dual': str(dual_file)
            }
            
        except Exception as e:
            logger.error(f"pdf2zh translation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def translate_simple(self, input_path: str, output_dir: str,
                        pages: Optional[List[int]] = None) -> Dict:
        """Simple translation without pdf2zh"""
        try:
            import fitz  # PyMuPDF
            
            base_name = Path(input_path).stem
            output_path = Path(output_dir) / f"{base_name}-translated.pdf"
            
            doc = fitz.open(input_path)
            
            # Process pages
            if pages is None:
                pages = list(range(len(doc)))
            
            for page_num in pages:
                page = doc[page_num]
                
                # Extract text blocks
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    if block[6] == 0:  # Text block
                        original_text = block[4].strip()
                        if original_text:
                            # Translate
                            if self.config['service'] == 'openai':
                                translated = self.translate_text_openai(original_text)
                            else:
                                translated = self.translate_text_google(original_text)
                            
                            # Try to replace text (simplified)
                            try:
                                rect = fitz.Rect(block[:4])
                                page.add_redact_annot(rect)
                            except:
                                pass
            
            # Apply redactions
            for page_num in pages:
                try:
                    doc[page_num].apply_redactions()
                except:
                    pass
            
            # Add translated text
            for page_num in pages:
                page = doc[page_num]
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    if block[6] == 0:
                        original_text = block[4].strip()
                        if original_text:
                            if self.config['service'] == 'openai':
                                translated = self.translate_text_openai(original_text)
                            else:
                                translated = self.translate_text_google(original_text)
                            
                            try:
                                rect = fitz.Rect(block[:4])
                                page.insert_textbox(
                                    rect,
                                    translated,
                                    fontsize=10,
                                    align=fitz.TEXT_ALIGN_LEFT
                                )
                            except:
                                pass
            
            # Save
            doc.save(str(output_path), deflate=True, garbage=3)
            doc.close()
            
            return {
                'success': True,
                'mono': str(output_path),
                'dual': None
            }
            
        except Exception as e:
            logger.error(f"Simple translation error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📐 PDF Math Translator Pro</h1>
        <p>Scientific Paper Translation with Layout Preservation</p>
        <p>Powered by pdf2zh & OpenAI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check dependencies
    deps = check_dependencies()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        working = [k for k, v in deps.items() if v]
        if working:
            st.success(f"✅ Available: {', '.join(working)}")
        missing = [k for k, v in deps.items() if not v]
        if missing:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
    
    with col2:
        if st.button("🔄 Initialize pdf2zh"):
            if initialize_pdf2zh():
                st.success("✅ pdf2zh ready")
                st.rerun()
            else:
                st.error("❌ pdf2zh failed")
    
    with col3:
        st.metric("pdf2zh", "Ready" if st.session_state.pdf2zh_available else "Not ready")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Translation service
        st.subheader("🌐 Translation Service")
        service = st.selectbox(
            "Service",
            ["google", "openai"],
            help="Google is free, OpenAI needs API key"
        )
        
        api_key = ""
        model = "gpt-3.5-turbo"
        
        if service == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_key,
                placeholder="sk-..."
            )
            
            if api_key:
                st.session_state.api_key = api_key
                st.success("✅ API key set")
            else:
                st.warning("⚠️ API key required")
            
            model = st.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4o-mini"],
                help="gpt-3.5-turbo is cheaper"
            )
        
        # Languages
        st.subheader("🌍 Languages")
        lang_map = {
            "English": "en",
            "Korean": "ko",
            "Chinese (Simplified)": "zh",
            "Japanese": "ja",
            "Spanish": "es",
            "French": "fr",
            "German": "de"
        }
        
        source_lang = st.selectbox("Source", list(lang_map.keys()), index=0)
        target_lang = st.selectbox("Target", list(lang_map.keys()), index=1)
        
        # Options
        with st.expander("🔧 Advanced"):
            use_pdf2zh = st.checkbox(
                "Use pdf2zh (if available)",
                value=True,
                help="Better layout preservation"
            )
            
            pages_input = st.text_input(
                "Pages",
                placeholder="e.g., 1-10, 15",
                help="Leave empty for all"
            )
    
    # Main area
    tab1, tab2 = st.tabs(["📤 Translate", "ℹ️ Info"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose PDF file",
            type=['pdf'],
            help="Works best with scientific papers"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ File: {uploaded_file.name}")
                file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.info(f"Size: {file_size:.1f} MB")
                
                if file_size > 10:
                    st.warning("⚠️ Large files may take time")
            
            with col2:
                # Translate button
                can_translate = True
                if service == "openai" and not api_key:
                    st.error("API key required")
                    can_translate = False
                
                if st.button("🚀 Start Translation", 
                           type="primary", 
                           disabled=not can_translate,
                           use_container_width=True):
                    
                    # Progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Download fonts
                    download_fonts()
                    
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        input_path = tmp.name
                    
                    output_dir = tempfile.mkdtemp()
                    
                    try:
                        # Parse pages
                        pages_list = None
                        if pages_input:
                            try:
                                pages_list = []
                                for p in pages_input.split(','):
                                    p = p.strip()
                                    if '-' in p:
                                        start, end = p.split('-')
                                        pages_list.extend(range(int(start)-1, int(end)))
                                    else:
                                        pages_list.append(int(p)-1)
                            except:
                                st.error("Invalid page range")
                        
                        # Translation config
                        config = {
                            'service': service,
                            'api_key': api_key,
                            'model': model,
                            'lang_from': lang_map[source_lang],
                            'lang_to': lang_map[target_lang],
                            'use_pdf2zh': use_pdf2zh and st.session_state.pdf2zh_available
                        }
                        
                        # Create translator
                        translator = SimplePDFTranslator(config)
                        
                        # Translate
                        status_text.text("Translating...")
                        progress_bar.progress(0.5)
                        
                        if config['use_pdf2zh']:
                            result = translator.translate_with_pdf2zh(
                                input_path, output_dir, pages_list
                            )
                        else:
                            result = translator.translate_simple(
                                input_path, output_dir, pages_list
                            )
                        
                        progress_bar.progress(1.0)
                        
                        if result['success']:
                            st.balloons()
                            status_text.text("✅ Translation complete!")
                            
                            # Download buttons
                            if result.get('mono') and os.path.exists(result['mono']):
                                with open(result['mono'], 'rb') as f:
                                    st.download_button(
                                        "📥 Download Translated PDF",
                                        f.read(),
                                        f"{uploaded_file.name.replace('.pdf', '')}_translated.pdf",
                                        "application/pdf",
                                        use_container_width=True
                                    )
                            
                            if result.get('dual') and os.path.exists(result['dual']):
                                with open(result['dual'], 'rb') as f:
                                    st.download_button(
                                        "📥 Download Dual-language PDF",
                                        f.read(),
                                        f"{uploaded_file.name.replace('.pdf', '')}_dual.pdf",
                                        "application/pdf",
                                        use_container_width=True
                                    )
                        else:
                            st.error(f"❌ Translation failed: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Translation error: {e}", exc_info=True)
                    
                    finally:
                        # Cleanup
                        try:
                            if os.path.exists(input_path):
                                os.unlink(input_path)
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir, ignore_errors=True)
                        except:
                            pass
                        
                        # Memory cleanup
                        gc.collect()
    
    with tab2:
        st.markdown("""
        ### ℹ️ PDF Math Translator Pro
        
        **Features:**
        - ✅ Layout preservation with pdf2zh
        - ✅ Math formula preservation
        - ✅ Multi-language support
        - ✅ Free Google translation
        - ✅ High-quality OpenAI translation
        
        **Tips:**
        - Use pdf2zh for better results
        - Google translation is free but lower quality
        - OpenAI gives better translation but needs API key
        - Large files may take several minutes
        
        **Limitations:**
        - Streamlit Cloud: 1GB RAM limit
        - Max file size: 200MB
        - OCR not available in lite version
        
        **Links:**
        - [GitHub: PDFMathTranslate](https://github.com/Byaidu/PDFMathTranslate)
        - [pdf2zh Documentation](https://pdf2zh.com)
        - [OpenAI Platform](https://platform.openai.com)
        """)

if __name__ == "__main__":
    main()

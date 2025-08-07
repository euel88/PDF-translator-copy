"""
Configuration Module - Simplified version for Streamlit Cloud
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """Translation configuration"""
    
    # Basic settings
    service: str = 'google'
    lang_from: str = 'en'
    lang_to: str = 'ko'
    
    # API settings
    api_key: Optional[str] = None
    model: Optional[str] = 'gpt-3.5-turbo'
    
    # Processing settings
    thread_count: int = 2
    batch_size: int = 5
    timeout: int = 30
    
    # Output settings
    create_dual: bool = True
    skip_subset_fonts: bool = True
    use_cache: bool = True
    
    def __post_init__(self):
        """Load API key from environment"""
        if not self.api_key:
            if self.service == 'openai':
                self.api_key = os.getenv('OPENAI_API_KEY')


def get_font_path(language: str) -> Optional[Path]:
    """Get font path for language"""
    font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
    
    font_mapping = {
        'ko': 'NanumGothic.ttf',
        'zh': 'SourceHanSerifCN-Regular.ttf',
        'ja': 'SourceHanSerifJP-Regular.ttf',
        'default': 'GoNotoKurrent-Regular.ttf'
    }
    
    font_name = font_mapping.get(language, font_mapping['default'])
    font_path = font_dir / font_name
    
    return font_path if font_path.exists() else None

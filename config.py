"""
Configuration Module - 메모리 최적화 버전
Streamlit Cloud용 경량 설정 관리
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """번역 설정 (최소한의 메모리 사용)"""
    
    # 기본 설정
    service: str = 'google'
    lang_from: str = 'en'
    lang_to: str = 'ko'
    
    # API 설정
    api_key: Optional[str] = None
    model: Optional[str] = 'gpt-4o-mini'
    
    # OCR 설정 (간소화)
    ocr_settings: Dict = field(default_factory=lambda: {
        'enable_ocr': False,
        'ocr_languages': ['en'],
        'confidence_threshold': 0.5,
        'replace_images': True,
        'overlay_text': True
    })
    
    # 처리 설정 (메모리 절약)
    thread_count: int = 2  # 줄임
    batch_size: int = 5    # 줄임
    timeout: int = 30
    
    # 출력 설정
    create_dual: bool = True
    skip_subset_fonts: bool = True
    use_cache: bool = True
    
    def __post_init__(self):
        """환경 변수에서 API 키 로드"""
        if not self.api_key:
            if self.service == 'openai':
                self.api_key = os.getenv('OPENAI_API_KEY')
            elif self.service == 'deepl':
                self.api_key = os.getenv('DEEPL_AUTH_KEY')
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (직렬화용)"""
        return {
            'service': self.service,
            'lang_from': self.lang_from,
            'lang_to': self.lang_to,
            'api_key': '***' if self.api_key else None,
            'model': self.model,
            'ocr_settings': self.ocr_settings,
            'thread_count': self.thread_count,
            'create_dual': self.create_dual,
            'skip_subset_fonts': self.skip_subset_fonts
        }


class LightweightConfig:
    """경량 설정 관리자"""
    
    def __init__(self):
        self.translation_config = TranslationConfig()
        self.font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
        self.font_dir.mkdir(parents=True, exist_ok=True)
    
    def get_font_path(self, language: str) -> Optional[Path]:
        """언어별 폰트 경로"""
        font_mapping = {
            'ko': 'NanumGothic.ttf',
            'zh': 'SourceHanSerifCN-Regular.ttf',
            'ja': 'SourceHanSerifJP-Regular.ttf',
            'default': 'GoNotoKurrent-Regular.ttf'
        }
        
        font_name = font_mapping.get(language, font_mapping['default'])
        font_path = self.font_dir / font_name
        
        return font_path if font_path.exists() else None
    
    def update(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.translation_config, key):
                setattr(self.translation_config, key, value)
    
    def get_memory_limit(self) -> int:
        """메모리 제한 (MB)"""
        # Streamlit Cloud 제한
        return 1024  # 1GB
    
    def get_max_file_size(self) -> int:
        """최대 파일 크기 (MB)"""
        return 200  # Streamlit 기본값


# 전역 설정 인스턴스 (싱글톤)
_config = None

def get_config() -> LightweightConfig:
    """전역 설정 반환"""
    global _config
    if _config is None:
        _config = LightweightConfig()
    return _config

def reset_config():
    """설정 초기화"""
    global _config
    _config = None

"""
Configuration Module
번역 및 OCR 설정을 관리하는 모듈
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationConfig:
    """번역 설정 클래스"""
    
    # 기본 설정
    service: str = 'google'  # 번역 서비스 (google, openai, deepl, azure, ollama)
    lang_from: str = 'en'    # 원본 언어
    lang_to: str = 'ko'      # 대상 언어
    
    # API 설정
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    model: Optional[str] = None
    
    # OCR 설정
    ocr_settings: Dict = field(default_factory=lambda: {
        'enable_ocr': False,
        'ocr_languages': ['en'],
        'confidence_threshold': 0.5,
        'enhance_image': True,
        'replace_images': True,
        'overlay_text': True,
        'preserve_layout': True,
        'quality': 'medium'  # fast, medium, accurate
    })
    
    # 처리 설정
    thread_count: int = 4
    batch_size: int = 10
    timeout: int = 30
    retry_count: int = 3
    
    # 출력 설정
    output_format: str = 'pdf'  # pdf, docx, html
    create_dual: bool = True    # 대조본 생성 여부
    skip_subset_fonts: bool = True
    
    # 캐시 설정
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 캐시 디렉토리 설정
        if self.cache_dir is None:
            self.cache_dir = Path.home() / '.cache' / 'pdf2zh_ocr'
        
        # 캐시 디렉토리 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # API 키 환경 변수에서 로드
        if not self.api_key:
            self._load_api_key_from_env()
    
    def _load_api_key_from_env(self):
        """환경 변수에서 API 키 로드"""
        env_mapping = {
            'openai': 'OPENAI_API_KEY',
            'deepl': 'DEEPL_AUTH_KEY',
            'azure': 'AZURE_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'ollama': 'OLLAMA_API_KEY'
        }
        
        env_var = env_mapping.get(self.service)
        if env_var:
            self.api_key = os.getenv(env_var)
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            'service': self.service,
            'lang_from': self.lang_from,
            'lang_to': self.lang_to,
            'api_key': '***' if self.api_key else None,  # 보안을 위해 마스킹
            'api_endpoint': self.api_endpoint,
            'model': self.model,
            'ocr_settings': self.ocr_settings,
            'thread_count': self.thread_count,
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'output_format': self.output_format,
            'create_dual': self.create_dual,
            'skip_subset_fonts': self.skip_subset_fonts,
            'use_cache': self.use_cache,
            'cache_dir': str(self.cache_dir)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TranslationConfig':
        """딕셔너리에서 설정 생성"""
        # cache_dir을 Path 객체로 변환
        if 'cache_dir' in data and isinstance(data['cache_dir'], str):
            data['cache_dir'] = Path(data['cache_dir'])
        
        return cls(**data)
    
    def save(self, path: Path):
        """설정을 파일로 저장"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"설정 저장됨: {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'TranslationConfig':
        """파일에서 설정 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"설정 로드됨: {path}")
        return cls.from_dict(data)


@dataclass
class FontConfig:
    """폰트 설정 클래스"""
    
    # 언어별 기본 폰트
    font_mapping: Dict[str, str] = field(default_factory=lambda: {
        'ko': 'NanumGothic.ttf',
        'zh': 'SourceHanSerifCN-Regular.ttf',
        'zh-CN': 'SourceHanSerifCN-Regular.ttf',
        'zh-TW': 'SourceHanSerifTW-Regular.ttf',
        'ja': 'SourceHanSerifJP-Regular.ttf',
        'en': 'LiberationSans-Regular.ttf',
        'default': 'GoNotoKurrent-Regular.ttf'
    })
    
    # 폰트 디렉토리
    font_dir: Path = field(default_factory=lambda: Path.home() / '.cache' / 'pdf2zh' / 'fonts')
    
    # 폰트 크기 설정
    default_size: int = 12
    min_size: int = 6
    max_size: int = 72
    
    # 폰트 스타일
    bold: bool = False
    italic: bool = False
    
    def get_font_path(self, language: str) -> Optional[Path]:
        """언어에 맞는 폰트 경로 반환"""
        font_name = self.font_mapping.get(language, self.font_mapping['default'])
        font_path = self.font_dir / font_name
        
        if font_path.exists():
            return font_path
        
        # 대체 폰트 찾기
        for font_file in self.font_dir.glob('*.ttf'):
            return font_file
        
        return None
    
    def download_font(self, language: str):
        """필요한 폰트 다운로드"""
        font_urls = {
            'NanumGothic.ttf': 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf',
            'SourceHanSerifCN-Regular.ttf': 'https://github.com/adobe-fonts/source-han-serif/releases/download/2.001R/SourceHanSerifCN-Regular.otf',
            'GoNotoKurrent-Regular.ttf': 'https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf'
        }
        
        font_name = self.font_mapping.get(language, self.font_mapping['default'])
        
        if font_name in font_urls:
            import requests
            
            self.font_dir.mkdir(parents=True, exist_ok=True)
            font_path = self.font_dir / font_name
            
            if not font_path.exists():
                try:
                    logger.info(f"폰트 다운로드 중: {font_name}")
                    response = requests.get(font_urls[font_name], timeout=30)
                    
                    if response.status_code == 200:
                        with open(font_path, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"폰트 다운로드 완료: {font_path}")
                    else:
                        logger.error(f"폰트 다운로드 실패: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"폰트 다운로드 오류: {e}")


@dataclass
class OCRConfig:
    """OCR 설정 클래스"""
    
    # OCR 엔진
    engine: str = 'easyocr'  # easyocr, tesseract, paddleocr
    
    # 언어 설정
    languages: List[str] = field(default_factory=lambda: ['en'])
    
    # 성능 설정
    use_gpu: bool = True
    batch_size: int = 1
    workers: int = 4
    
    # 정확도 설정
    confidence_threshold: float = 0.5
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text: float = 0.4
    
    # 이미지 전처리
    preprocessing: Dict = field(default_factory=lambda: {
        'grayscale': True,
        'denoise': True,
        'deskew': True,
        'contrast_enhance': True,
        'sharpen': True,
        'binarize': False,
        'resize_factor': 1.0
    })
    
    # 후처리
    postprocessing: Dict = field(default_factory=lambda: {
        'spell_check': False,
        'remove_line_breaks': True,
        'merge_nearby_text': True,
        'filter_short_text': True,
        'min_text_length': 2
    })
    
    def get_easyocr_config(self) -> Dict:
        """EasyOCR용 설정 반환"""
        return {
            'gpu': self.use_gpu,
            'batch_size': self.batch_size,
            'workers': self.workers,
            'text_threshold': self.text_threshold,
            'link_threshold': self.link_threshold,
            'low_text': self.low_text
        }
    
    def get_tesseract_config(self) -> str:
        """Tesseract용 설정 문자열 반환"""
        config = []
        
        # PSM (Page Segmentation Mode)
        config.append('--psm 3')  # 자동 페이지 분할
        
        # OCR Engine Mode
        config.append('--oem 3')  # 기본 + LSTM
        
        return ' '.join(config)


class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        초기화
        
        Args:
            config_file: 설정 파일 경로
        """
        self.config_file = config_file or Path.home() / '.config' / 'pdf2zh_ocr' / 'config.json'
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 설정 로드 또는 생성
        if self.config_file.exists():
            self.translation_config = TranslationConfig.load(self.config_file)
        else:
            self.translation_config = TranslationConfig()
            self.save()
        
        # 폰트 설정
        self.font_config = FontConfig()
        
        # OCR 설정
        self.ocr_config = OCRConfig()
    
    def save(self):
        """설정 저장"""
        self.translation_config.save(self.config_file)
    
    def reset(self):
        """설정 초기화"""
        self.translation_config = TranslationConfig()
        self.font_config = FontConfig()
        self.ocr_config = OCRConfig()
        self.save()
    
    def update_translation_config(self, **kwargs):
        """번역 설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self.translation_config, key):
                setattr(self.translation_config, key, value)
        self.save()
    
    def update_ocr_settings(self, **kwargs):
        """OCR 설정 업데이트"""
        self.translation_config.ocr_settings.update(kwargs)
        self.save()
    
    def get_config_summary(self) -> str:
        """설정 요약 반환"""
        summary = []
        summary.append("=== 번역 설정 ===")
        summary.append(f"서비스: {self.translation_config.service}")
        summary.append(f"언어: {self.translation_config.lang_from} → {self.translation_config.lang_to}")
        summary.append(f"OCR: {'활성화' if self.translation_config.ocr_settings['enable_ocr'] else '비활성화'}")
        
        if self.translation_config.ocr_settings['enable_ocr']:
            summary.append("\n=== OCR 설정 ===")
            summary.append(f"엔진: {self.ocr_config.engine}")
            summary.append(f"언어: {', '.join(self.ocr_config.languages)}")
            summary.append(f"GPU: {'사용' if self.ocr_config.use_gpu else '미사용'}")
            summary.append(f"신뢰도 임계값: {self.ocr_config.confidence_threshold}")
        
        return '\n'.join(summary)


# 전역 설정 인스턴스
_config_manager = None


def get_config_manager() -> ConfigManager:
    """전역 설정 관리자 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def reset_config():
    """설정 초기화"""
    global _config_manager
    _config_manager = None

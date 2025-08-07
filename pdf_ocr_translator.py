"""
PDF OCR Translator - Stub version for Streamlit Cloud
Provides compatibility without OCR functionality
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class HybridPDFTranslator:
    """Stub translator for compatibility"""
    
    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        logger.info("HybridPDFTranslator initialized (OCR disabled)")
    
    def translate_pdf(self, input_path: str, output_dir: str,
                     pages: Optional[List[int]] = None,
                     progress_callback=None) -> Dict[str, str]:
        """Stub method - returns empty result"""
        return {
            'mono': None,
            'dual': None
        }
    
    def cleanup(self):
        """Cleanup (no-op)"""
        pass


# Alias for compatibility
PDFOCRTranslator = HybridPDFTranslator

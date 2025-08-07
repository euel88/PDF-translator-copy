"""
Image Processor Module - Stub version for Streamlit Cloud
OCR functionality disabled to save resources
"""

import logging
from typing import List, Dict, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Stub image processor (OCR disabled)"""
    
    def __init__(self, ocr_languages: List[str] = None, max_workers: int = 2):
        """Initialize (no-op)"""
        self.ocr_languages = ocr_languages or ['en']
        logger.info("ImageProcessor initialized (OCR disabled)")
    
    def detect_text_in_image(self, image: Image.Image) -> List[Dict]:
        """Stub method - returns empty list"""
        return []
    
    def extract_images_from_pdf_page(self, page, page_num: int) -> List[Dict]:
        """Stub method - returns empty list"""
        return []
    
    def cleanup(self):
        """Cleanup (no-op)"""
        pass


# Alias for compatibility
LightweightImageProcessor = ImageProcessor

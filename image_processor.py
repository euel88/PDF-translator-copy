"""
Image Processor Module - 메모리 최적화 버전
Streamlit Cloud용 경량 이미지 처리
"""

import logging
from typing import List, Dict, Tuple, Optional
import io
import gc
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


class LightweightImageProcessor:
    """메모리 효율적인 이미지 처리기"""
    
    def __init__(self, ocr_languages: List[str] = None, max_workers: int = 2):
        """
        초기화 (최소한의 리소스 사용)
        
        Args:
            ocr_languages: OCR 언어 목록
            max_workers: 최대 워커 수 (메모리 절약을 위해 제한)
        """
        self.ocr_languages = ocr_languages or ['en']
        self.ocr_reader = None
        self.max_image_size = (1920, 1080)  # 최대 이미지 크기 제한
        self.max_workers = min(max_workers, 2)  # 최대 2개로 제한
    
    def init_ocr_lazy(self):
        """OCR 엔진 지연 초기화 (필요할 때만)"""
        if self.ocr_reader is not None:
            return self.ocr_reader
        
        try:
            import easyocr
            
            # 언어 매핑
            lang_map = {
                'ko': 'ko',
                'en': 'en', 
                'zh': 'ch_sim',
                'ja': 'ja',
                'es': 'es',
                'fr': 'fr',
                'de': 'de'
            }
            
            ocr_langs = []
            for lang in self.ocr_languages:
                mapped = lang_map.get(lang, lang)
                if mapped not in ocr_langs:
                    ocr_langs.append(mapped)
            
            # CPU 모드로만 실행 (GPU 비활성화)
            logger.info(f"EasyOCR 초기화: {ocr_langs} (CPU mode)")
            self.ocr_reader = easyocr.Reader(ocr_langs, gpu=False)
            
            # 메모리 정리
            gc.collect()
            
            return self.ocr_reader
            
        except Exception as e:
            logger.error(f"OCR 초기화 실패: {e}")
            return None
    
    def resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        """이미지 크기 제한 (메모리 절약)"""
        if image.width > self.max_image_size[0] or image.height > self.max_image_size[1]:
            image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            logger.info(f"이미지 리사이즈: {image.size}")
        return image
    
    def extract_images_from_pdf_page(self, page, page_num: int) -> List[Dict]:
        """단일 페이지에서 이미지 추출 (메모리 효율적)"""
        images = []
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            if img_index >= 5:  # 페이지당 최대 5개 이미지만 처리
                logger.warning(f"페이지 {page_num}: 이미지 수 제한 초과")
                break
            
            try:
                import fitz
                
                # 이미지 추출
                xref = img_info[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # 알파 채널 제거
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                
                # PIL 이미지로 변환
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # 크기 제한
                image = self.resize_image_if_needed(image)
                
                # 메타데이터 저장
                images.append({
                    'page': page_num,
                    'index': img_index,
                    'xref': xref,
                    'image': image,
                    'size': image.size
                })
                
                # 즉시 메모리 해제
                pix = None
                del img_data
                
            except Exception as e:
                logger.error(f"이미지 추출 오류 (page {page_num}, img {img_index}): {e}")
                continue
        
        # 메모리 정리
        gc.collect()
        
        return images
    
    def detect_text_in_image(self, image: Image.Image, 
                           confidence_threshold: float = 0.5) -> List[Dict]:
        """이미지에서 텍스트 감지 (메모리 효율적)"""
        # OCR 초기화 (지연 로딩)
        reader = self.init_ocr_lazy()
        if not reader:
            return []
        
        try:
            # 이미지 전처리 (간단한 처리만)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 크기 제한
            image = self.resize_image_if_needed(image)
            
            # numpy 배열 변환
            img_array = np.array(image)
            
            # OCR 실행
            results = reader.readtext(img_array, batch_size=1)  # 배치 크기 제한
            
            texts = []
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    texts.append({
                        'bbox': bbox,
                        'text': text.strip(),
                        'confidence': confidence
                    })
            
            # 메모리 해제
            del img_array
            gc.collect()
            
            logger.info(f"텍스트 감지: {len(texts)}개")
            return texts
            
        except Exception as e:
            logger.error(f"OCR 실행 오류: {e}")
            return []
    
    def create_text_overlay_simple(self, image: Image.Image,
                                  texts: List[Dict],
                                  font_size: int = 16) -> Image.Image:
        """간단한 텍스트 오버레이 (메모리 효율적)"""
        try:
            # 이미지 복사
            img_copy = image.copy()
            
            # RGB로 변환
            if img_copy.mode != 'RGB':
                img_copy = img_copy.convert('RGB')
            
            draw = ImageDraw.Draw(img_copy)
            
            # 기본 폰트 사용
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # 텍스트 그리기 (간단한 방식)
            for text_info in texts:
                translated = text_info.get('translated', text_info['text'])
                bbox = text_info['bbox']
                
                # 바운딩 박스 좌표
                x = int(bbox[0][0])
                y = int(bbox[0][1])
                
                # 배경 박스
                draw.rectangle(
                    [(x-2, y-2), (x+200, y+font_size+4)],
                    fill=(255, 255, 255),
                    outline=(0, 0, 0)
                )
                
                # 텍스트
                draw.text(
                    (x, y),
                    translated[:50],  # 길이 제한
                    fill=(0, 0, 0),
                    font=font
                )
            
            return img_copy
            
        except Exception as e:
            logger.error(f"오버레이 생성 오류: {e}")
            return image
    
    def cleanup(self):
        """리소스 정리"""
        self.ocr_reader = None
        gc.collect()
        logger.info("ImageProcessor 리소스 정리 완료")


# 유틸리티 함수들
def process_image_batch(images: List[Image.Image], 
                        processor: LightweightImageProcessor,
                        batch_size: int = 2) -> List[Dict]:
    """이미지 배치 처리 (메모리 효율적)"""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        for image in batch:
            texts = processor.detect_text_in_image(image)
            results.append({
                'image': image,
                'texts': texts
            })
        
        # 배치 처리 후 메모리 정리
        gc.collect()
    
    return results


def estimate_memory_usage(image: Image.Image) -> float:
    """이미지 메모리 사용량 추정 (MB)"""
    # RGB 이미지 기준: width * height * 3 bytes
    bytes_used = image.width * image.height * 3
    return bytes_used / (1024 * 1024)


def check_memory_available(required_mb: float) -> bool:
    """사용 가능한 메모리 확인"""
    try:
        import psutil
        available = psutil.virtual_memory().available / (1024 * 1024)
        return available > required_mb
    except:
        # psutil 없으면 항상 True
        return True


# 하위 호환성을 위한 별칭
ImageProcessor = LightweightImageProcessor

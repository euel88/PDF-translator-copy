"""
Image Processor Module
이미지 추출, 텍스트 감지, 이미지 수정을 담당하는 모듈
"""

import logging
from typing import List, Dict, Tuple, Optional
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
import fitz  # PyMuPDF
import easyocr
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    """이미지 처리 전문 클래스"""
    
    def __init__(self, ocr_languages: List[str] = None):
        """
        초기화
        
        Args:
            ocr_languages: OCR에 사용할 언어 목록
        """
        self.ocr_languages = ocr_languages or ['en']
        self.ocr_reader = None
        self._init_ocr()
    
    def _init_ocr(self):
        """OCR 엔진 초기화"""
        try:
            # 언어 코드 매핑
            lang_map = {
                'ko': 'ko',
                'en': 'en',
                'zh': 'ch_sim',
                'zh-TW': 'ch_tra',
                'ja': 'ja',
                'de': 'de',
                'fr': 'fr',
                'es': 'es',
                'ru': 'ru',
                'ar': 'ar',
                'th': 'th',
                'vi': 'vi'
            }
            
            # OCR 언어 설정
            ocr_langs = []
            for lang in self.ocr_languages:
                mapped_lang = lang_map.get(lang, lang)
                if mapped_lang not in ocr_langs:
                    ocr_langs.append(mapped_lang)
            
            # GPU 사용 가능 여부 확인
            gpu = self._check_gpu()
            
            logger.info(f"EasyOCR 초기화: languages={ocr_langs}, gpu={gpu}")
            self.ocr_reader = easyocr.Reader(ocr_langs, gpu=gpu)
            
        except Exception as e:
            logger.error(f"OCR 초기화 실패: {e}")
            # 폴백: 영어만 사용
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except:
                logger.error("OCR 완전 실패")
                self.ocr_reader = None
    
    def _check_gpu(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        PDF에서 모든 이미지 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            이미지 정보 리스트
        """
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num, page in enumerate(doc):
            page_images = self._extract_images_from_page(page, page_num)
            images.extend(page_images)
        
        doc.close()
        return images
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """페이지에서 이미지 추출"""
        images = []
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            try:
                # 이미지 추출
                xref = img_info[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # 이미지 메타데이터
                img_rect = page.get_image_bbox(img_info)
                
                # PIL 이미지로 변환
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)  # 알파 채널 제거
                img_data = pix.pil_tobytes(format="PNG")
                image = Image.open(io.BytesIO(img_data))
                
                # 이미지 정보 저장
                images.append({
                    'page': page_num,
                    'index': img_index,
                    'xref': xref,
                    'bbox': img_rect,
                    'image': image,
                    'width': image.width,
                    'height': image.height
                })
                
                pix = None  # 메모리 해제
                
            except Exception as e:
                logger.error(f"이미지 추출 오류 (page {page_num}, img {img_index}): {e}")
                continue
        
        return images
    
    def detect_text_in_image(self, image: Image.Image, 
                           enhance: bool = True,
                           confidence_threshold: float = 0.5) -> List[Dict]:
        """
        이미지에서 텍스트 감지
        
        Args:
            image: PIL 이미지
            enhance: 이미지 향상 여부
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            감지된 텍스트 정보 리스트
        """
        if not self.ocr_reader:
            logger.warning("OCR reader not initialized")
            return []
        
        # 이미지 전처리
        if enhance:
            image = self._enhance_image_for_ocr(image)
        
        # numpy 배열로 변환
        img_array = np.array(image)
        
        # OCR 실행
        try:
            results = self.ocr_reader.readtext(img_array)
            
            texts = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    # 바운딩 박스를 표준 형식으로 변환
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    texts.append({
                        'bbox': bbox,
                        'text': text.strip(),
                        'confidence': confidence,
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords),
                        'center_x': sum(x_coords) / len(x_coords),
                        'center_y': sum(y_coords) / len(y_coords)
                    })
            
            logger.info(f"감지된 텍스트: {len(texts)}개")
            return texts
            
        except Exception as e:
            logger.error(f"OCR 실행 오류: {e}")
            return []
    
    def _enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """OCR을 위한 이미지 향상"""
        try:
            # 그레이스케일 변환
            if image.mode != 'L':
                image = image.convert('L')
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # 선명도 향상
            image = image.filter(ImageFilter.SHARPEN)
            
            # 이진화 (선택적)
            # threshold = 128
            # image = image.point(lambda p: p > threshold and 255)
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 향상 오류: {e}")
            return image
    
    def extract_images_with_text(self, pdf_path: str, 
                                confidence_threshold: float = 0.5) -> List[Dict]:
        """
        PDF에서 텍스트가 포함된 이미지만 추출
        
        Args:
            pdf_path: PDF 파일 경로
            confidence_threshold: OCR 신뢰도 임계값
            
        Returns:
            텍스트가 포함된 이미지 정보 리스트
        """
        # 모든 이미지 추출
        all_images = self.extract_images_from_pdf(pdf_path)
        
        # 텍스트가 있는 이미지만 필터링
        images_with_text = []
        
        for img_info in all_images:
            # 텍스트 감지
            texts = self.detect_text_in_image(
                img_info['image'],
                confidence_threshold=confidence_threshold
            )
            
            if texts:
                img_info['texts'] = texts
                images_with_text.append(img_info)
                logger.info(f"페이지 {img_info['page']}, 이미지 {img_info['index']}: {len(texts)}개 텍스트 감지")
        
        return images_with_text
    
    def create_text_overlay(self, image: Image.Image, 
                          texts: List[Dict],
                          font_path: Optional[str] = None,
                          font_size: int = 20,
                          text_color: Tuple[int, int, int] = (0, 0, 0),
                          bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
                          preserve_layout: bool = True) -> Image.Image:
        """
        이미지에 텍스트 오버레이 생성
        
        Args:
            image: 원본 이미지
            texts: 텍스트 정보 리스트 (번역된 텍스트 포함)
            font_path: 폰트 파일 경로
            font_size: 폰트 크기
            text_color: 텍스트 색상
            bg_color: 배경 색상 (RGBA)
            preserve_layout: 원본 레이아웃 유지 여부
            
        Returns:
            텍스트가 오버레이된 이미지
        """
        # 이미지 복사
        img_copy = image.copy()
        
        # RGBA로 변환
        if img_copy.mode != 'RGBA':
            img_copy = img_copy.convert('RGBA')
        
        # 오버레이 레이어 생성
        overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # 폰트 로드
        try:
            if font_path and Path(font_path).exists():
                font = ImageFont.truetype(font_path, size=font_size)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 각 텍스트 처리
        for text_info in texts:
            if preserve_layout:
                # 원본 위치에 텍스트 배치
                self._draw_text_at_position(
                    draw_overlay, 
                    text_info, 
                    font, 
                    text_color, 
                    bg_color
                )
            else:
                # 새로운 레이아웃으로 텍스트 배치
                self._draw_text_new_layout(
                    draw_overlay,
                    text_info,
                    font,
                    text_color,
                    bg_color
                )
        
        # 오버레이 합성
        result = Image.alpha_composite(img_copy, overlay)
        
        # RGB로 변환
        if result.mode != 'RGB':
            result = result.convert('RGB')
        
        return result
    
    def _draw_text_at_position(self, draw, text_info: Dict, font, 
                              text_color, bg_color):
        """원본 위치에 텍스트 그리기"""
        # 번역된 텍스트 가져오기
        translated_text = text_info.get('translated', text_info['text'])
        
        # 바운딩 박스
        x_min = text_info['x_min']
        x_max = text_info['x_max']
        y_min = text_info['y_min']
        y_max = text_info['y_max']

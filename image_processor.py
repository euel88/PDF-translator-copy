"""
Image Processor Module - 수정된 버전
이미지 추출, 텍스트 감지, 이미지 수정을 담당하는 모듈
PyMuPDF API 호환성 문제 해결
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
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ImageProcessor:
    """이미지 처리 전문 클래스 - 수정된 버전"""
    
    def __init__(self, ocr_languages: List[str] = None, max_workers: int = 4):
        """
        초기화
        
        Args:
            ocr_languages: OCR에 사용할 언어 목록
            max_workers: 병렬 처리를 위한 최대 워커 수
        """
        self.ocr_languages = ocr_languages or ['en']
        self.ocr_reader = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._init_ocr()
    
    def _init_ocr(self):
        """OCR 엔진 초기화 - 개선된 에러 처리"""
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
            
            # 기본 언어 추가 (폴백용)
            if 'en' not in ocr_langs:
                ocr_langs.append('en')
            
            # GPU 사용 가능 여부 확인
            gpu = self._check_gpu()
            
            logger.info(f"EasyOCR 초기화: languages={ocr_langs}, gpu={gpu}")
            self.ocr_reader = easyocr.Reader(ocr_langs, gpu=gpu)
            
        except Exception as e:
            logger.error(f"OCR 초기화 실패: {e}")
            # 최소한의 폴백 설정
            try:
                logger.info("폴백: CPU에서 영어 OCR만 사용")
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except:
                logger.error("OCR 완전 실패 - OCR 기능 비활성화")
                self.ocr_reader = None
    
    def _check_gpu(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    async def extract_images_from_pdf_async(self, pdf_path: str) -> List[Dict]:
        """PDF에서 모든 이미지 비동기 추출"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.extract_images_from_pdf,
            pdf_path
        )
    
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
                    
                img_data = pix.tobytes("png")  # 수정: pil_tobytes -> tobytes
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
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 향상 오류: {e}")
            return image
    
    def replace_image_in_page(self, page, xref: int, new_image: Image.Image) -> bool:
        """
        페이지의 이미지 교체 - PyMuPDF API 호환성 수정
        
        Args:
            page: PDF 페이지 객체
            xref: 교체할 이미지의 xref
            new_image: 새 이미지
            
        Returns:
            성공 여부
        """
        try:
            # 이미지를 PNG 바이트로 변환
            img_buffer = io.BytesIO()
            new_image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            # 방법 1: 스트림 직접 업데이트 시도
            try:
                # 기존 이미지의 스트림을 새 이미지로 교체
                page.parent.update_stream(xref, img_data)
                
                # 이미지 딕셔너리 업데이트
                img_dict = page.parent.xref_object(xref)
                if img_dict:
                    img_dict["Width"] = new_image.width
                    img_dict["Height"] = new_image.height
                    page.parent.update_object(xref, str(img_dict))
                
                logger.info(f"이미지 교체 성공: xref={xref}")
                return True
                
            except AttributeError:
                # 방법 2: 새 이미지 추가 후 참조 변경
                logger.info("스트림 업데이트 실패, 대체 방법 시도")
                
                # 새 이미지 삽입
                img_rect = page.get_image_bbox(xref)
                page.insert_image(
                    img_rect,
                    stream=img_data,
                    keep_proportion=True
                )
                
                # 기존 이미지 숨기기 (흰색 사각형으로 덮기)
                page.draw_rect(img_rect, color=(1, 1, 1), fill=(1, 1, 1))
                
                return True
                
        except Exception as e:
            logger.error(f"이미지 교체 실패: {e}")
            # 방법 3: 폴백 - 이미지 위에 텍스트 오버레이
            return self._add_text_overlay_fallback(page, xref, new_image)
    
    def _add_text_overlay_fallback(self, page, xref: int, new_image: Image.Image) -> bool:
        """
        폴백: 이미지를 교체하는 대신 텍스트를 오버레이로 추가
        
        Args:
            page: PDF 페이지
            xref: 이미지 xref  
            new_image: 텍스트가 포함된 새 이미지
            
        Returns:
            성공 여부
        """
        try:
            # 이미지 위치 가져오기
            img_info = None
            for img in page.get_images():
                if img[0] == xref:
                    img_info = img
                    break
            
            if not img_info:
                return False
            
            img_rect = page.get_image_bbox(img_info)
            
            # 반투명 흰색 배경 추가
            page.draw_rect(
                img_rect,
                color=(1, 1, 1),
                fill=(1, 1, 1),
                opacity=0.7
            )
            
            # 새 이미지를 오버레이로 추가
            img_buffer = io.BytesIO()
            new_image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            page.insert_image(
                img_rect,
                stream=img_data,
                overlay=True
            )
            
            logger.info(f"텍스트 오버레이 추가 완료: xref={xref}")
            return True
            
        except Exception as e:
            logger.error(f"텍스트 오버레이 추가 실패: {e}")
            return False
    
    def create_text_overlay(self, image: Image.Image, 
                          texts: List[Dict],
                          font_path: Optional[str] = None,
                          font_size: int = 20,
                          text_color: Tuple[int, int, int] = (0, 0, 0),
                          bg_color: Tuple[int, int, int, int] = (255, 255, 255, 200),
                          preserve_layout: bool = True) -> Image.Image:
        """
        이미지에 번역된 텍스트 오버레이 생성
        
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
                self._draw_text_at_position(
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
        
        # 배경 그리기
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            fill=bg_color
        )
        
        # 텍스트 그리기
        draw.text(
            (x_min, y_min),
            translated_text,
            fill=text_color,
            font=font
        )
    
    def cleanup(self):
        """리소스 정리"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.ocr_reader = None
        logger.info("ImageProcessor 리소스 정리 완료")

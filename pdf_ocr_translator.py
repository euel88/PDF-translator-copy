"""
PDF OCR Translator
이미지 내 텍스트를 감지하고 번역하는 핵심 모듈
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import easyocr
import openai
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PDFOCRTranslator:
    """PDF 문서의 텍스트와 이미지를 모두 번역하는 클래스"""
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config: TranslationConfig 인스턴스
        """
        self.config = config
        self.ocr_reader = None
        self.translator = None
        
        # OCR 초기화
        if config.ocr_settings.get('enable_ocr', False):
            self._init_ocr()
        
        # 번역기 초기화
        self._init_translator()
        
        # 폰트 설정
        self.font_path = self._get_font_path()
        
    def _init_ocr(self):
        """OCR 엔진 초기화"""
        try:
            # EasyOCR 초기화
            languages = self.config.ocr_settings.get('ocr_languages', ['en'])
            # 한국어 매핑
            lang_map = {
                'ko': 'ko',
                'en': 'en',
                'zh': 'ch_sim',
                'zh-TW': 'ch_tra',
                'ja': 'ja',
                'de': 'de',
                'fr': 'fr',
                'es': 'es',
                'ru': 'ru'
            }
            
            ocr_langs = []
            for lang in languages:
                mapped_lang = lang_map.get(lang, lang)
                if mapped_lang not in ocr_langs:
                    ocr_langs.append(mapped_lang)
            
            # GPU 사용 가능 여부 확인
            gpu = self._check_gpu_available()
            
            logger.info(f"EasyOCR 초기화: languages={ocr_langs}, gpu={gpu}")
            self.ocr_reader = easyocr.Reader(ocr_langs, gpu=gpu)
            
        except Exception as e:
            logger.error(f"OCR 초기화 실패: {e}")
            raise
    
    def _check_gpu_available(self):
        """GPU 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _init_translator(self):
        """번역기 초기화"""
        if self.config.service == 'openai':
            openai.api_key = self.config.api_key
            self.translator = self._translate_openai
        elif self.config.service == 'google':
            self.translator = self._translate_google
        else:
            # 기본값으로 Google 사용
            self.translator = self._translate_google
    
    def _get_font_path(self):
        """폰트 경로 가져오기"""
        font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
        
        # 언어별 폰트 선택
        if self.config.lang_to in ['ko', 'kr']:
            font_file = "NanumGothic.ttf"
        elif self.config.lang_to in ['zh', 'zh-CN', 'zh-TW']:
            font_file = "SourceHanSerifCN-Regular.ttf"
        elif self.config.lang_to == 'ja':
            font_file = "SourceHanSerifJP-Regular.ttf"
        else:
            font_file = "GoNotoKurrent-Regular.ttf"
        
        font_path = font_dir / font_file
        
        # 폰트가 없으면 기본 폰트 사용
        if not font_path.exists():
            # 시스템 기본 폰트 찾기
            if os.name == 'nt':  # Windows
                return "C:/Windows/Fonts/arial.ttf"
            elif os.name == 'posix':  # Linux/Mac
                return "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        
        return str(font_path)
    
    def translate_pdf(self, input_path: str, output_dir: str, 
                     pages: Optional[List[int]] = None,
                     progress_callback=None) -> Dict[str, str]:
        """
        PDF 번역 (텍스트 + 이미지)
        
        Args:
            input_path: 입력 PDF 경로
            output_dir: 출력 디렉토리
            pages: 번역할 페이지 목록
            progress_callback: 진행률 콜백 함수
            
        Returns:
            {'mono': 번역본_경로, 'dual': 대조본_경로}
        """
        logger.info(f"PDF 번역 시작: {input_path}")
        
        # 출력 파일 경로
        base_name = Path(input_path).stem
        mono_path = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_path = Path(output_dir) / f"{base_name}-dual.pdf"
        
        # PDF 열기
        doc = fitz.open(input_path)
        doc_mono = fitz.open()  # 번역본
        doc_dual = fitz.open()  # 대조본
        
        # 페이지 범위 설정
        if pages is None:
            pages = list(range(len(doc)))
        
        total_pages = len(pages)
        
        # 각 페이지 처리
        for idx, page_num in enumerate(pages):
            if progress_callback:
                progress_callback(idx / total_pages)
            
            logger.info(f"페이지 {page_num + 1}/{len(doc)} 처리 중...")
            
            page = doc[page_num]
            
            # 1. 텍스트 레이어 번역 (pdf2zh 방식)
            translated_page = self._translate_text_layer(page)
            
            # 2. 이미지 번역 (OCR)
            if self.config.ocr_settings.get('enable_ocr', False):
                translated_page = self._translate_images_in_page(translated_page)
            
            # 번역된 페이지를 문서에 추가
            doc_mono.insert_pdf(
                fitz.open(stream=translated_page.write(deflate=True, garbage=3), filetype="pdf"),
                from_page=0,
                to_page=0
            )
            
            # 대조본 생성 (원본 + 번역본)
            if idx % 2 == 0:
                doc_dual.insert_pdf(doc, from_page=page_num, to_page=page_num)
                doc_dual.insert_pdf(
                    fitz.open(stream=translated_page.write(deflate=True, garbage=3), filetype="pdf"),
                    from_page=0,
                    to_page=0
                )
        
        # PDF 저장
        doc_mono.save(str(mono_path), deflate=True, garbage=3)
        doc_dual.save(str(dual_path), deflate=True, garbage=3)
        
        # 정리
        doc.close()
        doc_mono.close()
        doc_dual.close()
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info(f"번역 완료: mono={mono_path}, dual={dual_path}")
        
        return {
            'mono': str(mono_path),
            'dual': str(dual_path)
        }
    
    def _translate_text_layer(self, page):
        """텍스트 레이어 번역 (기존 pdf2zh 방식)"""
        # 텍스트 추출
        text = page.get_text()
        
        if not text.strip():
            return page
        
        # 텍스트 번역
        blocks = page.get_text("blocks")
        
        for block in blocks:
            if block[6] == 0:  # 텍스트 블록
                original_text = block[4]
                if original_text.strip():
                    # 번역
                    translated_text = self.translator(original_text)
                    
                    # 텍스트 교체
                    rect = fitz.Rect(block[:4])
                    page.add_redact_annot(rect)
                    page.apply_redactions()
                    
                    # 번역된 텍스트 삽입
                    try:
                        fontsize = self._estimate_fontsize(rect, translated_text)
                        page.insert_textbox(
                            rect,
                            translated_text,
                            fontsize=fontsize,
                            align=fitz.TEXT_ALIGN_LEFT
                        )
                    except:
                        # 폴백: 기본 크기로 삽입
                        page.insert_textbox(
                            rect,
                            translated_text,
                            fontsize=10,
                            align=fitz.TEXT_ALIGN_LEFT
                        )
        
        return page
    
    def _translate_images_in_page(self, page):
        """페이지 내 이미지 번역"""
        # 이미지 추출
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            try:
                # 이미지 추출
                xref = img_info[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                # PIL 이미지로 변환
                img_data = pix.pil_tobytes(format="PNG")
                image = Image.open(io.BytesIO(img_data))
                
                # 이미지에서 텍스트 감지
                texts = self._detect_text_in_image(image)
                
                if texts:
                    # 텍스트 번역 및 이미지 수정
                    modified_image = self._translate_and_overlay_text(image, texts)
                    
                    # 수정된 이미지로 교체
                    if self.config.ocr_settings.get('replace_images', True):
                        self._replace_image_in_page(page, xref, modified_image)
                
            except Exception as e:
                logger.error(f"이미지 처리 오류: {e}")
                continue
        
        return page
    
    def _detect_text_in_image(self, image: Image.Image) -> List[Dict]:
        """이미지에서 텍스트 감지"""
        if not self.ocr_reader:
            return []
        
        # PIL 이미지를 numpy 배열로 변환
        img_array = np.array(image)
        
        # OCR 실행
        try:
            results = self.ocr_reader.readtext(img_array)
            
            texts = []
            for (bbox, text, prob) in results:
                if prob > 0.5:  # 신뢰도 임계값
                    texts.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': prob
                    })
            
            return texts
            
        except Exception as e:
            logger.error(f"OCR 오류: {e}")
            return []
    
    def _translate_and_overlay_text(self, image: Image.Image, texts: List[Dict]) -> Image.Image:
        """텍스트 번역 및 이미지에 오버레이"""
        # 이미지 복사
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # 폰트 로드
        try:
            font = ImageFont.truetype(self.font_path, size=20)
        except:
            font = ImageFont.load_default()
        
        for text_info in texts:
            original_text = text_info['text']
            bbox = text_info['bbox']
            
            # 텍스트 번역
            translated_text = self.translator(original_text)
            
            if self.config.ocr_settings.get('overlay_text', True):
                # 배경 박스 그리기 (반투명)
                overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # 바운딩 박스 좌표 추출
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 배경 그리기
                overlay_draw.rectangle(
                    [(x_min, y_min), (x_max, y_max)],
                    fill=(255, 255, 255, 200)  # 반투명 흰색
                )
                
                # 오버레이 합성
                img_copy = Image.alpha_composite(
                    img_copy.convert('RGBA'),
                    overlay
                ).convert('RGB')
                
                # 번역된 텍스트 그리기
                draw = ImageDraw.Draw(img_copy)
                
                # 텍스트 래핑 처리
                wrapped_text = self._wrap_text(translated_text, font, x_max - x_min)
                
                # 텍스트 그리기
                y_offset = y_min
                for line in wrapped_text:
                    draw.text(
                        (x_min, y_offset),
                        line,
                        fill=(0, 0, 0),  # 검은색
                        font=font
                    )
                    y_offset += font.getsize(line)[1]
        
        return img_copy
    
    def _wrap_text(self, text: str, font: ImageFont, max_width: int) -> List[str]:
        """텍스트를 최대 너비에 맞게 래핑"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            width = font.getsize(test_line)[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _replace_image_in_page(self, page, xref: int, new_image: Image.Image):
        """페이지의 이미지 교체"""
        try:
            # 이미지를 바이트로 변환
            img_buffer = io.BytesIO()
            new_image.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()
            
            # 새 이미지로 교체
            page.parent._deleteObject(xref)
            new_xref = page.parent._add_image(img_data)
            
            # 이미지 참조 업데이트
            # 이 부분은 PyMuPDF 버전에 따라 다를 수 있음
            
        except Exception as e:
            logger.error(f"이미지 교체 오류: {e}")
    
    def _translate_openai(self, text: str) -> str:
        """OpenAI API를 사용한 번역"""
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Translate the following text from {self.config.lang_from} to {self.config.lang_to}. Only return the translation."},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=len(text) * 2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI 번역 오류: {e}")
            return text
    
    def _translate_google(self, text: str) -> str:
        """Google Translate를 사용한 번역"""
        try:
            # Google Translate 무료 API (비공식)
            url = "https://translate.google.com/m"
            params = {
                'sl': self.config.lang_from,
                'tl': self.config.lang_to,
                'q': text
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                # HTML 파싱 (간단한 방법)
                import re
                result = re.findall(r'class="(?:t0|result-container)">(.*?)<', response.text)
                if result:
                    return html.unescape(result[0])
            
            return text
            
        except Exception as e:
            logger.error(f"Google 번역 오류: {e}")
            return text
    
    def _estimate_fontsize(self, rect: fitz.Rect, text: str) -> float:
        """텍스트 박스에 맞는 폰트 크기 추정"""
        # 간단한 추정: 박스 크기와 텍스트 길이 기반
        width = rect.width
        height = rect.height
        text_len = len(text)
        
        # 기본값
        fontsize = 10
        
        # 박스 크기에 따라 조정
        if text_len > 0:
            # 가로 기준
            chars_per_line = width / 6  # 대략적인 문자 너비
            lines = text_len / chars_per_line
            
            if lines > 0:
                fontsize = min(height / lines, 20)
                fontsize = max(fontsize, 6)  # 최소 크기
        
        return fontsize


# 보조 함수들
import io
import html


def extract_images_from_pdf(pdf_path: str) -> List[Tuple[int, Image.Image]]:
    """PDF에서 모든 이미지 추출"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                pix = fitz.Pixmap(doc, xref)
                
                # PIL 이미지로 변환
                img_data = pix.pil_tobytes(format="PNG")
                image = Image.open(io.BytesIO(img_data))
                
                images.append((page_num, image))
                
            except Exception as e:
                logger.error(f"이미지 추출 오류: {e}")
                continue
    
    doc.close()
    return images


def merge_translated_pdfs(original_pdf: str, text_translated_pdf: str, 
                         image_translated_pdf: str, output_pdf: str):
    """텍스트 번역본과 이미지 번역본 병합"""
    # 구현 필요: 두 PDF를 지능적으로 병합
    pass

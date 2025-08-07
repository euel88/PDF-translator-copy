"""
PDF OCR Translator
이미지 내 텍스트를 감지하고 번역하는 핵심 모듈
pdf2zh와 통합되어 작동
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import html

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import easyocr
import openai
import requests
from tqdm import tqdm

# pdf2zh 모듈 import
try:
    from pdf2zh import translate as pdf2zh_translate
    from pdf2zh.doclayout import ModelInstance, OnnxModel
    PDF2ZH_AVAILABLE = True
except ImportError:
    PDF2ZH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("pdf2zh 모듈을 사용할 수 없습니다. 기본 번역 기능만 사용됩니다.")

from image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class HybridPDFTranslator:
    """pdf2zh와 OCR을 통합하는 하이브리드 번역기"""
    
    def __init__(self, config):
        """
        초기화
        
        Args:
            config: TranslationConfig 인스턴스
        """
        self.config = config
        self.image_processor = None
        self.translator = None
        self.executor = ThreadPoolExecutor(max_workers=config.thread_count or 4)
        
        # OCR 초기화
        if config.ocr_settings.get('enable_ocr', False):
            self.image_processor = ImageProcessor(
                ocr_languages=[config.lang_from]
            )
        
        # 번역기 초기화
        self._init_translator()
        
        # 폰트 설정
        self.font_path = self._get_font_path()
        
        # pdf2zh 모델 초기화
        if PDF2ZH_AVAILABLE and ModelInstance.value is None:
            try:
                ModelInstance.value = OnnxModel.from_pretrained()
            except Exception as e:
                logger.error(f"ONNX 모델 로드 실패: {e}")
    
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
    
    async def translate_pdf_async(self, input_path: str, output_dir: str, 
                                 pages: Optional[List[int]] = None,
                                 progress_callback=None) -> Dict[str, str]:
        """
        PDF 번역 (비동기) - pdf2zh와 OCR 통합
        
        Args:
            input_path: 입력 PDF 경로
            output_dir: 출력 디렉토리
            pages: 번역할 페이지 목록
            progress_callback: 진행률 콜백 함수
            
        Returns:
            {'mono': 번역본_경로, 'dual': 대조본_경로}
        """
        logger.info(f"하이브리드 PDF 번역 시작: {input_path}")
        
        # 출력 파일 경로
        base_name = Path(input_path).stem
        mono_path = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_path = Path(output_dir) / f"{base_name}-dual.pdf"
        
        # 1단계: pdf2zh로 텍스트 레이어 번역
        if PDF2ZH_AVAILABLE:
            logger.info("pdf2zh로 텍스트 레이어 번역 중...")
            mono_file, dual_file = await self._translate_with_pdf2zh_async(
                input_path, output_dir, pages, progress_callback
            )
        else:
            logger.info("pdf2zh 없이 기본 번역 진행...")
            mono_file, dual_file = await self._translate_basic_async(
                input_path, output_dir, pages, progress_callback
            )
        
        # 2단계: OCR로 이미지 텍스트 번역 (활성화된 경우)
        if self.config.ocr_settings.get('enable_ocr', False) and self.image_processor:
            logger.info("OCR로 이미지 텍스트 번역 중...")
            mono_file = await self._enhance_with_ocr_async(
                mono_file, pages, progress_callback
            )
        
        return {
            'mono': mono_file,
            'dual': dual_file
        }
    
    async def _translate_with_pdf2zh_async(self, input_path: str, output_dir: str,
                                          pages: Optional[List[int]], 
                                          progress_callback) -> Tuple[str, str]:
        """pdf2zh를 사용한 비동기 번역"""
        loop = asyncio.get_event_loop()
        
        # pdf2zh는 동기 함수이므로 executor에서 실행
        def run_pdf2zh():
            try:
                # 환경 변수 설정
                envs = {}
                if self.config.api_key:
                    if self.config.service == 'openai':
                        envs['OPENAI_API_KEY'] = self.config.api_key
                        envs['OPENAI_MODEL'] = self.config.model or 'gpt-4o-mini'
                
                # pdf2zh 실행
                result = pdf2zh_translate(
                    files=[input_path],
                    output=output_dir,
                    pages=pages,
                    lang_in=self.config.lang_from,
                    lang_out=self.config.lang_to,
                    service=self.config.service,
                    thread=self.config.thread_count,
                    model=ModelInstance.value,
                    envs=envs,
                    skip_subset_fonts=self.config.skip_subset_fonts,
                    ignore_cache=not self.config.use_cache
                )
                
                # 결과 파일 경로
                base_name = Path(input_path).stem
                mono_file = str(Path(output_dir) / f"{base_name}-mono.pdf")
                dual_file = str(Path(output_dir) / f"{base_name}-dual.pdf")
                
                return mono_file, dual_file
                
            except Exception as e:
                logger.error(f"pdf2zh 번역 오류: {e}")
                raise
        
        # 비동기 실행
        result = await loop.run_in_executor(self.executor, run_pdf2zh)
        
        if progress_callback:
            progress_callback(0.5)
        
        return result
    
    async def _translate_basic_async(self, input_path: str, output_dir: str,
                                    pages: Optional[List[int]], 
                                    progress_callback) -> Tuple[str, str]:
        """기본 번역 (pdf2zh 없이)"""
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
        
        # 페이지별 비동기 번역
        tasks = []
        for page_num in pages:
            task = self._translate_page_async(doc[page_num])
            tasks.append(task)
        
        # 모든 페이지 번역 완료 대기
        translated_pages = await asyncio.gather(*tasks)
        
        # 번역된 페이지를 문서에 추가
        for idx, (page_num, translated_page_data) in enumerate(zip(pages, translated_pages)):
            if progress_callback:
                progress_callback(idx / len(pages) * 0.5)
            
            # 번역된 페이지 추가
            temp_doc = fitz.open(stream=translated_page_data, filetype="pdf")
            doc_mono.insert_pdf(temp_doc, from_page=0, to_page=0)
            temp_doc.close()
            
            # 대조본 생성
            if idx % 2 == 0:
                doc_dual.insert_pdf(doc, from_page=page_num, to_page=page_num)
                temp_doc = fitz.open(stream=translated_page_data, filetype="pdf")
                doc_dual.insert_pdf(temp_doc, from_page=0, to_page=0)
                temp_doc.close()
        
        # PDF 저장
        doc_mono.save(str(mono_path), deflate=True, garbage=3)
        doc_dual.save(str(dual_path), deflate=True, garbage=3)
        
        # 정리
        doc.close()
        doc_mono.close()
        doc_dual.close()
        
        return str(mono_path), str(dual_path)
    
    async def _translate_page_async(self, page) -> bytes:
        """단일 페이지 비동기 번역"""
        loop = asyncio.get_event_loop()
        
        def translate_page_sync():
            # 페이지 복사
            doc_temp = fitz.open()
            doc_temp.insert_pdf(
                fitz.open(),
                from_page=-1,
                to_page=-1
            )
            new_page = doc_temp.new_page(
                width=page.rect.width,
                height=page.rect.height
            )
            
            # 텍스트 추출 및 번역
            blocks = page.get_text("blocks")
            
            for block in blocks:
                if block[6] == 0:  # 텍스트 블록
                    original_text = block[4]
                    if original_text.strip():
                        # 번역
                        translated_text = self.translator(original_text)
                        
                        # 텍스트 삽입
                        rect = fitz.Rect(block[:4])
                        new_page.insert_textbox(
                            rect,
                            translated_text,
                            fontsize=10,
                            align=fitz.TEXT_ALIGN_LEFT
                        )
            
            # 페이지를 바이트로 변환
            return doc_temp.write(deflate=True, garbage=3)
        
        # 비동기 실행
        return await loop.run_in_executor(self.executor, translate_page_sync)
    
    async def _enhance_with_ocr_async(self, pdf_path: str, 
                                     pages: Optional[List[int]],
                                     progress_callback) -> str:
        """OCR로 이미지 텍스트 번역 (비동기)"""
        if not self.image_processor:
            return pdf_path
        
        doc = fitz.open(pdf_path)
        
        # 페이지 범위 설정
        if pages is None:
            pages = list(range(len(doc)))
        
        # 각 페이지의 이미지 처리
        tasks = []
        for page_num in pages:
            task = self._process_page_images_async(doc, page_num)
            tasks.append(task)
        
        # 모든 이미지 처리 완료 대기
        results = await asyncio.gather(*tasks)
        
        # 처리 결과 적용
        for page_num, modified in results:
            if modified and progress_callback:
                progress_callback(0.5 + (page_num / len(pages)) * 0.5)
        
        # 수정된 PDF 저장
        output_path = pdf_path.replace('.pdf', '_ocr.pdf')
        doc.save(output_path, deflate=True, garbage=3)
        doc.close()
        
        return output_path
    
    async def _process_page_images_async(self, doc, page_num: int) -> Tuple[int, bool]:
        """페이지의 이미지 비동기 처리"""
        loop = asyncio.get_event_loop()
        
        def process_images_sync():
            page = doc[page_num]
            image_list = page.get_images()
            modified = False
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # 이미지 추출
                    xref = img_info[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # PIL 이미지로 변환
                    img_data = pix.pil_tobytes(format="PNG")
                    image = Image.open(io.BytesIO(img_data))
                    
                    # 텍스트 감지
                    texts = self.image_processor.detect_text_in_image(image)
                    
                    if texts:
                        # 텍스트 번역
                        for text_info in texts:
                            original = text_info['text']
                            translated = self.translator(original)
                            text_info['translated'] = translated
                        
                        # 이미지에 번역된 텍스트 오버레이
                        modified_image = self.image_processor.create_text_overlay(
                            image, texts, self.font_path
                        )
                        
                        # 이미지 교체
                        if self.config.ocr_settings.get('replace_images', True):
                            success = self.image_processor.replace_image_in_pdf(
                                doc, page_num, xref, modified_image
                            )
                            if success:
                                modified = True
                    
                    pix = None  # 메모리 해제
                    
                except Exception as e:
                    logger.error(f"이미지 처리 오류 (page {page_num}, img {img_index}): {e}")
            
            return page_num, modified
        
        # 비동기 실행
        return await loop.run_in_executor(self.executor, process_images_sync)
    
    def translate_pdf(self, input_path: str, output_dir: str, 
                     pages: Optional[List[int]] = None,
                     progress_callback=None) -> Dict[str, str]:
        """
        동기 인터페이스 (하위 호환성)
        """
        return asyncio.run(
            self.translate_pdf_async(input_path, output_dir, pages, progress_callback)
        )
    
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
    
    def __del__(self):
        """소멸자 - 리소스 정리"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# 하위 호환성을 위한 별칭
PDFOCRTranslator = HybridPDFTranslator

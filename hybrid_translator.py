"""
Hybrid PDF Translator
pdf2zh와 OCR을 통합하여 완벽한 PDF 번역을 제공하는 모듈
비동기 처리 및 메모리 최적화 포함
"""

import os
import gc
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

# pdf2zh 관련 imports
try:
    from pdf2zh import translate as pdf2zh_translate
    from pdf2zh.doclayout import ModelInstance, OnnxModel
    PDF2ZH_AVAILABLE = True
except ImportError:
    PDF2ZH_AVAILABLE = False
    logging.warning("pdf2zh를 사용할 수 없습니다. OCR 모드만 사용 가능합니다.")

# 커스텀 모듈
from image_processor import ImageProcessor
from config import TranslationConfig, ConfigManager

logger = logging.getLogger(__name__)


class HybridTranslator:
    """pdf2zh와 OCR을 통합하는 하이브리드 번역기"""
    
    def __init__(self, config: TranslationConfig):
        """
        초기화
        
        Args:
            config: TranslationConfig 인스턴스
        """
        self.config = config
        self.pdf2zh_enabled = PDF2ZH_AVAILABLE
        self.ocr_enabled = config.ocr_settings.get('enable_ocr', False)
        
        # OCR 프로세서 초기화
        if self.ocr_enabled:
            self.image_processor = ImageProcessor(
                ocr_languages=[config.lang_from],
                max_workers=config.thread_count
            )
        else:
            self.image_processor = None
        
        # 비동기 처리를 위한 executor
        self.thread_executor = ThreadPoolExecutor(max_workers=config.thread_count)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # 번역기 초기화
        self._init_translator()
        
        # 통계 정보
        self.stats = {
            'pages_processed': 0,
            'text_blocks_translated': 0,
            'images_processed': 0,
            'ocr_texts_found': 0,
            'errors': []
        }
    
    def _init_translator(self):
        """번역기 초기화"""
        if self.config.service == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
            self.translator = self._translate_openai
        elif self.config.service == 'google':
            self.translator = self._translate_google
        else:
            # 기본값으로 Google 사용
            self.translator = self._translate_google
    
    async def translate_document_async(
        self, 
        input_path: str, 
        output_dir: str, 
        pages: Optional[List[int]] = None,
        progress_callback=None
    ) -> Dict[str, str]:
        """
        문서를 비동기적으로 번역
        
        Args:
            input_path: 입력 PDF 경로
            output_dir: 출력 디렉토리
            pages: 번역할 페이지 목록
            progress_callback: 진행률 콜백
            
        Returns:
            {'mono': 번역본_경로, 'dual': 대조본_경로, 'stats': 통계}
        """
        logger.info(f"하이브리드 번역 시작: {input_path}")
        logger.info(f"설정: pdf2zh={self.pdf2zh_enabled}, OCR={self.ocr_enabled}")
        
        # 출력 경로 설정
        base_name = Path(input_path).stem
        mono_path = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_path = Path(output_dir) / f"{base_name}-dual.pdf"
        
        try:
            # 1단계: pdf2zh로 텍스트 레이어 번역
            if self.pdf2zh_enabled:
                mono_file, dual_file = await self._translate_with_pdf2zh_async(
                    input_path, output_dir, pages, progress_callback
                )
            else:
                # pdf2zh 없이 기본 번역
                mono_file, dual_file = await self._translate_basic_async(
                    input_path, output_dir, pages, progress_callback
                )
            
            # 2단계: OCR로 이미지 추가 번역
            if self.ocr_enabled and mono_file:
                await self._enhance_with_ocr_async(
                    mono_file, output_dir, pages, progress_callback
                )
            
            # 3단계: 최종 파일 정리
            final_files = self._finalize_output(mono_file, dual_file, mono_path, dual_path)
            
            # 메모리 정리
            gc.collect()
            
            return {
                'mono': str(final_files['mono']),
                'dual': str(final_files['dual']),
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"번역 중 오류 발생: {e}", exc_info=True)
            self.stats['errors'].append(str(e))
            raise
        finally:
            # 리소스 정리
            self._cleanup()
    
    async def _translate_with_pdf2zh_async(
        self,
        input_path: str,
        output_dir: str,
        pages: Optional[List[int]],
        progress_callback
    ) -> Tuple[str, str]:
        """pdf2zh를 사용한 비동기 번역"""
        logger.info("pdf2zh 번역 시작")
        
        # pdf2zh는 동기 함수이므로 executor에서 실행
        loop = asyncio.get_event_loop()
        
        def pdf2zh_wrapper():
            """pdf2zh 래퍼 함수"""
            try:
                # ONNX 모델 확인
                if ModelInstance.value is None:
                    ModelInstance.value = OnnxModel.from_pretrained()
                
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
                mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
                dual_file = Path(output_dir) / f"{base_name}-dual.pdf"
                
                return str(mono_file), str(dual_file)
                
            except Exception as e:
                logger.error(f"pdf2zh 번역 오류: {e}")
                raise
        
        # 비동기 실행
        mono_file, dual_file = await loop.run_in_executor(
            self.process_executor,
            pdf2zh_wrapper
        )
        
        self.stats['pages_processed'] += len(pages) if pages else 0
        
        if progress_callback:
            progress_callback(0.5)
        
        return mono_file, dual_file
    
    async def _translate_basic_async(
        self,
        input_path: str,
        output_dir: str,
        pages: Optional[List[int]],
        progress_callback
    ) -> Tuple[str, str]:
        """pdf2zh 없이 기본 번역 (텍스트 레이어만)"""
        logger.info("기본 번역 모드 (pdf2zh 없음)")
        
        # 출력 파일 경로
        base_name = Path(input_path).stem
        mono_path = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_path = Path(output_dir) / f"{base_name}-dual.pdf"
        
        # PDF 열기
        doc = fitz.open(input_path)
        doc_mono = fitz.open()
        doc_dual = fitz.open()
        
        # 페이지 범위 설정
        if pages is None:
            pages = list(range(len(doc)))
        
        # 각 페이지 비동기 처리
        tasks = []
        for page_num in pages:
            task = self._translate_page_async(doc[page_num])
            tasks.append(task)
        
        # 모든 페이지 번역 대기
        translated_pages = await asyncio.gather(*tasks)
        
        # 번역된 페이지 조합
        for idx, (page_num, translated_page) in enumerate(zip(pages, translated_pages)):
            # 번역본에 추가
            doc_mono.insert_pdf(
                fitz.open(stream=translated_page, filetype="pdf"),
                from_page=0,
                to_page=0
            )
            
            # 대조본 생성
            if idx % 2 == 0:
                doc_dual.insert_pdf(doc, from_page=page_num, to_page=page_num)
                doc_dual.insert_pdf(
                    fitz.open(stream=translated_page, filetype="pdf"),
                    from_page=0,
                    to_page=0
                )
            
            if progress_callback:
                progress_callback((idx + 1) / len(pages) * 0.5)
        
        # 저장
        doc_mono.save(str(mono_path))
        doc_dual.save(str(dual_path))
        
        # 정리
        doc.close()
        doc_mono.close()
        doc_dual.close()
        
        return str(mono_path), str(dual_path)
    
    async def _translate_page_async(self, page) -> bytes:
        """페이지를 비동기적으로 번역"""
        loop = asyncio.get_event_loop()
        
        def translate_page_sync():
            """동기 페이지 번역 함수"""
            # 텍스트 추출
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
                        
            # 교정 적용
            page.apply_redactions()
            
            # 번역된 텍스트 추가
            for block in blocks:
                if block[6] == 0:
                    original_text = block[4]
                    if original_text.strip():
                        translated_text = self.translator(original_text)
                        rect = fitz.Rect(block[:4])
                        
                        try:
                            page.insert_textbox(
                                rect,
                                translated_text,
                                fontsize=10,
                                align=fitz.TEXT_ALIGN_LEFT
                            )
                        except:
                            pass
            
            # 페이지를 바이트로 변환
            pdf_bytes = page.parent.write(
                garbage=3,
                deflate=True
            )
            
            self.stats['text_blocks_translated'] += len(blocks)
            
            return pdf_bytes
        
        return await loop.run_in_executor(
            self.thread_executor,
            translate_page_sync
        )
    
    async def _enhance_with_ocr_async(
        self,
        pdf_path: str,
        output_dir: str,
        pages: Optional[List[int]],
        progress_callback
    ):
        """OCR을 사용하여 이미지 내 텍스트 번역 추가"""
        if not self.image_processor:
            return
        
        logger.info("OCR 향상 시작")
        
        doc = fitz.open(pdf_path)
        
        # 페이지 범위 설정
        if pages is None:
            pages = list(range(len(doc)))
        
        # 각 페이지의 이미지 처리
        for idx, page_num in enumerate(pages):
            page = doc[page_num]
            
            # 이미지 추출 및 OCR
            images = await self._process_page_images_async(page)
            
            # 번역된 이미지로 교체
            for img_data in images:
                if img_data.get('has_text'):
                    success = self.image_processor.replace_image_in_page(
                        page,
                        img_data['xref'],
                        img_data['modified_image']
                    )
                    if success:
                        self.stats['images_processed'] += 1
            
            if progress_callback:
                progress_callback(0.5 + (idx + 1) / len(pages) * 0.5)
        
        # 수정된 PDF 저장
        base_name = Path(pdf_path).stem
        enhanced_path = Path(output_dir) / f"{base_name}-ocr.pdf"
        doc.save(str(enhanced_path))
        doc.close()
        
        # 원본 파일 교체
        import shutil
        shutil.move(str(enhanced_path), pdf_path)
        
        logger.info(f"OCR 향상 완료: {self.stats['images_processed']}개 이미지 처리")
    
    async def _process_page_images_async(self, page) -> List[Dict]:
        """페이지의 이미지를 비동기적으로 처리"""
        image_list = page.get_images()
        processed_images = []
        
        for img_info in image_list:
            try:
                xref = img_info[0]
                
                # 이미지 추출
                pix = fitz.Pixmap(page.parent, xref)
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # OCR로 텍스트 감지
                texts = self.image_processor.detect_text_in_image(image)
                
                if texts:
                    self.stats['ocr_texts_found'] += len(texts)
                    
                    # 텍스트 번역
                    for text_info in texts:
                        original = text_info['text']
                        translated = await self._translate_text_async(original)
                        text_info['translated'] = translated
                    
                    # 번역된 텍스트로 이미지 수정
                    modified_image = self.image_processor.create_text_overlay(
                        image,
                        texts,
                        font_path=self.config.font_config.get_font_path(self.config.lang_to),
                        preserve_layout=self.config.ocr_settings.get('preserve_layout', True)
                    )
                    
                    processed_images.append({
                        'xref': xref,
                        'original_image': image,
                        'modified_image': modified_image,
                        'has_text': True,
                        'texts': texts
                    })
                else:
                    processed_images.append({
                        'xref': xref,
                        'original_image': image,
                        'has_text': False
                    })
                
            except Exception as e:
                logger.error(f"이미지 처리 오류: {e}")
                self.stats['errors'].append(f"Image processing: {e}")
                continue
        
        return processed_images
    
    async def _translate_text_async(self, text: str) -> str:
        """텍스트를 비동기적으로 번역"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_executor,
            self.translator,
            text
        )
    
    def _translate_openai(self, text: str) -> str:
        """OpenAI API를 사용한 번역"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model or "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate from {self.config.lang_from} to {self.config.lang_to}. Only return the translation."
                    },
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
            import requests
            import html
            
            # Google Translate 무료 API (비공식)
            url = "https://translate.google.com/m"
            params = {
                'sl': self.config.lang_from,
                'tl': self.config.lang_to,
                'q': text[:5000]  # 최대 길이 제한
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                # HTML 파싱
                import re
                result = re.findall(r'class="(?:t0|result-container)">(.*?)<', response.text)
                if result:
                    return html.unescape(result[0])
            
            return text
            
        except Exception as e:
            logger.error(f"Google 번역 오류: {e}")
            return text
    
    def _finalize_output(
        self,
        mono_file: str,
        dual_file: str,
        final_mono: Path,
        final_dual: Path
    ) -> Dict[str, Path]:
        """출력 파일 최종 처리"""
        import shutil
        
        # 파일 이동 및 이름 변경
        if Path(mono_file).exists():
            shutil.move(mono_file, final_mono)
        
        if Path(dual_file).exists():
            shutil.move(dual_file, final_dual)
        
        logger.info(f"최종 파일: mono={final_mono}, dual={final_dual}")
        
        return {
            'mono': final_mono,
            'dual': final_dual
        }
    
    def _cleanup(self):
        """리소스 정리"""
        if self.image_processor:
            self.image_processor.cleanup()
        
        self.thread_executor.shutdown(wait=False)
        self.process_executor.shutdown(wait=False)
        
        gc.collect()
        logger.info("HybridTranslator 리소스 정리 완료")
    
    # 동기 인터페이스 (하위 호환성)
    def translate_document(
        self,
        input_path: str,
        output_dir: str,
        pages: Optional[List[int]] = None,
        progress_callback=None
    ) -> Dict[str, str]:
        """동기 인터페이스 - 비동기 함수를 동기적으로 실행"""
        return asyncio.run(
            self.translate_document_async(
                input_path, output_dir, pages, progress_callback
            )
        )


# 헬퍼 함수들
import io


def create_progress_bar(total: int, desc: str = "Processing"):
    """진행률 표시줄 생성"""
    return tqdm(total=total, desc=desc)


async def create_async_progress_bar(total: int, desc: str = "Processing"):
    """비동기 진행률 표시줄 생성"""
    return async_tqdm(total=total, desc=desc)

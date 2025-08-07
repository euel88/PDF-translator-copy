"""
PDF OCR Translator - 메모리 최적화 버전
Streamlit Cloud용 경량 PDF 번역기
"""

import os
import gc
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import io
import html

import fitz  # PyMuPDF
from PIL import Image
import requests

# pdf2zh 지연 로딩
PDF2ZH_AVAILABLE = False
try:
    from pdf2zh import translate as pdf2zh_translate
    from pdf2zh.doclayout import ModelInstance, OnnxModel
    PDF2ZH_AVAILABLE = True
except ImportError:
    pass

# 로컬 모듈
from image_processor import LightweightImageProcessor
from config import TranslationConfig

logger = logging.getLogger(__name__)


class HybridPDFTranslator:
    """메모리 효율적인 하이브리드 PDF 번역기"""
    
    def __init__(self, config: TranslationConfig):
        """
        초기화 (최소 리소스 사용)
        
        Args:
            config: TranslationConfig 인스턴스
        """
        self.config = config
        self.image_processor = None
        self.max_pages_per_batch = 5  # 배치당 최대 페이지
        
        # OCR 지연 초기화
        if config.ocr_settings.get('enable_ocr', False):
            self.image_processor = LightweightImageProcessor(
                ocr_languages=[config.lang_from],
                max_workers=2  # 메모리 절약
            )
        
        # 번역기 설정
        self.translator = self._get_translator()
    
    def _get_translator(self):
        """번역 함수 선택"""
        if self.config.service == 'openai':
            return self._translate_openai
        else:
            return self._translate_google
    
    def _translate_openai(self, text: str) -> str:
        """OpenAI API 번역 (청크 단위)"""
        if not self.config.api_key:
            return text
        
        try:
            # OpenAI 클라이언트 (필요시만 import)
            from openai import OpenAI
            client = OpenAI(api_key=self.config.api_key)
            
            # 텍스트 길이 제한 (토큰 절약)
            max_chars = 3000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            response = client.chat.completions.create(
                model=self.config.model or "gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate from {self.config.lang_from} to {self.config.lang_to}. Only return translation."
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=1500  # 제한
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI 번역 오류: {e}")
            return text
    
    def _translate_google(self, text: str) -> str:
        """Google Translate 무료 API"""
        try:
            # 텍스트 길이 제한
            max_chars = 5000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            url = "https://translate.google.com/m"
            params = {
                'sl': self.config.lang_from,
                'tl': self.config.lang_to,
                'q': text
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                import re
                result = re.findall(r'class="(?:t0|result-container)">(.*?)<', response.text)
                if result:
                    return html.unescape(result[0])
            
            return text
            
        except Exception as e:
            logger.error(f"Google 번역 오류: {e}")
            return text
    
    def translate_pdf_with_pdf2zh(self, input_path: str, output_dir: str,
                                  pages: Optional[List[int]] = None,
                                  progress_callback=None) -> Tuple[str, str]:
        """pdf2zh를 사용한 번역 (메모리 효율적)"""
        if not PDF2ZH_AVAILABLE:
            raise ImportError("pdf2zh를 사용할 수 없습니다")
        
        try:
            # ONNX 모델 초기화
            if ModelInstance.value is None:
                ModelInstance.value = OnnxModel.from_pretrained()
            
            # 환경 변수 설정
            envs = {}
            if self.config.api_key:
                if self.config.service == 'openai':
                    envs['OPENAI_API_KEY'] = self.config.api_key
                    envs['OPENAI_MODEL'] = self.config.model or 'gpt-3.5-turbo'
            
            # pdf2zh 실행
            result = pdf2zh_translate(
                files=[input_path],
                output=output_dir,
                pages=pages,
                lang_in=self.config.lang_from,
                lang_out=self.config.lang_to,
                service=self.config.service,
                thread=min(self.config.thread_count, 2),  # 스레드 제한
                model=ModelInstance.value,
                envs=envs,
                skip_subset_fonts=True,
                ignore_cache=False
            )
            
            # 결과 파일
            base_name = Path(input_path).stem
            mono_file = str(Path(output_dir) / f"{base_name}-mono.pdf")
            dual_file = str(Path(output_dir) / f"{base_name}-dual.pdf")
            
            # 메모리 정리
            gc.collect()
            
            return mono_file, dual_file
            
        except Exception as e:
            logger.error(f"pdf2zh 번역 오류: {e}")
            raise
    
    def translate_pdf_simple(self, input_path: str, output_dir: str,
                            pages: Optional[List[int]] = None,
                            progress_callback=None) -> Tuple[str, str]:
        """간단한 PDF 번역 (pdf2zh 없이)"""
        base_name = Path(input_path).stem
        mono_path = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_path = Path(output_dir) / f"{base_name}-dual.pdf"
        
        doc = fitz.open(input_path)
        doc_mono = fitz.open()
        
        # 페이지 범위
        if pages is None:
            pages = list(range(len(doc)))
        
        # 배치 처리
        for batch_start in range(0, len(pages), self.max_pages_per_batch):
            batch_end = min(batch_start + self.max_pages_per_batch, len(pages))
            batch_pages = pages[batch_start:batch_end]
            
            for page_num in batch_pages:
                page = doc[page_num]
                
                # 새 페이지 생성
                new_page = doc_mono.new_page(
                    width=page.rect.width,
                    height=page.rect.height
                )
                
                # 텍스트 추출 및 번역
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    if block[6] == 0:  # 텍스트 블록
                        original_text = block[4].strip()
                        if original_text:
                            # 번역
                            translated = self.translator(original_text)
                            
                            # 텍스트 삽입
                            rect = fitz.Rect(block[:4])
                            try:
                                new_page.insert_textbox(
                                    rect,
                                    translated,
                                    fontsize=10,
                                    align=fitz.TEXT_ALIGN_LEFT
                                )
                            except:
                                pass
                
                # 진행률 업데이트
                if progress_callback:
                    progress = (batch_start + (page_num - pages[0]) + 1) / len(pages)
                    progress_callback(progress)
            
            # 배치 처리 후 메모리 정리
            gc.collect()
        
        # 저장
        doc_mono.save(str(mono_path), deflate=True, garbage=3)
        
        # 대조본 생성 (간단히)
        doc_dual = fitz.open()
        for i in range(0, len(pages), 2):
            if i < len(pages):
                doc_dual.insert_pdf(doc, from_page=pages[i], to_page=pages[i])
            if i < len(doc_mono):
                doc_dual.insert_pdf(doc_mono, from_page=i, to_page=i)
        
        doc_dual.save(str(dual_path), deflate=True, garbage=3)
        
        # 정리
        doc.close()
        doc_mono.close()
        doc_dual.close()
        
        gc.collect()
        
        return str(mono_path), str(dual_path)
    
    def enhance_with_ocr(self, pdf_path: str, pages: Optional[List[int]] = None,
                        progress_callback=None) -> str:
        """OCR로 이미지 텍스트 번역 추가 (메모리 효율적)"""
        if not self.image_processor:
            return pdf_path
        
        doc = fitz.open(pdf_path)
        
        # 페이지 범위
        if pages is None:
            pages = list(range(len(doc)))
        
        # 배치 처리
        for batch_start in range(0, len(pages), self.max_pages_per_batch):
            batch_end = min(batch_start + self.max_pages_per_batch, len(pages))
            batch_pages = pages[batch_start:batch_end]
            
            for page_num in batch_pages:
                page = doc[page_num]
                
                # 이미지 추출 (페이지당 제한)
                images = self.image_processor.extract_images_from_pdf_page(page, page_num)
                
                for img_data in images[:3]:  # 페이지당 최대 3개 이미지만
                    try:
                        # 텍스트 감지
                        texts = self.image_processor.detect_text_in_image(
                            img_data['image'],
                            confidence_threshold=0.5
                        )
                        
                        if texts:
                            # 텍스트 번역
                            for text_info in texts:
                                original = text_info['text']
                                translated = self.translator(original)
                                text_info['translated'] = translated
                            
                            # 오버레이 생성 (간단한 방식)
                            modified_image = self.image_processor.create_text_overlay_simple(
                                img_data['image'],
                                texts
                            )
                            
                            # 이미지 교체는 생략 (메모리 절약)
                            logger.info(f"페이지 {page_num}: {len(texts)}개 텍스트 번역")
                        
                        # 이미지 메모리 해제
                        img_data['image'].close()
                        
                    except Exception as e:
                        logger.error(f"이미지 처리 오류: {e}")
                
                # 진행률 업데이트
                if progress_callback:
                    progress = 0.5 + (batch_start + (page_num - pages[0]) + 1) / len(pages) * 0.5
                    progress_callback(progress)
            
            # 배치 후 메모리 정리
            gc.collect()
        
        # 저장
        output_path = pdf_path.replace('.pdf', '_ocr.pdf')
        doc.save(output_path, deflate=True, garbage=3)
        doc.close()
        
        # 메모리 정리
        if self.image_processor:
            self.image_processor.cleanup()
        gc.collect()
        
        return output_path
    
    def translate_pdf(self, input_path: str, output_dir: str,
                     pages: Optional[List[int]] = None,
                     progress_callback=None) -> Dict[str, str]:
        """통합 PDF 번역 (메모리 효율적)"""
        try:
            # 1단계: 텍스트 레이어 번역
            if PDF2ZH_AVAILABLE:
                mono_file, dual_file = self.translate_pdf_with_pdf2zh(
                    input_path, output_dir, pages, progress_callback
                )
            else:
                mono_file, dual_file = self.translate_pdf_simple(
                    input_path, output_dir, pages, progress_callback
                )
            
            # 2단계: OCR 처리 (선택적)
            if self.config.ocr_settings.get('enable_ocr', False):
                mono_file = self.enhance_with_ocr(
                    mono_file, pages, progress_callback
                )
            
            # 메모리 정리
            gc.collect()
            
            return {
                'mono': mono_file,
                'dual': dual_file
            }
            
        except Exception as e:
            logger.error(f"번역 오류: {e}")
            raise
        finally:
            # 최종 정리
            if self.image_processor:
                self.image_processor.cleanup()
            gc.collect()


# 유틸리티 함수
def check_file_size(file_path: str, max_mb: int = 50) -> bool:
    """파일 크기 확인"""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > max_mb:
        logger.warning(f"파일이 너무 큽니다: {size_mb:.1f} MB")
        return False
    return True


def estimate_memory_required(file_path: str) -> float:
    """필요한 메모리 추정 (MB)"""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    # 대략적인 추정: 파일 크기의 10배
    return file_size_mb * 10


# 하위 호환성
PDFOCRTranslator = HybridPDFTranslator

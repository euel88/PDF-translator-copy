"""
PDF 이미지 번역 모듈
이미지 내 텍스트를 OCR로 추출하고 번역 후 원본 위치에 오버레이
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import pytesseract
import cv2
import easyocr
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class TextRegion:
    """텍스트 영역 정보"""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    language: str
    translated_text: Optional[str] = None

class ImageTranslator:
    """이미지 번역 처리 클래스"""
    
    def __init__(
        self,
        ocr_engine: str = "easyocr",  # "tesseract" or "easyocr"
        translator=None,
        source_lang: str = "en",
        target_lang: str = "ko",
        font_path: Optional[str] = None
    ):
        self.ocr_engine = ocr_engine
        self.translator = translator
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # OCR 엔진 초기화
        if ocr_engine == "easyocr":
            # EasyOCR은 더 나은 아시아 언어 지원
            lang_map = {
                'en': 'en',
                'ko': 'ko',
                'ja': 'ja',
                'zh': 'ch_sim',
                'zh-tw': 'ch_tra'
            }
            ocr_langs = [lang_map.get(source_lang, 'en')]
            if target_lang in lang_map and target_lang != source_lang:
                ocr_langs.append(lang_map[target_lang])
            self.reader = easyocr.Reader(ocr_langs, gpu=False)
        
        # 폰트 설정
        self.font_path = font_path or self._get_default_font()
        
    def _get_default_font(self) -> str:
        """기본 폰트 경로 반환"""
        # pdf2zh와 동일한 폰트 사용
        font_candidates = [
            Path.home() / ".cache" / "pdf2zh" / "fonts" / "GoNotoKurrent-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf"
        ]
        
        for font in font_candidates:
            if font.exists():
                return str(font)
        
        logger.warning("기본 폰트를 찾을 수 없습니다")
        return None
    
    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """이미지에서 텍스트 영역 추출"""
        regions = []
        
        if self.ocr_engine == "easyocr":
            results = self.reader.readtext(image)
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 신뢰도 임계값
                    # bbox를 x1, y1, x2, y2 형식으로 변환
                    points = np.array(bbox)
                    x1, y1 = points.min(axis=0).astype(int)
                    x2, y2 = points.max(axis=0).astype(int)
                    
                    regions.append(TextRegion(
                        text=text,
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        language=self.source_lang
                    ))
        else:
            # Tesseract OCR
            data = pytesseract.image_to_data(
                image, 
                output_type=pytesseract.Output.DICT,
                lang=self._get_tesseract_lang()
            )
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 50:  # 신뢰도 임계값
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = (
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                        regions.append(TextRegion(
                            text=text,
                            bbox=(x, y, x + w, y + h),
                            confidence=float(data['conf'][i]) / 100,
                            language=self.source_lang
                        ))
        
        return regions
    
    def _get_tesseract_lang(self) -> str:
        """Tesseract 언어 코드 반환"""
        lang_map = {
            'en': 'eng',
            'ko': 'kor',
            'ja': 'jpn',
            'zh': 'chi_sim',
            'zh-tw': 'chi_tra',
            'de': 'deu',
            'fr': 'fra',
            'es': 'spa',
            'ru': 'rus'
        }
        return lang_map.get(self.source_lang, 'eng')
    
    def translate_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """텍스트 영역 번역"""
        if not self.translator:
            logger.warning("번역기가 설정되지 않았습니다")
            return regions
        
        for region in regions:
            try:
                # 최소 길이 체크
                if len(region.text.strip()) > 1:
                    region.translated_text = self.translator.translate(region.text)
                else:
                    region.translated_text = region.text
            except Exception as e:
                logger.error(f"번역 오류: {e}")
                region.translated_text = region.text
        
        return regions
    
    def create_overlay_image(
        self, 
        original_image: np.ndarray,
        regions: List[TextRegion],
        overlay_mode: str = "replace"  # "replace", "dual", "below"
    ) -> np.ndarray:
        """번역된 텍스트를 이미지에 오버레이"""
        
        # PIL 이미지로 변환
        if len(original_image.shape) == 2:
            image = Image.fromarray(original_image).convert('RGB')
        else:
            image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        draw = ImageDraw.Draw(image)
        
        for region in regions:
            if not region.translated_text:
                continue
            
            x1, y1, x2, y2 = region.bbox
            
            # 폰트 크기 자동 조정
            box_height = y2 - y1
            font_size = int(box_height * 0.8)
            
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, font_size)
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            if overlay_mode == "replace":
                # 원본 텍스트 영역을 배경색으로 채우기
                draw.rectangle([x1, y1, x2, y2], fill="white")
                # 번역된 텍스트 그리기
                draw.text((x1, y1), region.translated_text, fill="black", font=font)
                
            elif overlay_mode == "dual":
                # 원본과 번역 둘 다 표시
                mid_y = (y1 + y2) // 2
                # 원본 텍스트 영역 반투명 배경
                overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x1, y1, x2, mid_y], fill=(255, 255, 255, 200))
                image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                
                # 번역 텍스트
                draw = ImageDraw.Draw(image)
                draw.text((x1, mid_y), region.translated_text, fill="blue", font=font)
                
            elif overlay_mode == "below":
                # 원본 아래에 번역 추가
                draw.rectangle([x1, y2, x2, y2 + box_height], fill="yellow")
                draw.text((x1, y2), region.translated_text, fill="black", font=font)
        
        # numpy 배열로 변환
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def process_image(
        self,
        image: np.ndarray,
        overlay_mode: str = "replace"
    ) -> Tuple[np.ndarray, List[TextRegion]]:
        """이미지 처리 전체 파이프라인"""
        
        logger.info("텍스트 영역 추출 중...")
        regions = self.extract_text_regions(image)
        logger.info(f"{len(regions)}개 텍스트 영역 발견")
        
        if regions:
            logger.info("텍스트 번역 중...")
            regions = self.translate_regions(regions)
            
            logger.info("오버레이 이미지 생성 중...")
            result_image = self.create_overlay_image(image, regions, overlay_mode)
        else:
            result_image = image
        
        return result_image, regions


class PDFImageTranslator:
    """PDF 문서의 이미지 번역 처리"""
    
    def __init__(
        self,
        image_translator: ImageTranslator,
        min_image_size: int = 100  # 최소 이미지 크기 (픽셀)
    ):
        self.image_translator = image_translator
        self.min_image_size = min_image_size
    
    def extract_and_translate_images(
        self,
        pdf_path: str,
        output_path: Optional[str] = None,
        pages: Optional[List[int]] = None,
        overlay_mode: str = "replace",
        save_extracted_images: bool = False
    ) -> str:
        """PDF에서 이미지 추출 및 번역"""
        
        doc = fitz.open(pdf_path)
        
        if output_path is None:
            output_path = pdf_path.replace('.pdf', '_img_translated.pdf')
        
        # 임시 디렉토리 생성
        temp_dir = Path(tempfile.mkdtemp())
        
        for page_num in range(len(doc)):
            if pages and page_num not in pages:
                continue
            
            logger.info(f"페이지 {page_num + 1} 처리 중...")
            page = doc[page_num]
            
            # 페이지의 모든 이미지 추출
            image_list = page.get_images()
            
            for img_index, img_info in enumerate(image_list):
                try:
                    # 이미지 추출
                    xref = img_info[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # 크기 확인
                    if pix.width < self.min_image_size or pix.height < self.min_image_size:
                        logger.debug(f"이미지 크기가 너무 작음: {pix.width}x{pix.height}")
                        continue
                    
                    # numpy 배열로 변환
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8)
                    img_array = img_array.reshape(pix.height, pix.width, pix.n)
                    
                    # 알파 채널 제거
                    if pix.n == 4:
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                    
                    # 이미지 번역
                    translated_img, regions = self.image_translator.process_image(
                        img_array, overlay_mode
                    )
                    
                    if regions:  # 텍스트가 발견된 경우만
                        logger.info(f"페이지 {page_num + 1}, 이미지 {img_index + 1}: {len(regions)}개 텍스트 번역")
                        
                        # 번역된 이미지 저장
                        if save_extracted_images:
                            img_path = temp_dir / f"page{page_num}_img{img_index}.png"
                            cv2.imwrite(str(img_path), translated_img)
                        
                        # PDF에 이미지 교체
                        self._replace_image_in_pdf(
                            doc, page, xref, translated_img
                        )
                    
                except Exception as e:
                    logger.error(f"이미지 처리 오류 (페이지 {page_num + 1}, 이미지 {img_index + 1}): {e}")
                    continue
        
        # PDF 저장
        doc.save(output_path)
        doc.close()
        
        logger.info(f"번역된 PDF 저장: {output_path}")
        
        # 임시 파일 정리
        if not save_extracted_images:
            import shutil
            shutil.rmtree(temp_dir)
        
        return output_path
    
    def _replace_image_in_pdf(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        xref: int,
        new_image: np.ndarray
    ):
        """PDF 페이지의 이미지 교체"""
        
        # 새 이미지를 PNG로 인코딩
        _, img_buffer = cv2.imencode('.png', new_image)
        img_bytes = img_buffer.tobytes()
        
        # 새 이미지로 교체
        try:
            # 기존 이미지 스트림 교체
            doc.update_stream(xref, img_bytes)
        except:
            # 실패 시 새 이미지 삽입
            logger.warning(f"이미지 교체 실패, 오버레이 방식 사용")
            # 이미지 위치 찾기
            for img in page.get_images():
                if img[0] == xref:
                    # 이미지 위치에 새 이미지 삽입
                    rect = page.get_image_bbox(img)
                    page.insert_image(rect, stream=img_bytes)
                    break


def integrate_with_pdf2zh(
    pdf_path: str,
    output_dir: str,
    translator,
    source_lang: str = "en",
    target_lang: str = "ko",
    translate_images: bool = True,
    overlay_mode: str = "replace",
    ocr_engine: str = "easyocr",
    pages: Optional[List[int]] = None
) -> Tuple[str, str]:
    """pdf2zh와 통합된 이미지 번역"""
    
    from pdf2zh import translate
    import shutil
    
    # 1단계: 이미지 번역이 활성화된 경우
    if translate_images:
        logger.info("이미지 번역 시작...")
        
        # 이미지 번역기 초기화
        img_translator = ImageTranslator(
            ocr_engine=ocr_engine,
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        pdf_img_translator = PDFImageTranslator(img_translator)
        
        # 이미지 번역된 PDF 생성
        temp_pdf = pdf_path.replace('.pdf', '_temp_img.pdf')
        pdf_img_translator.extract_and_translate_images(
            pdf_path,
            temp_pdf,
            pages=pages,
            overlay_mode=overlay_mode
        )
        
        # 이미지 번역된 PDF를 pdf2zh로 처리
        input_file = temp_pdf
    else:
        input_file = pdf_path
    
    # 2단계: pdf2zh로 텍스트 번역
    logger.info("텍스트 번역 시작...")
    
    try:
        # pdf2zh 번역 실행
        result = translate(
            files=[input_file],
            output=output_dir,
            pages=pages,
            lang_in=source_lang,
            lang_out=target_lang,
            service=translator.name if hasattr(translator, 'name') else 'google',
            thread=4
        )
        
        # 결과 파일 경로
        base_name = Path(input_file).stem.replace('_temp_img', '')
        mono_file = Path(output_dir) / f"{base_name}-mono.pdf"
        dual_file = Path(output_dir) / f"{base_name}-dual.pdf"
        
        # 임시 파일 삭제
        if translate_images and Path(temp_pdf).exists():
            os.unlink(temp_pdf)
        
        return str(mono_file), str(dual_file)
        
    except Exception as e:
        logger.error(f"pdf2zh 번역 오류: {e}")
        raise

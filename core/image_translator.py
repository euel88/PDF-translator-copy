"""
Image Translator Module
이미지 내 텍스트를 번역하는 통합 모듈

OCR -> 번역 -> 이미지 편집 파이프라인
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from core.image_ocr import ImageOCR, OCRResult, TextRegion, get_ocr_engine, ocr_image
from core.image_editor import ImageTextEditor, TextReplacement, replace_image_text

logger = logging.getLogger(__name__)


@dataclass
class ImageTranslationResult:
    """이미지 번역 결과"""
    original_image: np.ndarray
    translated_image: np.ndarray
    text_regions: List[TextRegion]
    translations: List[Dict[str, str]]  # [{"original": str, "translated": str}, ...]
    success: bool
    error_message: Optional[str] = None


class ImageTranslator:
    """이미지 번역 클래스"""

    def __init__(
        self,
        translator_func: Optional[Callable[[str, str, str], str]] = None,
        service: str = "google",
        envs: Optional[Dict[str, str]] = None
    ):
        """
        Args:
            translator_func: 번역 함수 (text, lang_in, lang_out) -> translated_text
            service: 번역 서비스 이름
            envs: 환경 변수 (API 키 등)
        """
        self.ocr = get_ocr_engine()
        self.editor = ImageTextEditor()
        self.translator_func = translator_func
        self.service = service
        self.envs = envs or {}

        # 기본 번역 함수 설정
        if self.translator_func is None:
            self.translator_func = self._default_translator

    def _default_translator(
        self,
        text: str,
        lang_in: str,
        lang_out: str
    ) -> str:
        """기본 번역 함수 (pdf2zh의 번역기 사용)"""
        try:
            from pdf2zh.translator import (
                GoogleTranslator,
                OpenAITranslator,
                DeepLTranslator,
                OllamaTranslator,
            )

            # 서비스별 번역기 생성
            if self.service == "openai":
                api_key = self.envs.get("OPENAI_API_KEY", "")
                if not api_key:
                    logger.warning("OpenAI API 키가 없습니다")
                    return text
                translator = OpenAITranslator(
                    lang_in=lang_in,
                    lang_out=lang_out,
                    model="gpt-4o-mini",
                    api_key=api_key
                )
            elif self.service == "deepl":
                api_key = self.envs.get("DEEPL_API_KEY", "")
                if not api_key:
                    return text
                translator = DeepLTranslator(
                    lang_in=lang_in,
                    lang_out=lang_out,
                    api_key=api_key
                )
            elif self.service == "ollama":
                translator = OllamaTranslator(
                    lang_in=lang_in,
                    lang_out=lang_out,
                    model=self.envs.get("OLLAMA_MODEL", "gemma2"),
                    host=self.envs.get("OLLAMA_HOST", "http://localhost:11434")
                )
            else:
                # 기본: Google 번역
                translator = GoogleTranslator(
                    lang_in=lang_in,
                    lang_out=lang_out
                )

            return translator.translate(text)

        except Exception as e:
            logger.error(f"번역 실패: {e}")
            return text

    def translate_image(
        self,
        image: np.ndarray,
        lang_in: str = "ko",
        lang_out: str = "en",
        min_confidence: float = 0.5,
        min_text_length: int = 2,
        use_inpaint: bool = True
    ) -> ImageTranslationResult:
        """
        이미지 내 텍스트 번역

        Args:
            image: numpy 배열 이미지 (BGR)
            lang_in: 원본 언어
            lang_out: 대상 언어
            min_confidence: 최소 OCR 신뢰도
            min_text_length: 최소 텍스트 길이
            use_inpaint: 인페인팅 사용 여부

        Returns:
            ImageTranslationResult: 번역 결과
        """
        try:
            # 1. OCR 수행
            logger.info("OCR 수행 중...")
            ocr_result = ocr_image(image, lang_in, merge_nearby=True)

            if not ocr_result.regions:
                logger.info("이미지에서 텍스트를 찾을 수 없습니다")
                return ImageTranslationResult(
                    original_image=image,
                    translated_image=image.copy(),
                    text_regions=[],
                    translations=[],
                    success=True
                )

            # 2. 텍스트 필터링 및 번역
            logger.info(f"{len(ocr_result.regions)}개 텍스트 영역 발견, 번역 중...")
            translations = []
            replacements = []

            for region in ocr_result.regions:
                # 신뢰도 및 길이 필터링
                if region.confidence < min_confidence:
                    continue
                if len(region.text.strip()) < min_text_length:
                    continue

                # 번역
                original_text = region.text.strip()
                translated_text = self.translator_func(original_text, lang_in, lang_out)

                translations.append({
                    "original": original_text,
                    "translated": translated_text
                })

                replacements.append(TextReplacement(
                    original_text=original_text,
                    translated_text=translated_text,
                    bbox=region.bbox,
                    polygon=region.polygon
                ))

            # 3. 이미지 편집
            logger.info(f"{len(replacements)}개 텍스트 교체 중...")
            translated_image = self.editor.replace_text(
                image,
                replacements,
                lang=lang_out,
                use_inpaint=use_inpaint
            )

            logger.info("이미지 번역 완료")
            return ImageTranslationResult(
                original_image=image,
                translated_image=translated_image,
                text_regions=ocr_result.regions,
                translations=translations,
                success=True
            )

        except Exception as e:
            logger.error(f"이미지 번역 실패: {e}")
            return ImageTranslationResult(
                original_image=image,
                translated_image=image.copy(),
                text_regions=[],
                translations=[],
                success=False,
                error_message=str(e)
            )

    def translate_image_file(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        lang_in: str = "ko",
        lang_out: str = "en",
        **kwargs
    ) -> ImageTranslationResult:
        """
        이미지 파일 번역

        Args:
            image_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (None이면 저장 안 함)
            lang_in: 원본 언어
            lang_out: 대상 언어
            **kwargs: translate_image에 전달할 추가 인자

        Returns:
            ImageTranslationResult: 번역 결과
        """
        import cv2

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            return ImageTranslationResult(
                original_image=np.array([]),
                translated_image=np.array([]),
                text_regions=[],
                translations=[],
                success=False,
                error_message=f"이미지를 읽을 수 없습니다: {image_path}"
            )

        # 번역
        result = self.translate_image(image, lang_in, lang_out, **kwargs)

        # 저장
        if output_path and result.success:
            cv2.imwrite(output_path, result.translated_image)
            logger.info(f"번역된 이미지 저장: {output_path}")

        return result


class PDFImageTranslator:
    """PDF 내 이미지 번역 클래스"""

    def __init__(
        self,
        service: str = "google",
        envs: Optional[Dict[str, str]] = None
    ):
        self.service = service
        self.envs = envs or {}
        self.image_translator = ImageTranslator(service=service, envs=envs)

    def extract_images_from_pdf(
        self,
        pdf_path: str,
        min_size: Tuple[int, int] = (100, 100)
    ) -> List[Dict[str, Any]]:
        """
        PDF에서 이미지 추출

        Args:
            pdf_path: PDF 파일 경로
            min_size: 최소 이미지 크기 (width, height)

        Returns:
            이미지 정보 리스트: [{"page": int, "index": int, "image": np.ndarray, "bbox": tuple, "xref": int}, ...]
        """
        import fitz  # PyMuPDF

        images = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    # 이미지 추출
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # numpy 배열로 변환
                    import cv2
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image is None:
                        continue

                    # 크기 필터링
                    height, width = image.shape[:2]
                    if width < min_size[0] or height < min_size[1]:
                        continue

                    # 이미지 위치 찾기
                    bbox = None
                    for img_rect in page.get_image_rects(xref):
                        bbox = (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)
                        break

                    images.append({
                        "page": page_num,
                        "index": img_index,
                        "image": image,
                        "bbox": bbox,
                        "xref": xref,
                        "ext": image_ext,
                        "width": width,
                        "height": height
                    })

                except Exception as e:
                    logger.warning(f"이미지 추출 실패 (page {page_num}, xref {xref}): {e}")
                    continue

        doc.close()
        logger.info(f"PDF에서 {len(images)}개 이미지 추출")
        return images

    def translate_pdf_images(
        self,
        pdf_path: str,
        output_path: str,
        lang_in: str = "ko",
        lang_out: str = "en",
        min_image_size: Tuple[int, int] = (100, 100),
        callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        PDF 내 이미지 번역 및 교체

        Args:
            pdf_path: 입력 PDF 경로
            output_path: 출력 PDF 경로
            lang_in: 원본 언어
            lang_out: 대상 언어
            min_image_size: 최소 이미지 크기
            callback: 진행 콜백 (current, total)

        Returns:
            처리 결과: {"success": bool, "images_processed": int, "errors": list}
        """
        import fitz  # PyMuPDF
        import cv2

        result = {
            "success": False,
            "images_processed": 0,
            "images_total": 0,
            "errors": []
        }

        try:
            # 이미지 추출
            images = self.extract_images_from_pdf(pdf_path, min_image_size)
            result["images_total"] = len(images)

            if not images:
                logger.info("번역할 이미지가 없습니다")
                # 원본 복사
                import shutil
                shutil.copy(pdf_path, output_path)
                result["success"] = True
                return result

            # PDF 열기
            doc = fitz.open(pdf_path)

            # 각 이미지 번역 및 교체
            for idx, img_info in enumerate(images):
                try:
                    if callback:
                        callback(idx + 1, len(images))

                    # 이미지 번역
                    translation_result = self.image_translator.translate_image(
                        img_info["image"],
                        lang_in=lang_in,
                        lang_out=lang_out
                    )

                    if not translation_result.success:
                        result["errors"].append(f"Page {img_info['page']}: 번역 실패")
                        continue

                    if not translation_result.translations:
                        # 텍스트가 없으면 건너뛰기
                        continue

                    # 번역된 이미지를 PDF에 삽입
                    page = doc[img_info["page"]]

                    # 이미지를 PNG 바이트로 인코딩
                    success, encoded = cv2.imencode(".png", translation_result.translated_image)
                    if not success:
                        result["errors"].append(f"Page {img_info['page']}: 이미지 인코딩 실패")
                        continue

                    # 기존 이미지 교체
                    if img_info["bbox"]:
                        rect = fitz.Rect(img_info["bbox"])

                        # 기존 이미지 삭제 후 새 이미지 삽입
                        # 주의: 직접 교체가 어려워 오버레이 방식 사용
                        page.insert_image(
                            rect,
                            stream=encoded.tobytes(),
                            keep_proportion=True
                        )

                    result["images_processed"] += 1

                except Exception as e:
                    logger.error(f"이미지 처리 실패: {e}")
                    result["errors"].append(f"Page {img_info['page']}: {str(e)}")

            # PDF 저장
            doc.save(output_path)
            doc.close()

            result["success"] = True
            logger.info(f"PDF 이미지 번역 완료: {result['images_processed']}/{result['images_total']}")

        except Exception as e:
            logger.error(f"PDF 이미지 번역 실패: {e}")
            result["errors"].append(str(e))

        return result


def translate_image(
    image: np.ndarray,
    lang_in: str = "ko",
    lang_out: str = "en",
    service: str = "google",
    envs: Optional[Dict[str, str]] = None,
    **kwargs
) -> ImageTranslationResult:
    """
    이미지 번역 (편의 함수)

    Args:
        image: numpy 배열 이미지
        lang_in: 원본 언어
        lang_out: 대상 언어
        service: 번역 서비스
        envs: 환경 변수
        **kwargs: 추가 인자

    Returns:
        ImageTranslationResult: 번역 결과
    """
    translator = ImageTranslator(service=service, envs=envs)
    return translator.translate_image(image, lang_in, lang_out, **kwargs)


def translate_pdf_images(
    pdf_path: str,
    output_path: str,
    lang_in: str = "ko",
    lang_out: str = "en",
    service: str = "google",
    envs: Optional[Dict[str, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    PDF 이미지 번역 (편의 함수)

    Args:
        pdf_path: 입력 PDF 경로
        output_path: 출력 PDF 경로
        lang_in: 원본 언어
        lang_out: 대상 언어
        service: 번역 서비스
        envs: 환경 변수
        **kwargs: 추가 인자

    Returns:
        처리 결과 딕셔너리
    """
    translator = PDFImageTranslator(service=service, envs=envs)
    return translator.translate_pdf_images(
        pdf_path, output_path, lang_in, lang_out, **kwargs
    )

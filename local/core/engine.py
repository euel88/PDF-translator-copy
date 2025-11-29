"""
번역 엔진 - 모든 모듈 통합
"""
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass
from PIL import Image

from .pdf_handler import PDFHandler, PDFInfo
from .ocr import OCR, OCRResult
from .translator import Translator
from .image_processor import ImageProcessor


@dataclass
class TranslationResult:
    """번역 결과"""
    success: bool
    output_path: Optional[str]
    pages_processed: int
    error: Optional[str] = None


class TranslationEngine:
    """PDF 번역 엔진"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        source_lang: str = "English",
        target_lang: str = "Korean",
        ocr_lang: str = "eng",
        dpi: int = 150,
        font_path: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.ocr_lang = ocr_lang
        self.dpi = dpi
        self.font_path = font_path

        # 콜백
        self._log_callback: Optional[Callable[[str], None]] = None
        self._progress_callback: Optional[Callable[[int, str], None]] = None
        self._image_callback: Optional[Callable[[int, Image.Image], None]] = None

        # 취소 플래그
        self._cancelled = False

    def set_callbacks(
        self,
        log: Optional[Callable[[str], None]] = None,
        progress: Optional[Callable[[int, str], None]] = None,
        image: Optional[Callable[[int, Image.Image], None]] = None,
    ):
        """콜백 설정"""
        self._log_callback = log
        self._progress_callback = progress
        self._image_callback = image

    def cancel(self):
        """번역 취소"""
        self._cancelled = True

    def _log(self, msg: str):
        if self._log_callback:
            self._log_callback(msg)

    def _progress(self, pct: int, msg: str):
        if self._progress_callback:
            self._progress_callback(pct, msg)

    def _on_image(self, page: int, img: Image.Image):
        if self._image_callback:
            self._image_callback(page, img)

    def translate(
        self,
        input_path: str,
        output_path: str,
        page_start: int = 0,
        page_end: Optional[int] = None,
        ocr_confidence: float = 0.5,
    ) -> TranslationResult:
        """
        PDF 번역

        Args:
            input_path: 입력 PDF 경로
            output_path: 출력 PDF 경로
            page_start: 시작 페이지 (0-based)
            page_end: 끝 페이지 (포함, None이면 마지막까지)
            ocr_confidence: OCR 신뢰도 임계값

        Returns:
            TranslationResult
        """
        self._cancelled = False

        try:
            self._log("=" * 40)
            self._log("번역 시작")
            self._log(f"입력: {input_path}")
            self._log(f"출력: {output_path}")
            self._progress(0, "초기화 중...")

            # 컴포넌트 초기화
            self._log("OCR 엔진 초기화...")
            ocr = OCR(self.ocr_lang)

            self._log("번역기 초기화...")
            translator = Translator(
                self.api_key,
                self.model,
                self.source_lang,
                self.target_lang,
            )

            self._log("이미지 프로세서 초기화...")
            processor = ImageProcessor(self.font_path)

            self._log(f"설정: {self.source_lang} → {self.target_lang}")
            self._log(f"모델: {self.model}")
            self._log(f"OCR 언어: {self.ocr_lang}")
            self._progress(5, "PDF 로딩 중...")

            # PDF 열기
            with PDFHandler(input_path, self.dpi) as pdf:
                info = pdf.get_info()
                self._log(f"총 페이지: {info.page_count}")
                self._log(f"스캔 PDF: {'예' if info.is_scanned else '아니오'}")

                # 페이지 범위
                if page_end is None:
                    page_end = info.page_count - 1
                page_start = max(0, page_start)
                page_end = min(page_end, info.page_count - 1)

                self._log(f"처리 범위: {page_start + 1} ~ {page_end + 1}")
                self._log("=" * 40)

                total = page_end - page_start + 1
                translated_images: List[Image.Image] = []

                for idx, page_num in enumerate(range(page_start, page_end + 1)):
                    if self._cancelled:
                        self._log("사용자에 의해 취소됨")
                        return TranslationResult(
                            success=False,
                            output_path=None,
                            pages_processed=idx,
                            error="취소됨",
                        )

                    pct = int(10 + (idx / total) * 80)
                    self._progress(pct, f"페이지 {page_num + 1}/{info.page_count}")
                    self._log(f"\n[페이지 {page_num + 1}]")

                    # 렌더링
                    self._log("  렌더링...")
                    img = pdf.render_page(page_num)
                    self._log(f"  크기: {img.width}x{img.height}")

                    # OCR
                    self._log("  OCR...")
                    ocr_result = ocr.extract(img, ocr_confidence)
                    self._log(f"  텍스트 영역: {ocr_result.box_count}개")

                    if ocr_result.boxes:
                        # 샘플 출력
                        for i, box in enumerate(ocr_result.boxes[:3]):
                            preview = box.text[:25] + "..." if len(box.text) > 25 else box.text
                            self._log(f"    [{i+1}] {preview}")
                        if ocr_result.box_count > 3:
                            self._log(f"    ... 외 {ocr_result.box_count - 3}개")

                        # 번역
                        self._log("  번역 중...")
                        texts = [b.text for b in ocr_result.boxes]
                        translations = translator.translate_batch(texts)

                        # 샘플 출력
                        for i, (orig, trans) in enumerate(zip(texts[:2], translations[:2])):
                            o = orig[:15] + "..." if len(orig) > 15 else orig
                            t = trans[:15] + "..." if len(trans) > 15 else trans
                            self._log(f"    \"{o}\" → \"{t}\"")

                        # 이미지 편집
                        self._log("  이미지 편집...")
                        img = processor.replace_all_text(
                            img,
                            ocr_result.boxes,
                            translations,
                        )
                    else:
                        self._log("  텍스트 없음 - 원본 유지")

                    translated_images.append(img)
                    self._on_image(page_num, img)
                    self._log(f"  페이지 {page_num + 1} 완료")

                # PDF 생성
                self._log("\n" + "=" * 40)
                self._progress(95, "PDF 생성 중...")
                self._log("PDF 생성 중...")
                PDFHandler.create_pdf(translated_images, output_path)

                self._progress(100, "완료!")
                self._log(f"저장됨: {output_path}")
                self._log("번역 완료!")
                self._log("=" * 40)

                return TranslationResult(
                    success=True,
                    output_path=output_path,
                    pages_processed=total,
                )

        except Exception as e:
            self._log(f"\n오류: {str(e)}")
            return TranslationResult(
                success=False,
                output_path=None,
                pages_processed=0,
                error=str(e),
            )

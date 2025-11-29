"""
번역 엔진 - 모든 모듈 통합 (개선된 버전)
"""
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass
from PIL import Image

from core.pdf_handler import PDFHandler, PDFInfo, TextBlock
from core.ocr import OCR, OCRResult
from core.translator import Translator
from core.image_processor import ImageProcessor


@dataclass
class TranslationResult:
    """번역 결과"""
    success: bool
    output_path: Optional[str]
    pages_processed: int
    error: Optional[str] = None


class TranslationEngine:
    """PDF 번역 엔진 (개선된 버전)"""

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
        self._log_callback = log
        self._progress_callback = progress
        self._image_callback = image

    def cancel(self):
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
            ocr_confidence: OCR 신뢰도 임계값 (스캔 PDF용)

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

            # 번역기 초기화
            self._log("번역기 초기화...")
            translator = Translator(
                self.api_key,
                self.model,
                self.source_lang,
                self.target_lang,
            )

            # 이미지 프로세서 초기화
            self._log("이미지 프로세서 초기화...")
            processor = ImageProcessor(self.font_path)

            self._log(f"설정: {self.source_lang} → {self.target_lang}")
            self._log(f"모델: {self.model}")
            self._progress(5, "PDF 로딩 중...")

            # PDF 열기
            with PDFHandler(input_path, self.dpi) as pdf:
                info = pdf.get_info()
                self._log(f"총 페이지: {info.page_count}")
                self._log(f"스캔 PDF: {'예' if info.is_scanned else '아니오'}")

                # 스캔 PDF인 경우 OCR 초기화
                ocr = None
                if info.is_scanned:
                    self._log("OCR 엔진 초기화 (스캔 PDF)...")
                    ocr = OCR(self.ocr_lang)
                    self._log(f"OCR 언어: {self.ocr_lang}")
                else:
                    self._log("PDF 텍스트 직접 추출 모드 (정확한 위치)")

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

                    # 텍스트 추출 (디지털 PDF vs 스캔 PDF)
                    if info.is_scanned:
                        # 스캔 PDF: OCR 사용
                        img = self._process_scanned_page(
                            img, ocr, translator, processor, ocr_confidence
                        )
                    else:
                        # 디지털 PDF: PDF 구조에서 직접 추출
                        img = self._process_digital_page(
                            pdf, page_num, img, translator, processor
                        )

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
            import traceback
            self._log(traceback.format_exc())
            return TranslationResult(
                success=False,
                output_path=None,
                pages_processed=0,
                error=str(e),
            )

    def _process_digital_page(
        self,
        pdf: PDFHandler,
        page_num: int,
        img: Image.Image,
        translator: Translator,
        processor: ImageProcessor,
    ) -> Image.Image:
        """디지털 PDF 페이지 처리 (정확한 텍스트 추출)"""

        # PDF에서 직접 텍스트 블록 추출
        self._log("  텍스트 추출 (PDF 구조)...")
        blocks = pdf.extract_text_blocks_scaled(page_num, min_text_len=2)
        self._log(f"  텍스트 블록: {len(blocks)}개")

        if not blocks:
            self._log("  텍스트 없음 - 원본 유지")
            return img

        # 샘플 출력
        for i, block in enumerate(blocks[:3]):
            preview = block.text[:30] + "..." if len(block.text) > 30 else block.text
            self._log(f"    [{i+1}] {preview}")
        if len(blocks) > 3:
            self._log(f"    ... 외 {len(blocks) - 3}개")

        # 번역
        self._log("  번역 중...")
        texts = [b.text for b in blocks]
        translations = translator.translate_batch(texts)

        # 샘플 출력
        for i, (orig, trans) in enumerate(zip(texts[:2], translations[:2])):
            o = orig[:20] + "..." if len(orig) > 20 else orig
            t = trans[:20] + "..." if len(trans) > 20 else trans
            self._log(f"    \"{o}\" → \"{t}\"")

        # 이미지 편집
        self._log("  이미지 편집...")
        img = processor.replace_all_blocks(img, blocks, translations)

        return img

    def _process_scanned_page(
        self,
        img: Image.Image,
        ocr: OCR,
        translator: Translator,
        processor: ImageProcessor,
        ocr_confidence: float,
    ) -> Image.Image:
        """스캔 PDF 페이지 처리 (OCR 사용)"""

        # OCR
        self._log("  OCR...")
        ocr_result = ocr.extract(img, ocr_confidence)
        self._log(f"  텍스트 영역: {ocr_result.box_count}개")

        if not ocr_result.boxes:
            self._log("  텍스트 없음 - 원본 유지")
            return img

        # 샘플 출력
        for i, box in enumerate(ocr_result.boxes[:3]):
            preview = box.text[:30] + "..." if len(box.text) > 30 else box.text
            self._log(f"    [{i+1}] {preview}")
        if ocr_result.box_count > 3:
            self._log(f"    ... 외 {ocr_result.box_count - 3}개")

        # 번역
        self._log("  번역 중...")
        texts = [b.text for b in ocr_result.boxes]
        translations = translator.translate_batch(texts)

        # 이미지 편집
        self._log("  이미지 편집...")
        bboxes = [b.bbox for b in ocr_result.boxes]
        img = processor.replace_all_text(img, bboxes, translations)

        return img

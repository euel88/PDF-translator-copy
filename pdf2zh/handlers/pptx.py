"""
PowerPoint 문서 핸들러 (.pptx, .ppt)
python-pptx 라이브러리를 사용하여 PowerPoint 문서 번역

개선된 기능:
- Run 단위 스타일 완벽 보존
- 대용량 문서 청크 처리
- 동적 배치 크기 조정
"""
from typing import Optional, List, Tuple, Any, Dict
from pathlib import Path
from dataclasses import dataclass

from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult


@dataclass
class RunStyle:
    """Run 스타일 정보"""
    font_name: Optional[str] = None
    font_size: Optional[int] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    font_color_rgb: Optional[Tuple[int, int, int]] = None


class PowerPointHandler(BaseDocumentHandler):
    """PowerPoint 문서 핸들러 - 대용량 처리 개선"""

    SUPPORTED_EXTENSIONS = ['.pptx', '.ppt']

    # 대용량 처리 설정
    BATCH_SIZE_SMALL = 40   # 짧은 텍스트
    BATCH_SIZE_MEDIUM = 25  # 중간 텍스트
    BATCH_SIZE_LARGE = 12   # 긴 텍스트

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> DocumentResult:
        """
        PowerPoint 문서 번역

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로

        Returns:
            DocumentResult: 변환 결과
        """
        try:
            from pptx import Presentation
            from pptx.util import Pt
            from pptx.dml.color import RGBColor
        except ImportError:
            return DocumentResult(
                success=False,
                error="python-pptx 라이브러리가 설치되지 않았습니다. 'pip install python-pptx'를 실행하세요."
            )

        if output_path is None:
            output_path = self.generate_output_path(input_path)

        try:
            self.log(f"PowerPoint 문서 열기: {input_path}")
            prs = Presentation(input_path)

            # 번역할 텍스트 수집 (스타일 정보 포함)
            texts_to_translate = []
            text_locations = []  # 위치 정보
            style_info = []  # 스타일 정보

            for slide_idx, slide in enumerate(prs.slides):
                self.log(f"슬라이드 {slide_idx + 1} 분석 중...")

                for shape_idx, shape in enumerate(slide.shapes):
                    self._collect_shape_texts(
                        shape, slide_idx, shape_idx,
                        texts_to_translate, text_locations, style_info
                    )

            total_count = len(texts_to_translate)
            self.log(f"번역할 텍스트: {total_count}개")

            if total_count == 0:
                prs.save(output_path)
                return DocumentResult(
                    output_path=output_path,
                    success=True,
                    translated_count=0,
                    total_count=0
                )

            # 동적 배치 크기로 번역
            self.log("번역 중...")
            translated_texts = self._translate_with_dynamic_batch(texts_to_translate)

            # 번역 결과 적용
            translated_count = 0
            for location, translated, style in zip(text_locations, translated_texts, style_info):
                try:
                    self._apply_translation(prs, location, translated, style)
                    translated_count += 1
                except Exception as e:
                    self.log(f"경고: 텍스트 적용 실패 - {e}")

            # 저장
            self.log(f"저장 중: {output_path}")
            prs.save(output_path)

            return DocumentResult(
                output_path=output_path,
                success=True,
                translated_count=translated_count,
                total_count=total_count
            )

        except Exception as e:
            import traceback
            return DocumentResult(
                success=False,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )

    def _translate_with_dynamic_batch(self, texts: List[str]) -> List[str]:
        """동적 배치 크기로 번역"""
        total = len(texts)

        # 텍스트 길이에 따라 그룹화
        short_texts = []  # < 100자
        medium_texts = []  # 100-500자
        long_texts = []  # > 500자

        for i, text in enumerate(texts):
            text_len = len(text)
            if text_len < 100:
                short_texts.append((i, text))
            elif text_len < 500:
                medium_texts.append((i, text))
            else:
                long_texts.append((i, text))

        # 결과 저장용 딕셔너리
        result_map = {}

        # 짧은 텍스트 배치 처리
        if short_texts:
            self.log(f"짧은 텍스트 번역: {len(short_texts)}개")
            self._batch_translate(short_texts, result_map, self.BATCH_SIZE_SMALL)

        # 중간 텍스트 배치 처리
        if medium_texts:
            self.log(f"중간 텍스트 번역: {len(medium_texts)}개")
            self._batch_translate(medium_texts, result_map, self.BATCH_SIZE_MEDIUM)

        # 긴 텍스트 배치 처리
        if long_texts:
            self.log(f"긴 텍스트 번역: {len(long_texts)}개")
            self._batch_translate(long_texts, result_map, self.BATCH_SIZE_LARGE)

        # 원래 순서대로 결과 정렬
        translations = []
        for i in range(total):
            translations.append(result_map.get(i, texts[i]))

        return translations

    def _batch_translate(
        self,
        indexed_texts: List[Tuple[int, str]],
        result_map: Dict[int, str],
        batch_size: int
    ):
        """배치 번역 수행"""
        for i in range(0, len(indexed_texts), batch_size):
            batch = indexed_texts[i:i + batch_size]
            texts = [t[1] for t in batch]
            indices = [t[0] for t in batch]

            try:
                translated = self.translate_texts(texts)
                for idx, trans in zip(indices, translated):
                    result_map[idx] = trans
            except Exception as e:
                self.log(f"배치 번역 오류, 개별 번역 시도: {e}")
                # 개별 번역 폴백
                for idx, text in batch:
                    try:
                        result_map[idx] = self.translate_texts([text])[0]
                    except Exception:
                        result_map[idx] = text  # 실패 시 원본 유지

            self.log(f"진행: {min(i + batch_size, len(indexed_texts))}/{len(indexed_texts)}")

    def _collect_shape_texts(
        self,
        shape,
        slide_idx: int,
        shape_idx,
        texts: List[str],
        locations: List[Tuple],
        styles: List[RunStyle]
    ):
        """도형에서 텍스트 수집 (스타일 정보 포함)"""
        # 텍스트 프레임이 있는 도형
        if hasattr(shape, 'text_frame'):
            text_frame = shape.text_frame
            for para_idx, paragraph in enumerate(text_frame.paragraphs):
                for run_idx, run in enumerate(paragraph.runs):
                    if run.text and run.text.strip():
                        texts.append(run.text)
                        locations.append((
                            'shape', slide_idx, shape_idx,
                            para_idx, run_idx
                        ))
                        styles.append(self._extract_run_style(run))

        # 표
        if hasattr(shape, 'table'):
            table = shape.table
            for row_idx, row in enumerate(table.rows):
                for col_idx, cell in enumerate(row.cells):
                    if hasattr(cell, 'text_frame'):
                        for para_idx, paragraph in enumerate(cell.text_frame.paragraphs):
                            for run_idx, run in enumerate(paragraph.runs):
                                if run.text and run.text.strip():
                                    texts.append(run.text)
                                    locations.append((
                                        'table', slide_idx, shape_idx,
                                        row_idx, col_idx, para_idx, run_idx
                                    ))
                                    styles.append(self._extract_run_style(run))

        # 그룹 도형 (재귀)
        if hasattr(shape, 'shapes'):
            for sub_idx, sub_shape in enumerate(shape.shapes):
                self._collect_shape_texts(
                    sub_shape, slide_idx, f"{shape_idx}_{sub_idx}",
                    texts, locations, styles
                )

    def _extract_run_style(self, run) -> RunStyle:
        """Run에서 스타일 정보 추출"""
        font_color_rgb = None
        try:
            if run.font.color and run.font.color.rgb:
                rgb = run.font.color.rgb
                font_color_rgb = (rgb[0], rgb[1], rgb[2])
        except Exception:
            pass

        font_size = None
        try:
            if run.font.size:
                font_size = run.font.size.pt
        except Exception:
            pass

        return RunStyle(
            font_name=run.font.name,
            font_size=font_size,
            bold=run.font.bold,
            italic=run.font.italic,
            underline=run.font.underline,
            font_color_rgb=font_color_rgb
        )

    def _apply_translation(self, prs, location: Tuple, translated: str, style: RunStyle):
        """번역 결과 적용 (스타일 보존)"""
        from pptx.util import Pt
        from pptx.dml.color import RGBColor

        loc_type = location[0]

        if loc_type == 'shape':
            _, slide_idx, shape_idx, para_idx, run_idx = location
            slide = prs.slides[slide_idx]
            shape = self._get_shape_by_index(slide, shape_idx)
            if shape and hasattr(shape, 'text_frame'):
                run = shape.text_frame.paragraphs[para_idx].runs[run_idx]
                run.text = translated
                self._apply_style_to_run(run, style)

        elif loc_type == 'table':
            _, slide_idx, shape_idx, row_idx, col_idx, para_idx, run_idx = location
            slide = prs.slides[slide_idx]
            shape = self._get_shape_by_index(slide, shape_idx)
            if shape and hasattr(shape, 'table'):
                cell = shape.table.rows[row_idx].cells[col_idx]
                run = cell.text_frame.paragraphs[para_idx].runs[run_idx]
                run.text = translated
                self._apply_style_to_run(run, style)

    def _apply_style_to_run(self, run, style: RunStyle):
        """Run에 스타일 적용"""
        from pptx.util import Pt
        from pptx.dml.color import RGBColor

        try:
            if style.font_name:
                run.font.name = style.font_name
            if style.font_size:
                run.font.size = Pt(style.font_size)
            if style.bold is not None:
                run.font.bold = style.bold
            if style.italic is not None:
                run.font.italic = style.italic
            if style.underline is not None:
                run.font.underline = style.underline
            if style.font_color_rgb:
                run.font.color.rgb = RGBColor(*style.font_color_rgb)
        except Exception:
            pass  # 스타일 적용 실패해도 계속 진행

    def _get_shape_by_index(self, slide, shape_idx):
        """인덱스로 도형 가져오기 (중첩 지원)"""
        if isinstance(shape_idx, int):
            return slide.shapes[shape_idx]

        # 중첩된 경우 (예: "0_1")
        if isinstance(shape_idx, str) and '_' in shape_idx:
            indices = [int(i) for i in shape_idx.split('_')]
            shape = slide.shapes[indices[0]]
            for idx in indices[1:]:
                if hasattr(shape, 'shapes'):
                    shape = shape.shapes[idx]
                else:
                    return None
            return shape

        return slide.shapes[int(shape_idx)]

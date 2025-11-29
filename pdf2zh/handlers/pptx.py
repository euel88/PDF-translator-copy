"""
PowerPoint 문서 핸들러 (.pptx, .ppt)
python-pptx 라이브러리를 사용하여 PowerPoint 문서 번역
"""
from typing import Optional, List, Tuple, Any
from pathlib import Path

from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult


class PowerPointHandler(BaseDocumentHandler):
    """PowerPoint 문서 핸들러"""

    SUPPORTED_EXTENSIONS = ['.pptx', '.ppt']

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

            # 번역할 텍스트 수집
            texts_to_translate = []
            text_locations = []  # (slide_idx, shape_idx, paragraph_idx, run_idx)

            for slide_idx, slide in enumerate(prs.slides):
                self.log(f"슬라이드 {slide_idx + 1} 분석 중...")

                for shape_idx, shape in enumerate(slide.shapes):
                    self._collect_shape_texts(
                        shape, slide_idx, shape_idx,
                        texts_to_translate, text_locations
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

            # 배치 번역
            self.log("번역 중...")
            batch_size = 50
            translated_texts = []

            for i in range(0, total_count, batch_size):
                batch = texts_to_translate[i:i + batch_size]
                batch_translated = self.translate_texts(batch)
                translated_texts.extend(batch_translated)
                self.log(f"진행: {min(i + batch_size, total_count)}/{total_count}")

            # 번역 결과 적용
            translated_count = 0
            for location, translated in zip(text_locations, translated_texts):
                try:
                    self._apply_translation(prs, location, translated)
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
            return DocumentResult(
                success=False,
                error=str(e)
            )

    def _collect_shape_texts(
        self,
        shape,
        slide_idx: int,
        shape_idx: int,
        texts: List[str],
        locations: List[Tuple]
    ):
        """도형에서 텍스트 수집"""
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

        # 그룹 도형 (재귀)
        if hasattr(shape, 'shapes'):
            for sub_idx, sub_shape in enumerate(shape.shapes):
                self._collect_shape_texts(
                    sub_shape, slide_idx, f"{shape_idx}_{sub_idx}",
                    texts, locations
                )

    def _apply_translation(self, prs, location: Tuple, translated: str):
        """번역 결과 적용"""
        loc_type = location[0]

        if loc_type == 'shape':
            _, slide_idx, shape_idx, para_idx, run_idx = location
            slide = prs.slides[slide_idx]
            shape = self._get_shape_by_index(slide, shape_idx)
            if shape and hasattr(shape, 'text_frame'):
                run = shape.text_frame.paragraphs[para_idx].runs[run_idx]
                run.text = translated

        elif loc_type == 'table':
            _, slide_idx, shape_idx, row_idx, col_idx, para_idx, run_idx = location
            slide = prs.slides[slide_idx]
            shape = self._get_shape_by_index(slide, shape_idx)
            if shape and hasattr(shape, 'table'):
                cell = shape.table.rows[row_idx].cells[col_idx]
                run = cell.text_frame.paragraphs[para_idx].runs[run_idx]
                run.text = translated

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

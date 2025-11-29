"""
Word 문서 핸들러 (.docx, .doc)
python-docx 라이브러리를 사용하여 Word 문서 번역
원본의 페이지 구성, 문장 위치, 스타일을 완벽히 보존
"""
from typing import Optional, List, Tuple, Any
from pathlib import Path

from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult


class WordHandler(BaseDocumentHandler):
    """Word 문서 핸들러 - 레이아웃 및 스타일 완벽 보존"""

    SUPPORTED_EXTENSIONS = ['.docx', '.doc']

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> DocumentResult:
        """
        Word 문서 번역 - 원본 레이아웃 및 스타일 보존

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로

        Returns:
            DocumentResult: 변환 결과
        """
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError:
            return DocumentResult(
                success=False,
                error="python-docx 라이브러리가 설치되지 않았습니다. 'pip install python-docx'를 실행하세요."
            )

        if output_path is None:
            output_path = self.generate_output_path(input_path)

        try:
            self.log(f"Word 문서 열기: {input_path}")
            doc = Document(input_path)

            # Run 단위로 텍스트 수집 (스타일 완벽 보존을 위해)
            runs_to_translate: List[Tuple[Any, str]] = []  # (run 객체, 원본 텍스트)

            # 1. 본문 단락의 run 수집
            for para in doc.paragraphs:
                self._collect_paragraph_runs(para, runs_to_translate)

            # 2. 표 내 모든 셀의 모든 단락 run 수집
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for para in cell.paragraphs:
                            self._collect_paragraph_runs(para, runs_to_translate)

            # 3. 헤더/푸터의 run 수집
            for section in doc.sections:
                if section.header:
                    for para in section.header.paragraphs:
                        self._collect_paragraph_runs(para, runs_to_translate)
                if section.footer:
                    for para in section.footer.paragraphs:
                        self._collect_paragraph_runs(para, runs_to_translate)

            total_count = len(runs_to_translate)
            self.log(f"번역할 텍스트 run: {total_count}개")

            if total_count == 0:
                doc.save(output_path)
                return DocumentResult(
                    output_path=output_path,
                    success=True,
                    translated_count=0,
                    total_count=0
                )

            # 텍스트만 추출하여 배치 번역
            texts_to_translate = [text for _, text in runs_to_translate]

            # 배치 번역 (50개씩 분할)
            self.log("번역 중...")
            batch_size = 50
            translated_texts = []

            for i in range(0, total_count, batch_size):
                batch = texts_to_translate[i:i + batch_size]
                self.log(f"배치 번역: {i + 1}-{min(i + len(batch), total_count)}/{total_count}")
                batch_translated = self.translate_texts(batch)
                translated_texts.extend(batch_translated)

            # 번역 결과를 각 run에 직접 적용 (스타일 완벽 보존)
            translated_count = 0
            for idx, ((run, _), translated) in enumerate(zip(runs_to_translate, translated_texts)):
                try:
                    # run.text를 직접 교체하면 스타일이 그대로 유지됨
                    run.text = translated
                    translated_count += 1

                    # 진행률 로그
                    if (idx + 1) % 50 == 0 or idx + 1 == total_count:
                        self.log(f"적용 진행: {idx + 1}/{total_count}")

                except Exception as e:
                    self.log(f"경고: run {idx} 번역 적용 실패 - {e}")

            # 저장
            self.log(f"저장 중: {output_path}")
            doc.save(output_path)

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

    def _collect_paragraph_runs(
        self,
        paragraph,
        runs_list: List[Tuple[Any, str]]
    ):
        """
        단락에서 번역할 run들을 수집

        각 run은 고유한 스타일(굵기, 이탤릭, 글꼴, 색상 등)을 가지므로
        run 단위로 번역해야 원본 스타일이 완벽히 보존됨

        Args:
            paragraph: 단락 객체
            runs_list: 수집된 (run, 텍스트) 튜플 리스트
        """
        for run in paragraph.runs:
            text = run.text
            # 의미 있는 텍스트가 있는 run만 번역 대상
            # 공백만 있는 run은 레이아웃 유지를 위해 그대로 보존
            if text and text.strip():
                runs_list.append((run, text))

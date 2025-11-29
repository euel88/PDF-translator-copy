"""
Word 문서 핸들러 (.docx, .doc)
python-docx 라이브러리를 사용하여 Word 문서 번역
"""
from typing import Optional, List
from pathlib import Path

from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult


class WordHandler(BaseDocumentHandler):
    """Word 문서 핸들러"""

    SUPPORTED_EXTENSIONS = ['.docx', '.doc']

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> DocumentResult:
        """
        Word 문서 번역

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

            # 번역할 텍스트 수집
            texts_to_translate = []
            text_locations = []  # (type, index, sub_index, ...)

            # 1. 단락 수집
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    texts_to_translate.append(para.text)
                    text_locations.append(('paragraph', i))

            # 2. 표 수집
            for t_idx, table in enumerate(doc.tables):
                for r_idx, row in enumerate(table.rows):
                    for c_idx, cell in enumerate(row.cells):
                        if cell.text.strip():
                            texts_to_translate.append(cell.text)
                            text_locations.append(('table', t_idx, r_idx, c_idx))

            # 3. 헤더/푸터 수집
            for section in doc.sections:
                # 헤더
                if section.header:
                    for i, para in enumerate(section.header.paragraphs):
                        if para.text.strip():
                            texts_to_translate.append(para.text)
                            text_locations.append(('header', id(section), i))
                # 푸터
                if section.footer:
                    for i, para in enumerate(section.footer.paragraphs):
                        if para.text.strip():
                            texts_to_translate.append(para.text)
                            text_locations.append(('footer', id(section), i))

            total_count = len(texts_to_translate)
            self.log(f"번역할 텍스트: {total_count}개")

            if total_count == 0:
                doc.save(output_path)
                return DocumentResult(
                    output_path=output_path,
                    success=True,
                    translated_count=0,
                    total_count=0
                )

            # 배치 번역
            self.log("번역 중...")
            translated_texts = self.translate_texts(texts_to_translate)

            # 번역 결과 적용
            translated_count = 0
            for idx, (location, translated) in enumerate(zip(text_locations, translated_texts)):
                try:
                    if location[0] == 'paragraph':
                        para_idx = location[1]
                        para = doc.paragraphs[para_idx]
                        # 스타일 보존하며 텍스트 교체
                        self._replace_paragraph_text(para, translated)
                        translated_count += 1

                    elif location[0] == 'table':
                        t_idx, r_idx, c_idx = location[1], location[2], location[3]
                        cell = doc.tables[t_idx].rows[r_idx].cells[c_idx]
                        # 셀 내 첫 번째 단락만 교체
                        if cell.paragraphs:
                            self._replace_paragraph_text(cell.paragraphs[0], translated)
                        translated_count += 1

                    elif location[0] == 'header':
                        section_id, para_idx = location[1], location[2]
                        for section in doc.sections:
                            if id(section) == section_id and section.header:
                                self._replace_paragraph_text(
                                    section.header.paragraphs[para_idx], translated
                                )
                                translated_count += 1
                                break

                    elif location[0] == 'footer':
                        section_id, para_idx = location[1], location[2]
                        for section in doc.sections:
                            if id(section) == section_id and section.footer:
                                self._replace_paragraph_text(
                                    section.footer.paragraphs[para_idx], translated
                                )
                                translated_count += 1
                                break

                    # 진행률 로그
                    if (idx + 1) % 10 == 0 or idx + 1 == total_count:
                        self.log(f"진행: {idx + 1}/{total_count}")

                except Exception as e:
                    self.log(f"경고: 항목 {idx} 번역 적용 실패 - {e}")

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

    def _replace_paragraph_text(self, paragraph, new_text: str):
        """
        단락 텍스트를 스타일 보존하며 교체
        첫 번째 run의 스타일을 유지하고 나머지는 삭제
        """
        if not paragraph.runs:
            paragraph.text = new_text
            return

        # 첫 번째 run의 스타일 정보 저장
        first_run = paragraph.runs[0]

        # 모든 run 삭제
        for run in paragraph.runs:
            run.clear()

        # 첫 번째 run에 새 텍스트 설정
        first_run.text = new_text

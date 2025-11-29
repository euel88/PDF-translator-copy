"""
Excel 문서 핸들러 (.xlsx, .xls)
openpyxl 라이브러리를 사용하여 Excel 문서 번역
"""
from typing import Optional, List, Tuple
from pathlib import Path

from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult


class ExcelHandler(BaseDocumentHandler):
    """Excel 문서 핸들러"""

    SUPPORTED_EXTENSIONS = ['.xlsx', '.xls']

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> DocumentResult:
        """
        Excel 문서 번역

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로

        Returns:
            DocumentResult: 변환 결과
        """
        try:
            from openpyxl import load_workbook
        except ImportError:
            return DocumentResult(
                success=False,
                error="openpyxl 라이브러리가 설치되지 않았습니다. 'pip install openpyxl'을 실행하세요."
            )

        if output_path is None:
            output_path = self.generate_output_path(input_path)

        try:
            self.log(f"Excel 문서 열기: {input_path}")
            # data_only=False로 수식 보존
            workbook = load_workbook(input_path, data_only=False)

            # 번역할 텍스트 수집
            texts_to_translate = []
            cell_locations = []  # (sheet_name, row, col)

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                self.log(f"시트 분석: {sheet_name}")

                for row_idx, row in enumerate(sheet.iter_rows(), start=1):
                    for col_idx, cell in enumerate(row, start=1):
                        if self._should_translate_cell(cell):
                            texts_to_translate.append(str(cell.value))
                            cell_locations.append((sheet_name, row_idx, col_idx))

            total_count = len(texts_to_translate)
            self.log(f"번역할 셀: {total_count}개")

            if total_count == 0:
                workbook.save(output_path)
                return DocumentResult(
                    output_path=output_path,
                    success=True,
                    translated_count=0,
                    total_count=0
                )

            # 배치 번역 (너무 많으면 분할)
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
            for (sheet_name, row, col), translated in zip(cell_locations, translated_texts):
                try:
                    sheet = workbook[sheet_name]
                    cell = sheet.cell(row=row, column=col)
                    cell.value = translated
                    translated_count += 1
                except Exception as e:
                    self.log(f"경고: 셀 ({sheet_name}, {row}, {col}) 번역 적용 실패 - {e}")

            # 저장
            self.log(f"저장 중: {output_path}")
            workbook.save(output_path)

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

    def _should_translate_cell(self, cell) -> bool:
        """셀을 번역해야 하는지 확인"""
        if cell.value is None:
            return False

        value = cell.value

        # 문자열만 번역
        if not isinstance(value, str):
            return False

        # 빈 문자열 제외
        if not value.strip():
            return False

        # 수식 제외 (=로 시작)
        if value.startswith('='):
            return False

        # 숫자만 있는 경우 제외
        try:
            float(value.replace(',', '').replace('%', ''))
            return False
        except ValueError:
            pass

        # URL 제외
        if value.startswith(('http://', 'https://', 'www.')):
            return False

        # 이메일 제외
        if '@' in value and '.' in value.split('@')[-1]:
            return False

        # 최소 길이 (단일 문자/숫자 제외)
        if len(value.strip()) < 2:
            return False

        return True

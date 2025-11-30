"""
Excel 문서 핸들러 (.xlsx, .xls)
openpyxl 라이브러리를 사용하여 Excel 문서 번역

개선된 기능:
- 대용량 문서 청크 처리
- 동적 배치 크기 조정
- 셀 스타일 완벽 보존
"""
from typing import Optional, List, Tuple, Dict
from pathlib import Path

from pdf2zh.handlers.base import BaseDocumentHandler, DocumentResult


class ExcelHandler(BaseDocumentHandler):
    """Excel 문서 핸들러 - 대용량 처리 개선"""

    SUPPORTED_EXTENSIONS = ['.xlsx', '.xls']

    # 대용량 처리 설정
    BATCH_SIZE_SMALL = 50   # 짧은 텍스트
    BATCH_SIZE_MEDIUM = 30  # 중간 텍스트
    BATCH_SIZE_LARGE = 15   # 긴 텍스트

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

            # 동적 배치 크기로 번역
            self.log("번역 중...")
            translated_texts = self._translate_with_dynamic_batch(texts_to_translate)

            # 번역 결과 적용 (스타일 보존)
            translated_count = 0
            for (sheet_name, row, col), translated in zip(cell_locations, translated_texts):
                try:
                    sheet = workbook[sheet_name]
                    cell = sheet.cell(row=row, column=col)
                    # 기존 스타일 유지하며 값만 변경
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
            float(value.replace(',', '').replace('%', '').replace(' ', ''))
            return False
        except ValueError:
            pass

        # URL 제외
        if value.startswith(('http://', 'https://', 'www.', 'ftp://')):
            return False

        # 이메일 제외
        if '@' in value and '.' in value.split('@')[-1]:
            return False

        # 파일 경로 제외
        if '\\' in value or (value.startswith('/') and '/' in value[1:]):
            return False

        # 최소 길이 (단일 문자/숫자 제외)
        if len(value.strip()) < 2:
            return False

        return True

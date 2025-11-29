"""
메인 윈도우 - PDF 번역기 GUI
"""
import os
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox,
    QSplitter, QScrollArea, QMessageBox, QStatusBar,
    QToolBar, QAction, QFrame, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QTextCursor

from PIL import Image
import numpy as np

# 상위 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigManager, LANGUAGES, OCR_LANGUAGES
from core.ocr_engine import OCREngine, TextRegion
from core.translator import OpenAITranslator
from core.pdf_processor import PDFProcessor
from core.image_editor import ImageEditor


class TranslationWorker(QThread):
    """번역 작업 스레드"""

    progress = pyqtSignal(int, str)  # 진행률, 메시지
    log_message = pyqtSignal(str)  # 상세 로그 메시지
    page_completed = pyqtSignal(int, object)  # 페이지 번호, 이미지
    finished = pyqtSignal(bool, str)  # 성공 여부, 메시지
    error = pyqtSignal(str)

    def __init__(
        self,
        pdf_path: str,
        output_path: str,
        config: ConfigManager,
        page_range: Optional[tuple] = None
    ):
        super().__init__()
        self.pdf_path = pdf_path
        self.output_path = output_path
        self.config = config
        self.page_range = page_range
        self._is_cancelled = False

    def cancel(self):
        """작업 취소"""
        self._is_cancelled = True

    def _log(self, message: str):
        """로그 메시지 전송"""
        self.log_message.emit(message)

    def run(self):
        """번역 작업 실행"""
        try:
            # 초기화
            self.progress.emit(0, "초기화 중...")
            self._log("=" * 40)
            self._log("번역 작업 시작")
            self._log(f"입력: {self.pdf_path}")
            self._log(f"출력: {self.output_path}")

            # OCR 엔진 초기화
            self._log("OCR 엔진 초기화 중...")
            ocr = OCREngine()
            ocr_lang = OCR_LANGUAGES.get(
                self.config.config.source_lang, "eng"
            )
            ocr.set_language(ocr_lang)
            self._log(f"OCR 언어: {ocr_lang}")

            # 번역기 초기화
            self._log("번역기 초기화 중...")
            translator = OpenAITranslator(
                api_key=self.config.config.openai_api_key,
                model=self.config.config.openai_model,
                source_lang=self.config.config.source_lang,
                target_lang=self.config.config.target_lang
            )
            self._log(f"번역: {self.config.config.source_lang} → {self.config.config.target_lang}")
            self._log(f"모델: {self.config.config.openai_model}")

            # 이미지 편집기 초기화
            editor = ImageEditor(self.config.config.font_path or None)

            # PDF 처리
            self.progress.emit(5, "PDF 로딩 중...")
            self._log("PDF 로딩 중...")

            with PDFProcessor(
                self.pdf_path,
                dpi=self.config.config.render_dpi
            ) as processor:

                page_count = processor.get_page_count()
                self._log(f"총 페이지: {page_count}")
                self._log(f"렌더링 DPI: {self.config.config.render_dpi}")

                # 페이지 범위 결정
                if self.page_range:
                    start, end = self.page_range
                    start = max(0, start)
                    end = min(end, page_count - 1)
                else:
                    start, end = 0, page_count - 1

                self._log(f"처리 범위: {start + 1} ~ {end + 1} 페이지")
                self._log("=" * 40)

                total_pages = end - start + 1
                translated_images = []

                for idx, page_num in enumerate(range(start, end + 1)):
                    if self._is_cancelled:
                        self._log("사용자에 의해 취소됨")
                        self.finished.emit(False, "사용자에 의해 취소됨")
                        return

                    progress_pct = int(10 + (idx / total_pages) * 80)
                    self.progress.emit(
                        progress_pct,
                        f"페이지 {page_num + 1}/{page_count} 처리 중..."
                    )

                    self._log(f"\n[페이지 {page_num + 1}/{page_count}]")

                    # 페이지 렌더링
                    self._log("  렌더링 중...")
                    page_image = processor.render_page(page_num)
                    pil_image = page_image.image
                    self._log(f"  이미지 크기: {pil_image.width}x{pil_image.height}")

                    # OCR
                    self._log("  OCR 실행 중...")
                    ocr_result = ocr.extract_text_from_pil(
                        pil_image,
                        self.config.config.ocr_confidence_threshold
                    )

                    self._log(f"  감지된 텍스트 영역: {len(ocr_result.regions)}개")

                    if ocr_result.regions:
                        # 감지된 텍스트 로그
                        for i, region in enumerate(ocr_result.regions[:5]):  # 최대 5개만 표시
                            text_preview = region.text[:30] + "..." if len(region.text) > 30 else region.text
                            self._log(f"    [{i+1}] {text_preview} (신뢰도: {region.confidence:.2f})")

                        if len(ocr_result.regions) > 5:
                            self._log(f"    ... 외 {len(ocr_result.regions) - 5}개")

                        # 번역
                        self._log("  번역 중...")
                        texts = [r.text for r in ocr_result.regions]
                        translations = translator.translate_batch(texts)

                        # 번역 결과 로그
                        for i, (orig, trans) in enumerate(zip(texts[:3], translations[:3])):
                            orig_preview = orig[:20] + "..." if len(orig) > 20 else orig
                            trans_preview = trans[:20] + "..." if len(trans) > 20 else trans
                            self._log(f"    [{i+1}] \"{orig_preview}\" → \"{trans_preview}\"")

                        if len(texts) > 3:
                            self._log(f"    ... 외 {len(texts) - 3}개 번역됨")

                        # 이미지에 번역 적용
                        self._log("  이미지 편집 중...")
                        translated_img = editor.replace_all_text(
                            pil_image,
                            ocr_result.regions,
                            translations,
                            erase_method="inpaint"
                        )
                        self._log("  편집 완료")
                    else:
                        self._log("  텍스트 없음 - 원본 이미지 유지")
                        translated_img = pil_image

                    translated_images.append(translated_img)
                    self.page_completed.emit(page_num, translated_img)
                    self._log(f"  페이지 {page_num + 1} 완료")

                # PDF 생성
                self._log("\n" + "=" * 40)
                self.progress.emit(95, "PDF 생성 중...")
                self._log("PDF 생성 중...")
                PDFProcessor.create_pdf_from_images(
                    translated_images,
                    self.output_path
                )

                self.progress.emit(100, "완료!")
                self._log(f"PDF 저장됨: {self.output_path}")
                self._log("번역 작업 완료!")
                self._log("=" * 40)
                self.finished.emit(True, f"번역 완료: {self.output_path}")

        except Exception as e:
            self._log(f"\n오류 발생: {str(e)}")
            self.error.emit(str(e))
            self.finished.emit(False, f"오류 발생: {e}")


class ImagePreviewWidget(QScrollArea):
    """이미지 미리보기 위젯"""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: #f0f0f0;")
        self.setWidget(self.label)

        self.current_pixmap = None

    def set_image(self, image: Image.Image):
        """PIL 이미지 표시"""
        # PIL -> QPixmap 변환
        if image.mode != "RGB":
            image = image.convert("RGB")

        data = image.tobytes("raw", "RGB")
        qimg = QImage(
            data,
            image.width,
            image.height,
            image.width * 3,
            QImage.Format_RGB888
        )
        self.current_pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        """디스플레이 업데이트"""
        if self.current_pixmap:
            # 스크롤 영역에 맞게 크기 조정
            scaled = self.current_pixmap.scaled(
                self.size() * 0.95,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.label.setPixmap(scaled)

    def resizeEvent(self, event):
        """크기 변경 시 이미지 재조정"""
        super().resizeEvent(event)
        self._update_display()

    def clear(self):
        """이미지 지우기"""
        self.label.clear()
        self.current_pixmap = None


class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        self.current_pdf_path = None
        self.worker = None

        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("PDF Translator Local")
        self.setMinimumSize(1200, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 메인 레이아웃
        main_layout = QHBoxLayout(central_widget)

        # 왼쪽 패널 (설정)
        left_panel = self._create_left_panel()
        left_panel.setMaximumWidth(350)

        # 오른쪽 패널 (미리보기)
        right_panel = self._create_right_panel()

        # 스플리터
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])

        main_layout.addWidget(splitter)

        # 상태바
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("준비됨")

        # 메뉴바
        self._create_menubar()

    def _create_menubar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()

        # 파일 메뉴
        file_menu = menubar.addMenu("파일")

        open_action = QAction("PDF 열기", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_pdf)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("종료", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 설정 메뉴
        settings_menu = menubar.addMenu("설정")

        api_action = QAction("API 설정", self)
        api_action.triggered.connect(self._open_settings)
        settings_menu.addAction(api_action)

    def _create_left_panel(self) -> QWidget:
        """왼쪽 패널 생성 (설정)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 파일 선택
        file_group = QGroupBox("PDF 파일")
        file_layout = QVBoxLayout(file_group)

        self.file_label = QLabel("파일을 선택하세요")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        file_btn = QPushButton("PDF 열기")
        file_btn.clicked.connect(self._open_pdf)
        file_layout.addWidget(file_btn)

        layout.addWidget(file_group)

        # 언어 설정
        lang_group = QGroupBox("언어 설정")
        lang_layout = QVBoxLayout(lang_group)

        # 원본 언어
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("원본:"))
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(list(LANGUAGES.keys()))
        source_layout.addWidget(self.source_lang_combo)
        lang_layout.addLayout(source_layout)

        # 대상 언어
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("번역:"))
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(list(LANGUAGES.keys()))
        target_layout.addWidget(self.target_lang_combo)
        lang_layout.addLayout(target_layout)

        layout.addWidget(lang_group)

        # 페이지 설정
        page_group = QGroupBox("페이지 설정")
        page_layout = QVBoxLayout(page_group)

        page_range_layout = QHBoxLayout()
        page_range_layout.addWidget(QLabel("시작:"))
        self.start_page_spin = QSpinBox()
        self.start_page_spin.setMinimum(1)
        self.start_page_spin.setMaximum(9999)
        page_range_layout.addWidget(self.start_page_spin)

        page_range_layout.addWidget(QLabel("끝:"))
        self.end_page_spin = QSpinBox()
        self.end_page_spin.setMinimum(1)
        self.end_page_spin.setMaximum(9999)
        page_range_layout.addWidget(self.end_page_spin)

        page_layout.addLayout(page_range_layout)

        self.page_info_label = QLabel("총 페이지: -")
        page_layout.addWidget(self.page_info_label)

        layout.addWidget(page_group)

        # 고급 설정
        advanced_group = QGroupBox("고급 설정")
        advanced_layout = QVBoxLayout(advanced_group)

        # DPI 설정
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("렌더링 DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 300)
        self.dpi_spin.setValue(150)
        dpi_layout.addWidget(self.dpi_spin)
        advanced_layout.addLayout(dpi_layout)

        # 모델 선택
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("OpenAI 모델:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        model_layout.addWidget(self.model_combo)
        advanced_layout.addLayout(model_layout)

        layout.addWidget(advanced_group)

        # 진행 상태
        progress_group = QGroupBox("진행 상태")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("대기 중")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # 버튼
        btn_layout = QHBoxLayout()

        self.translate_btn = QPushButton("번역 시작")
        self.translate_btn.setEnabled(False)
        self.translate_btn.clicked.connect(self._start_translation)
        self.translate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        btn_layout.addWidget(self.translate_btn)

        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_translation)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        # 빈 공간
        layout.addStretch()

        return panel

    def _create_right_panel(self) -> QWidget:
        """오른쪽 패널 생성 (미리보기)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 미리보기 라벨
        preview_label = QLabel("페이지 미리보기")
        preview_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(preview_label)

        # 미리보기 위젯
        self.preview_widget = ImagePreviewWidget()
        layout.addWidget(self.preview_widget)

        # 로그
        log_group = QGroupBox("실시간 로그")
        log_layout = QVBoxLayout(log_group)

        # 로그 컨트롤
        log_control_layout = QHBoxLayout()

        self.auto_scroll_check = QCheckBox("자동 스크롤")
        self.auto_scroll_check.setChecked(True)
        log_control_layout.addWidget(self.auto_scroll_check)

        log_control_layout.addStretch()

        clear_log_btn = QPushButton("로그 지우기")
        clear_log_btn.clicked.connect(self._clear_log)
        log_control_layout.addWidget(clear_log_btn)

        log_layout.addLayout(log_control_layout)

        # 로그 텍스트
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        return panel

    def _load_settings(self):
        """설정 로드"""
        config = self.config.config

        # 언어 설정
        idx = self.source_lang_combo.findText(config.source_lang)
        if idx >= 0:
            self.source_lang_combo.setCurrentIndex(idx)

        idx = self.target_lang_combo.findText(config.target_lang)
        if idx >= 0:
            self.target_lang_combo.setCurrentIndex(idx)

        # DPI
        self.dpi_spin.setValue(config.render_dpi)

        # 모델
        idx = self.model_combo.findText(config.openai_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _save_settings(self):
        """설정 저장"""
        self.config.config.source_lang = self.source_lang_combo.currentText()
        self.config.config.target_lang = self.target_lang_combo.currentText()
        self.config.config.render_dpi = self.dpi_spin.value()
        self.config.config.openai_model = self.model_combo.currentText()
        self.config.save_config()

    def _open_pdf(self):
        """PDF 파일 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "PDF 파일 선택",
            "",
            "PDF Files (*.pdf)"
        )

        if file_path:
            self.current_pdf_path = file_path
            self.file_label.setText(Path(file_path).name)

            # PDF 정보 가져오기
            try:
                with PDFProcessor(file_path) as processor:
                    info = processor.get_info()
                    self.page_info_label.setText(f"총 페이지: {info.page_count}")

                    self.start_page_spin.setMaximum(info.page_count)
                    self.end_page_spin.setMaximum(info.page_count)
                    self.start_page_spin.setValue(1)
                    self.end_page_spin.setValue(info.page_count)

                    # 첫 페이지 미리보기
                    page_img = processor.render_page(0)
                    self.preview_widget.set_image(page_img.image)

                self.translate_btn.setEnabled(True)
                self.statusbar.showMessage(f"PDF 로드됨: {info.page_count} 페이지")
                self._log(f"PDF 열림: {file_path}")
                self._log(f"총 {info.page_count} 페이지")

            except Exception as e:
                QMessageBox.critical(self, "오류", f"PDF 로드 실패: {e}")
                self._log(f"오류: {e}")

    def _open_settings(self):
        """설정 다이얼로그 열기"""
        from .settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_():
            self._log("설정이 저장되었습니다")

    def _start_translation(self):
        """번역 시작"""
        if not self.current_pdf_path:
            QMessageBox.warning(self, "경고", "PDF 파일을 먼저 선택하세요")
            return

        if not self.config.config.openai_api_key:
            QMessageBox.warning(
                self, "경고",
                "OpenAI API 키가 설정되지 않았습니다.\n설정 > API 설정에서 입력하세요."
            )
            self._open_settings()
            return

        # 설정 저장
        self._save_settings()

        # 출력 파일 경로
        input_path = Path(self.current_pdf_path)
        output_path = input_path.parent / f"{input_path.stem}_translated.pdf"

        # 저장 위치 선택
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "번역 결과 저장",
            str(output_path),
            "PDF Files (*.pdf)"
        )

        if not save_path:
            return

        # 로그 초기화
        self._clear_log()

        # 페이지 범위
        start = self.start_page_spin.value() - 1  # 0-based
        end = self.end_page_spin.value() - 1

        # 워커 생성 및 시작
        self.worker = TranslationWorker(
            self.current_pdf_path,
            save_path,
            self.config,
            (start, end)
        )

        self.worker.progress.connect(self._on_progress)
        self.worker.log_message.connect(self._on_log_message)
        self.worker.page_completed.connect(self._on_page_completed)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.worker.start()

        # UI 상태 변경
        self.translate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _cancel_translation(self):
        """번역 취소"""
        if self.worker:
            self.worker.cancel()
            self._log("취소 요청됨...")

    def _on_progress(self, value: int, message: str):
        """진행 상태 업데이트"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        self.statusbar.showMessage(message)

    def _on_log_message(self, message: str):
        """로그 메시지 수신"""
        self._log(message)

    def _on_page_completed(self, page_num: int, image: Image.Image):
        """페이지 완료"""
        self.preview_widget.set_image(image)

    def _on_finished(self, success: bool, message: str):
        """번역 완료"""
        self.translate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            QMessageBox.information(self, "완료", message)

        self.statusbar.showMessage("준비됨")

    def _on_error(self, error_msg: str):
        """오류 발생"""
        self._log(f"오류: {error_msg}")
        QMessageBox.critical(self, "오류", error_msg)

    def _log(self, message: str):
        """로그 추가 (타임스탬프 포함)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        self.log_text.append(formatted_msg)

        # 자동 스크롤
        if self.auto_scroll_check.isChecked():
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_text.setTextCursor(cursor)

    def _clear_log(self):
        """로그 지우기"""
        self.log_text.clear()

    def closeEvent(self, event):
        """창 닫기 이벤트"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "확인",
                "번역이 진행 중입니다. 종료하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

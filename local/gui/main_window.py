"""
메인 윈도우
"""
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox,
    QFileDialog, QProgressBar, QTextEdit, QGroupBox,
    QSplitter, QScrollArea, QMessageBox, QStatusBar,
    QAction, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QTextCursor

from PIL import Image

# 경로 설정
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config, LANGUAGES, TESSERACT_LANGS
from core.engine import TranslationEngine
from core.pdf_handler import PDFHandler


class Worker(QThread):
    """번역 작업 스레드"""

    log = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    image = pyqtSignal(int, object)
    finished = pyqtSignal(bool, str)

    def __init__(self, engine: TranslationEngine, input_path: str,
                 output_path: str, page_start: int, page_end: int,
                 ocr_conf: float):
        super().__init__()
        self.engine = engine
        self.input_path = input_path
        self.output_path = output_path
        self.page_start = page_start
        self.page_end = page_end
        self.ocr_conf = ocr_conf

    def run(self):
        # 콜백 연결
        self.engine.set_callbacks(
            log=lambda m: self.log.emit(m),
            progress=lambda p, m: self.progress.emit(p, m),
            image=lambda n, i: self.image.emit(n, i),
        )

        result = self.engine.translate(
            self.input_path,
            self.output_path,
            self.page_start,
            self.page_end,
            self.ocr_conf,
        )

        if result.success:
            self.finished.emit(True, f"완료: {result.output_path}")
        else:
            self.finished.emit(False, result.error or "알 수 없는 오류")


class ImagePreview(QScrollArea):
    """이미지 미리보기"""

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background: #f0f0f0;")
        self.setWidget(self.label)
        self._pixmap = None

    def set_image(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._update()

    def _update(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size() * 0.95, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update()


class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.config = Config.load()
        self.pdf_path: Optional[str] = None
        self.worker: Optional[Worker] = None
        self.engine: Optional[TranslationEngine] = None

        self._init_ui()
        self._load_config()

    def _init_ui(self):
        self.setWindowTitle("PDF Translator")
        self.setMinimumSize(1200, 800)

        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # 좌측 패널
        left = self._create_left_panel()
        left.setMaximumWidth(350)

        # 우측 패널
        right = self._create_right_panel()

        # 스플리터
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 850])
        layout.addWidget(splitter)

        # 상태바
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("준비됨")

        # 메뉴
        self._create_menu()

    def _create_menu(self):
        menu = self.menuBar()

        # 파일
        file_menu = menu.addMenu("파일")
        open_act = QAction("PDF 열기", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_pdf)
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        exit_act = QAction("종료", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # 설정
        settings_menu = menu.addMenu("설정")
        api_act = QAction("API 설정", self)
        api_act.triggered.connect(self._open_settings)
        settings_menu.addAction(api_act)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 파일
        file_grp = QGroupBox("PDF 파일")
        file_lay = QVBoxLayout(file_grp)
        self.file_label = QLabel("파일을 선택하세요")
        self.file_label.setWordWrap(True)
        file_lay.addWidget(self.file_label)
        open_btn = QPushButton("PDF 열기")
        open_btn.clicked.connect(self._open_pdf)
        file_lay.addWidget(open_btn)
        layout.addWidget(file_grp)

        # 언어
        lang_grp = QGroupBox("언어")
        lang_lay = QVBoxLayout(lang_grp)

        src_lay = QHBoxLayout()
        src_lay.addWidget(QLabel("원본:"))
        self.src_combo = QComboBox()
        self.src_combo.addItems(list(LANGUAGES.keys()))
        src_lay.addWidget(self.src_combo)
        lang_lay.addLayout(src_lay)

        tgt_lay = QHBoxLayout()
        tgt_lay.addWidget(QLabel("번역:"))
        self.tgt_combo = QComboBox()
        self.tgt_combo.addItems(list(LANGUAGES.keys()))
        tgt_lay.addWidget(self.tgt_combo)
        lang_lay.addLayout(tgt_lay)

        layout.addWidget(lang_grp)

        # 페이지
        page_grp = QGroupBox("페이지")
        page_lay = QVBoxLayout(page_grp)

        range_lay = QHBoxLayout()
        range_lay.addWidget(QLabel("시작:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(1)
        self.start_spin.setMaximum(9999)
        range_lay.addWidget(self.start_spin)
        range_lay.addWidget(QLabel("끝:"))
        self.end_spin = QSpinBox()
        self.end_spin.setMinimum(1)
        self.end_spin.setMaximum(9999)
        range_lay.addWidget(self.end_spin)
        page_lay.addLayout(range_lay)

        self.page_label = QLabel("총 페이지: -")
        page_lay.addWidget(self.page_label)

        layout.addWidget(page_grp)

        # 고급
        adv_grp = QGroupBox("고급")
        adv_lay = QVBoxLayout(adv_grp)

        dpi_lay = QHBoxLayout()
        dpi_lay.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 300)
        self.dpi_spin.setValue(150)
        dpi_lay.addWidget(self.dpi_spin)
        adv_lay.addLayout(dpi_lay)

        model_lay = QHBoxLayout()
        model_lay.addWidget(QLabel("모델:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        model_lay.addWidget(self.model_combo)
        adv_lay.addLayout(model_lay)

        layout.addWidget(adv_grp)

        # 진행
        prog_grp = QGroupBox("진행")
        prog_lay = QVBoxLayout(prog_grp)
        self.progress_bar = QProgressBar()
        prog_lay.addWidget(self.progress_bar)
        self.progress_label = QLabel("대기 중")
        prog_lay.addWidget(self.progress_label)
        layout.addWidget(prog_grp)

        # 버튼
        btn_lay = QHBoxLayout()
        self.start_btn = QPushButton("번역 시작")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.start_btn.setStyleSheet("""
            QPushButton { background: #4CAF50; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background: #45a049; }
            QPushButton:disabled { background: #ccc; }
        """)
        btn_lay.addWidget(self.start_btn)

        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_lay.addWidget(self.cancel_btn)

        layout.addLayout(btn_lay)
        layout.addStretch()

        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 미리보기
        preview_label = QLabel("미리보기")
        preview_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(preview_label)

        self.preview = ImagePreview()
        layout.addWidget(self.preview)

        # 로그
        log_grp = QGroupBox("실시간 로그")
        log_lay = QVBoxLayout(log_grp)

        ctrl_lay = QHBoxLayout()
        self.auto_scroll = QCheckBox("자동 스크롤")
        self.auto_scroll.setChecked(True)
        ctrl_lay.addWidget(self.auto_scroll)
        ctrl_lay.addStretch()
        clear_btn = QPushButton("지우기")
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        ctrl_lay.addWidget(clear_btn)
        log_lay.addLayout(ctrl_lay)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit { background: #1e1e1e; color: #d4d4d4;
                font-family: Consolas, monospace; font-size: 12px; }
        """)
        log_lay.addWidget(self.log_text)

        layout.addWidget(log_grp)

        return panel

    def _load_config(self):
        c = self.config
        idx = self.src_combo.findText(c.source_lang)
        if idx >= 0: self.src_combo.setCurrentIndex(idx)
        idx = self.tgt_combo.findText(c.target_lang)
        if idx >= 0: self.tgt_combo.setCurrentIndex(idx)
        self.dpi_spin.setValue(c.render_dpi)
        idx = self.model_combo.findText(c.openai_model)
        if idx >= 0: self.model_combo.setCurrentIndex(idx)

    def _save_config(self):
        c = self.config
        c.source_lang = self.src_combo.currentText()
        c.target_lang = self.tgt_combo.currentText()
        c.render_dpi = self.dpi_spin.value()
        c.openai_model = self.model_combo.currentText()
        c.save()

    def _open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "PDF 선택", "", "PDF (*.pdf)")
        if not path:
            return

        self.pdf_path = path
        self.file_label.setText(Path(path).name)

        try:
            with PDFHandler(path) as pdf:
                info = pdf.get_info()
                self.page_label.setText(f"총 페이지: {info.page_count}")
                self.start_spin.setMaximum(info.page_count)
                self.end_spin.setMaximum(info.page_count)
                self.start_spin.setValue(1)
                self.end_spin.setValue(info.page_count)

                # 미리보기
                img = pdf.render_page(0)
                self.preview.set_image(img)

            self.start_btn.setEnabled(True)
            self.statusbar.showMessage(f"로드됨: {info.page_count}페이지")
            self._log(f"PDF 열림: {path}")

        except Exception as e:
            QMessageBox.critical(self, "오류", str(e))

    def _open_settings(self):
        from gui.dialogs import SettingsDialog
        dlg = SettingsDialog(self.config, self)
        if dlg.exec_():
            self._log("설정 저장됨")

    def _start(self):
        if not self.pdf_path:
            QMessageBox.warning(self, "경고", "PDF를 먼저 선택하세요")
            return

        if not self.config.openai_api_key:
            QMessageBox.warning(self, "경고", "API 키를 설정하세요")
            self._open_settings()
            return

        self._save_config()

        # 출력 경로
        inp = Path(self.pdf_path)
        out = inp.parent / f"{inp.stem}_translated.pdf"
        save_path, _ = QFileDialog.getSaveFileName(
            self, "저장", str(out), "PDF (*.pdf)"
        )
        if not save_path:
            return

        self.log_text.clear()

        # OCR 언어
        ocr_lang = TESSERACT_LANGS.get(self.src_combo.currentText(), "eng")

        # 엔진 생성
        self.engine = TranslationEngine(
            api_key=self.config.openai_api_key,
            model=self.model_combo.currentText(),
            source_lang=self.src_combo.currentText(),
            target_lang=self.tgt_combo.currentText(),
            ocr_lang=ocr_lang,
            dpi=self.dpi_spin.value(),
            font_path=self.config.font_path or None,
        )

        # 워커 시작
        self.worker = Worker(
            self.engine,
            self.pdf_path,
            save_path,
            self.start_spin.value() - 1,
            self.end_spin.value() - 1,
            self.config.ocr_confidence,
        )

        self.worker.log.connect(self._log)
        self.worker.progress.connect(self._on_progress)
        self.worker.image.connect(self._on_image)
        self.worker.finished.connect(self._on_finished)

        self.worker.start()

        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _cancel(self):
        if self.engine:
            self.engine.cancel()
            self._log("취소 요청...")

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)
        self.statusbar.showMessage(msg)

    def _on_image(self, page: int, img: Image.Image):
        self.preview.set_image(img)

    def _on_finished(self, success: bool, msg: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if success:
            QMessageBox.information(self, "완료", msg)
        else:
            QMessageBox.critical(self, "오류", msg)

        self.statusbar.showMessage("준비됨")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

        if self.auto_scroll.isChecked():
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.log_text.setTextCursor(cursor)

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "확인", "번역 중입니다. 종료할까요?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                if self.engine:
                    self.engine.cancel()
                self.worker.wait()
                e.accept()
            else:
                e.ignore()
        else:
            e.accept()

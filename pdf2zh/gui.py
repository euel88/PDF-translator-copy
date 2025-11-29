"""
PyQt5 GUI - PDFMathTranslate 구조 기반
"""
import sys
import os
from pathlib import Path
from typing import Optional
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QLineEdit,
    QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QTabWidget, QMessageBox, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QIcon

from pdf2zh.config import config, LANGUAGES
from pdf2zh.high_level import translate


class LogSignal(QObject):
    """스레드 안전한 로그 시그널"""
    message = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)


class TranslatorThread(threading.Thread):
    """번역 작업 스레드"""

    def __init__(self, params: dict, signal: LogSignal):
        super().__init__()
        self.params = params
        self.signal = signal
        self.daemon = True

    def run(self):
        try:
            def callback(msg: str):
                self.signal.message.emit(msg)
                # 진행률 파싱
                if "페이지" in msg and "/" in msg:
                    try:
                        parts = msg.split("/")
                        current = int(parts[0].split()[-1])
                        total = int(parts[1].split()[0])
                        self.signal.progress.emit(current, total)
                    except:
                        pass

            result = translate(
                input_path=self.params["input_path"],
                output_path=self.params["output_path"],
                source_lang=self.params["source_lang"],
                target_lang=self.params["target_lang"],
                service=self.params["service"],
                pages=self.params.get("pages"),
                dpi=self.params.get("dpi", 150),
                callback=callback,
                **self.params.get("kwargs", {})
            )

            if result.success:
                self.signal.finished.emit(True, result.output_path)
            else:
                self.signal.finished.emit(False, result.error or "Unknown error")

        except Exception as e:
            self.signal.finished.emit(False, str(e))


class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.signal = LogSignal()
        self.worker = None
        self.setup_ui()
        self.connect_signals()
        self.load_settings()

    def setup_ui(self):
        self.setWindowTitle("PDF Translator")
        self.setMinimumSize(800, 600)

        # 중앙 위젯
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # 탭 위젯
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # 번역 탭
        translate_tab = self.create_translate_tab()
        tabs.addTab(translate_tab, "번역")

        # 설정 탭
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "설정")

        # 로그 영역
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        # 진행률
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 상태 표시줄
        self.statusBar().showMessage("준비")

    def create_translate_tab(self) -> QWidget:
        """번역 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 파일 선택
        file_group = QGroupBox("파일")
        file_layout = QFormLayout(file_group)

        # 입력 파일
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("PDF 파일을 선택하세요")
        input_btn = QPushButton("찾아보기...")
        input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_path, 1)
        input_layout.addWidget(input_btn)
        file_layout.addRow("입력 파일:", input_layout)

        # 출력 파일
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("자동 생성 (선택사항)")
        output_btn = QPushButton("찾아보기...")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_path, 1)
        output_layout.addWidget(output_btn)
        file_layout.addRow("출력 파일:", output_layout)

        layout.addWidget(file_group)

        # 번역 설정
        trans_group = QGroupBox("번역 설정")
        trans_layout = QFormLayout(trans_group)

        # 언어 선택
        lang_layout = QHBoxLayout()
        self.source_lang = QComboBox()
        self.source_lang.addItems(LANGUAGES.keys())
        self.source_lang.setCurrentText("English")

        arrow_label = QLabel("→")
        arrow_label.setAlignment(Qt.AlignCenter)

        self.target_lang = QComboBox()
        self.target_lang.addItems(LANGUAGES.keys())
        self.target_lang.setCurrentText("Korean")

        lang_layout.addWidget(self.source_lang, 1)
        lang_layout.addWidget(arrow_label)
        lang_layout.addWidget(self.target_lang, 1)
        trans_layout.addRow("언어:", lang_layout)

        # 번역 서비스
        self.service = QComboBox()
        self.service.addItems(["openai", "google", "deepl", "ollama"])
        self.service.currentTextChanged.connect(self.on_service_changed)
        trans_layout.addRow("서비스:", self.service)

        # 페이지 범위
        page_layout = QHBoxLayout()
        self.page_start = QSpinBox()
        self.page_start.setMinimum(0)
        self.page_start.setMaximum(9999)
        self.page_start.setValue(0)
        self.page_start.setSpecialValueText("처음")

        page_to = QLabel("-")

        self.page_end = QSpinBox()
        self.page_end.setMinimum(0)
        self.page_end.setMaximum(9999)
        self.page_end.setValue(0)
        self.page_end.setSpecialValueText("끝")

        page_layout.addWidget(self.page_start)
        page_layout.addWidget(page_to)
        page_layout.addWidget(self.page_end)
        page_layout.addStretch()
        trans_layout.addRow("페이지:", page_layout)

        # DPI
        self.dpi = QSpinBox()
        self.dpi.setMinimum(72)
        self.dpi.setMaximum(600)
        self.dpi.setValue(150)
        trans_layout.addRow("DPI:", self.dpi)

        layout.addWidget(trans_group)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.translate_btn = QPushButton("번역 시작")
        self.translate_btn.setMinimumWidth(150)
        self.translate_btn.setMinimumHeight(40)
        self.translate_btn.clicked.connect(self.start_translation)
        btn_layout.addWidget(self.translate_btn)

        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.setMinimumWidth(100)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_translation)
        btn_layout.addWidget(self.cancel_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        return widget

    def create_settings_tab(self) -> QWidget:
        """설정 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # OpenAI 설정
        openai_group = QGroupBox("OpenAI")
        openai_layout = QFormLayout(openai_group)

        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.Password)
        self.openai_key.setPlaceholderText("sk-...")
        openai_layout.addRow("API Key:", self.openai_key)

        self.openai_model = QComboBox()
        self.openai_model.addItems(["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
        self.openai_model.setEditable(True)
        openai_layout.addRow("Model:", self.openai_model)

        self.openai_base = QLineEdit()
        self.openai_base.setPlaceholderText("https://api.openai.com/v1 (기본값)")
        openai_layout.addRow("Base URL:", self.openai_base)

        layout.addWidget(openai_group)

        # DeepL 설정
        deepl_group = QGroupBox("DeepL")
        deepl_layout = QFormLayout(deepl_group)

        self.deepl_key = QLineEdit()
        self.deepl_key.setEchoMode(QLineEdit.Password)
        deepl_layout.addRow("API Key:", self.deepl_key)

        layout.addWidget(deepl_group)

        # Ollama 설정
        ollama_group = QGroupBox("Ollama")
        ollama_layout = QFormLayout(ollama_group)

        self.ollama_host = QLineEdit()
        self.ollama_host.setPlaceholderText("http://localhost:11434 (기본값)")
        ollama_layout.addRow("Host:", self.ollama_host)

        self.ollama_model = QComboBox()
        self.ollama_model.addItems(["llama3", "llama2", "mistral", "gemma"])
        self.ollama_model.setEditable(True)
        ollama_layout.addRow("Model:", self.ollama_model)

        layout.addWidget(ollama_group)

        # 저장 버튼
        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()
        return widget

    def connect_signals(self):
        """시그널 연결"""
        self.signal.message.connect(self.on_log)
        self.signal.progress.connect(self.on_progress)
        self.signal.finished.connect(self.on_finished)

    def load_settings(self):
        """설정 로드"""
        self.openai_key.setText(config.get("OPENAI_API_KEY", ""))
        self.openai_base.setText(config.get("OPENAI_BASE_URL", ""))
        self.deepl_key.setText(config.get("DEEPL_API_KEY", ""))
        self.ollama_host.setText(config.get("OLLAMA_HOST", ""))

    def save_settings(self):
        """설정 저장"""
        config.set("OPENAI_API_KEY", self.openai_key.text())
        config.set("OPENAI_BASE_URL", self.openai_base.text())
        config.set("DEEPL_API_KEY", self.deepl_key.text())
        config.set("OLLAMA_HOST", self.ollama_host.text())
        config.save()
        self.log("설정이 저장되었습니다.")
        QMessageBox.information(self, "설정", "설정이 저장되었습니다.")

    def browse_input(self):
        """입력 파일 선택"""
        path, _ = QFileDialog.getOpenFileName(
            self, "PDF 파일 선택", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if path:
            self.input_path.setText(path)
            # 자동 출력 경로 생성
            if not self.output_path.text():
                inp = Path(path)
                self.output_path.setText(str(inp.parent / f"{inp.stem}_translated.pdf"))

    def browse_output(self):
        """출력 파일 선택"""
        path, _ = QFileDialog.getSaveFileName(
            self, "출력 파일 저장", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if path:
            self.output_path.setText(path)

    def on_service_changed(self, service: str):
        """서비스 변경 시"""
        pass

    def start_translation(self):
        """번역 시작"""
        input_path = self.input_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "오류", "입력 파일을 선택하세요.")
            return

        if not os.path.exists(input_path):
            QMessageBox.warning(self, "오류", "파일을 찾을 수 없습니다.")
            return

        # 페이지 범위
        pages = None
        start = self.page_start.value()
        end = self.page_end.value()
        if start > 0 or end > 0:
            if end == 0:
                end = 9999
            pages = list(range(start - 1 if start > 0 else 0, end))

        # 서비스별 추가 옵션
        kwargs = {}
        service = self.service.currentText()

        if service == "openai":
            if self.openai_key.text():
                kwargs["api_key"] = self.openai_key.text()
            kwargs["model"] = self.openai_model.currentText()
            if self.openai_base.text():
                kwargs["base_url"] = self.openai_base.text()
        elif service == "deepl":
            if self.deepl_key.text():
                kwargs["api_key"] = self.deepl_key.text()
        elif service == "ollama":
            if self.ollama_host.text():
                kwargs["host"] = self.ollama_host.text()
            kwargs["model"] = self.ollama_model.currentText()

        params = {
            "input_path": input_path,
            "output_path": self.output_path.text() or None,
            "source_lang": self.source_lang.currentText(),
            "target_lang": self.target_lang.currentText(),
            "service": service,
            "pages": pages,
            "dpi": self.dpi.value(),
            "kwargs": kwargs,
        }

        # UI 상태 변경
        self.translate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.statusBar().showMessage("번역 중...")

        self.log(f"번역 시작: {input_path}")
        self.log(f"서비스: {service}, {self.source_lang.currentText()} → {self.target_lang.currentText()}")

        # 스레드 시작
        self.worker = TranslatorThread(params, self.signal)
        self.worker.start()

    def cancel_translation(self):
        """번역 취소"""
        self.log("취소 요청됨...")
        self.cancel_btn.setEnabled(False)
        # 스레드는 daemon이므로 앱 종료 시 자동 종료

    def on_log(self, message: str):
        """로그 메시지"""
        self.log_text.append(message)
        # 스크롤 아래로
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_progress(self, current: int, total: int):
        """진행률 업데이트"""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)

    def on_finished(self, success: bool, message: str):
        """완료"""
        self.translate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

        if success:
            self.statusBar().showMessage("완료")
            self.log(f"번역 완료: {message}")
            QMessageBox.information(self, "완료", f"번역이 완료되었습니다.\n\n{message}")
        else:
            self.statusBar().showMessage("오류 발생")
            self.log(f"오류: {message}")
            QMessageBox.critical(self, "오류", f"번역 중 오류가 발생했습니다.\n\n{message}")

    def log(self, message: str):
        """로그 추가"""
        self.log_text.append(message)


def run_gui():
    """GUI 실행"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()

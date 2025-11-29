"""
설정 다이얼로그
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog,
    QGroupBox, QDoubleSpinBox, QMessageBox,
)
from PyQt5.QtCore import Qt

from ..utils.config import Config


class SettingsDialog(QDialog):
    """설정 다이얼로그"""

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        self.setWindowTitle("설정")
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)

        # API 설정
        api_grp = QGroupBox("OpenAI API")
        api_lay = QFormLayout(api_grp)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-...")
        api_lay.addRow("API Key:", self.api_key_edit)

        # API Key 보기 토글
        show_key_btn = QPushButton("보기")
        show_key_btn.setCheckable(True)
        show_key_btn.clicked.connect(self._toggle_api_key_visibility)
        api_lay.addRow("", show_key_btn)

        layout.addWidget(api_grp)

        # OCR 설정
        ocr_grp = QGroupBox("OCR 설정")
        ocr_lay = QFormLayout(ocr_grp)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.1)
        self.conf_spin.setDecimals(2)
        ocr_lay.addRow("신뢰도 임계값:", self.conf_spin)

        layout.addWidget(ocr_grp)

        # 폰트 설정
        font_grp = QGroupBox("폰트 설정")
        font_lay = QVBoxLayout(font_grp)

        font_row = QHBoxLayout()
        self.font_edit = QLineEdit()
        self.font_edit.setPlaceholderText("기본 폰트 사용 (비워두면 자동)")
        font_row.addWidget(self.font_edit)
        font_btn = QPushButton("찾아보기")
        font_btn.clicked.connect(self._browse_font)
        font_row.addWidget(font_btn)
        font_lay.addLayout(font_row)

        layout.addWidget(font_grp)

        # Tesseract 상태
        tess_grp = QGroupBox("Tesseract OCR 상태")
        tess_lay = QVBoxLayout(tess_grp)
        self.tess_status = QLabel()
        self._check_tesseract()
        tess_lay.addWidget(self.tess_status)
        layout.addWidget(tess_grp)

        # 버튼
        btn_lay = QHBoxLayout()
        btn_lay.addStretch()

        save_btn = QPushButton("저장")
        save_btn.clicked.connect(self._save)
        btn_lay.addWidget(save_btn)

        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        btn_lay.addWidget(cancel_btn)

        layout.addLayout(btn_lay)

    def _load_settings(self):
        """설정 로드"""
        self.api_key_edit.setText(self.config.openai_api_key)
        self.conf_spin.setValue(self.config.ocr_confidence)
        self.font_edit.setText(self.config.font_path)

    def _toggle_api_key_visibility(self, checked: bool):
        """API 키 표시/숨김"""
        if checked:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)

    def _browse_font(self):
        """폰트 파일 선택"""
        path, _ = QFileDialog.getOpenFileName(
            self, "폰트 선택", "",
            "Font Files (*.ttf *.otf *.ttc);;All Files (*)"
        )
        if path:
            self.font_edit.setText(path)

    def _check_tesseract(self):
        """Tesseract 상태 확인"""
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            langs = pytesseract.get_languages()
            self.tess_status.setText(
                f"<font color='green'>설치됨</font><br>"
                f"버전: {version}<br>"
                f"언어: {', '.join(langs[:5])}{'...' if len(langs) > 5 else ''}"
            )
        except Exception as e:
            self.tess_status.setText(
                f"<font color='red'>미설치 또는 오류</font><br>"
                f"Tesseract OCR을 설치해주세요.<br>"
                f"Windows: https://github.com/UB-Mannheim/tesseract/wiki<br>"
                f"Mac: brew install tesseract<br>"
                f"Linux: sudo apt install tesseract-ocr"
            )

    def _save(self):
        """설정 저장"""
        api_key = self.api_key_edit.text().strip()

        if not api_key:
            QMessageBox.warning(self, "경고", "API 키를 입력해주세요.")
            return

        if not api_key.startswith("sk-"):
            reply = QMessageBox.question(
                self, "확인",
                "API 키가 'sk-'로 시작하지 않습니다. 계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self.config.openai_api_key = api_key
        self.config.ocr_confidence = self.conf_spin.value()
        self.config.font_path = self.font_edit.text().strip()
        self.config.save()

        self.accept()

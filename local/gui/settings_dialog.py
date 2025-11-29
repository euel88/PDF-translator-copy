"""
설정 다이얼로그
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QGroupBox,
    QComboBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QWidget, QMessageBox
)
from PyQt5.QtCore import Qt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigManager


class SettingsDialog(QDialog):
    """설정 다이얼로그"""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self.config = config
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("설정")
        self.setMinimumWidth(500)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # 탭 위젯
        tab_widget = QTabWidget()

        # API 탭
        api_tab = self._create_api_tab()
        tab_widget.addTab(api_tab, "API 설정")

        # OCR 탭
        ocr_tab = self._create_ocr_tab()
        tab_widget.addTab(ocr_tab, "OCR 설정")

        # 출력 탭
        output_tab = self._create_output_tab()
        tab_widget.addTab(output_tab, "출력 설정")

        layout.addWidget(tab_widget)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        save_btn = QPushButton("저장")
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    def _create_api_tab(self) -> QWidget:
        """API 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # OpenAI 설정
        openai_group = QGroupBox("OpenAI")
        openai_layout = QFormLayout(openai_group)

        # API 키
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-...")

        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(self.api_key_edit)

        show_btn = QPushButton("표시")
        show_btn.setCheckable(True)
        show_btn.toggled.connect(self._toggle_api_key_visibility)
        api_key_layout.addWidget(show_btn)

        openai_layout.addRow("API 키:", api_key_layout)

        # 모델 선택
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ])
        openai_layout.addRow("모델:", self.model_combo)

        layout.addWidget(openai_group)

        # API 테스트 버튼
        test_btn = QPushButton("API 연결 테스트")
        test_btn.clicked.connect(self._test_api)
        layout.addWidget(test_btn)

        layout.addStretch()

        return tab

    def _create_ocr_tab(self) -> QWidget:
        """OCR 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # OCR 설정
        ocr_group = QGroupBox("PaddleOCR")
        ocr_layout = QFormLayout(ocr_group)

        # 신뢰도 임계값
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setDecimals(2)
        ocr_layout.addRow("신뢰도 임계값:", self.confidence_spin)

        # 설명
        desc_label = QLabel(
            "신뢰도가 이 값 이상인 텍스트만 번역됩니다.\n"
            "낮추면 더 많은 텍스트를 감지하지만 오류가 증가할 수 있습니다."
        )
        desc_label.setStyleSheet("color: gray;")
        desc_label.setWordWrap(True)
        ocr_layout.addRow(desc_label)

        layout.addWidget(ocr_group)

        layout.addStretch()

        return tab

    def _create_output_tab(self) -> QWidget:
        """출력 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 출력 설정
        output_group = QGroupBox("출력")
        output_layout = QFormLayout(output_group)

        # 기본 출력 디렉토리
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("기본값: 원본 파일과 같은 폴더")

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)

        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self._browse_output_dir)
        output_dir_layout.addWidget(browse_btn)

        output_layout.addRow("출력 폴더:", output_dir_layout)

        layout.addWidget(output_group)

        # 폰트 설정
        font_group = QGroupBox("폰트")
        font_layout = QFormLayout(font_group)

        self.font_path_edit = QLineEdit()
        self.font_path_edit.setPlaceholderText("시스템 기본 폰트 사용")

        font_path_layout = QHBoxLayout()
        font_path_layout.addWidget(self.font_path_edit)

        font_browse_btn = QPushButton("찾아보기")
        font_browse_btn.clicked.connect(self._browse_font)
        font_path_layout.addWidget(font_browse_btn)

        font_layout.addRow("폰트 파일:", font_path_layout)

        font_desc = QLabel(
            "번역된 텍스트에 사용할 폰트입니다.\n"
            "한글 지원 폰트를 선택하세요 (예: NanumGothic, Malgun Gothic)"
        )
        font_desc.setStyleSheet("color: gray;")
        font_desc.setWordWrap(True)
        font_layout.addRow(font_desc)

        layout.addWidget(font_group)

        layout.addStretch()

        return tab

    def _toggle_api_key_visibility(self, visible: bool):
        """API 키 표시/숨기기"""
        if visible:
            self.api_key_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.api_key_edit.setEchoMode(QLineEdit.Password)

    def _browse_output_dir(self):
        """출력 디렉토리 선택"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "출력 폴더 선택"
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _browse_font(self):
        """폰트 파일 선택"""
        font_path, _ = QFileDialog.getOpenFileName(
            self,
            "폰트 파일 선택",
            "",
            "Font Files (*.ttf *.ttc *.otf)"
        )
        if font_path:
            self.font_path_edit.setText(font_path)

    def _test_api(self):
        """API 연결 테스트"""
        api_key = self.api_key_edit.text().strip()

        if not api_key:
            QMessageBox.warning(self, "경고", "API 키를 입력하세요")
            return

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            # 간단한 테스트 요청
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )

            QMessageBox.information(
                self,
                "성공",
                "API 연결이 정상입니다!"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                f"API 연결 실패:\n{str(e)}"
            )

    def _load_settings(self):
        """설정 로드"""
        config = self.config.config

        # API
        self.api_key_edit.setText(config.openai_api_key)
        idx = self.model_combo.findText(config.openai_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        # OCR
        self.confidence_spin.setValue(config.ocr_confidence_threshold)

        # 출력
        self.output_dir_edit.setText(config.output_dir)
        self.font_path_edit.setText(config.font_path)

    def _save_settings(self):
        """설정 저장"""
        config = self.config.config

        # API
        config.openai_api_key = self.api_key_edit.text().strip()
        config.openai_model = self.model_combo.currentText()

        # OCR
        config.ocr_confidence_threshold = self.confidence_spin.value()

        # 출력
        config.output_dir = self.output_dir_edit.text().strip()
        config.font_path = self.font_path_edit.text().strip()

        # 저장
        self.config.save_config()

        QMessageBox.information(self, "저장", "설정이 저장되었습니다")
        self.accept()

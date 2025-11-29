"""
폰트 관리 모듈 - Go Noto Universal 및 다국어 폰트 지원
PDFMathTranslate 수준의 완벽한 다국어 렌더링
"""
import os
import sys
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import threading


@dataclass
class FontInfo:
    """폰트 정보"""
    name: str
    path: str
    languages: List[str]
    weight: str = "Regular"
    style: str = "Normal"


# Go Noto Universal 폰트 다운로드 정보
GO_NOTO_FONTS = {
    "GoNotoKurrent-Regular": {
        "url": "https://github.com/nicovank/Go-Noto-Universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf",
        "sha256": None,  # 실제 해시 필요
        "languages": ["all"],
    },
    "GoNotoKurrent-Bold": {
        "url": "https://github.com/nicovank/Go-Noto-Universal/releases/download/v7.0/GoNotoKurrent-Bold.ttf",
        "sha256": None,
        "languages": ["all"],
    },
}

# Noto Sans CJK 폰트 (한국어, 일본어, 중국어용 백업)
NOTO_CJK_FONTS = {
    "NotoSansCJK-Regular": {
        "url": "https://github.com/googlefonts/noto-cjk/releases/download/Sans2.004/03_NotoSansCJK-OTC.zip",
        "file_in_zip": "OTC/NotoSansCJK-Regular.ttc",
        "languages": ["ko", "ja", "zh", "zh-cn", "zh-tw"],
    },
}

# 언어별 권장 폰트 순서
LANGUAGE_FONT_PRIORITY = {
    # 동아시아
    "ko": ["GoNotoKurrent-Regular", "NotoSansCJK-Regular", "Malgun Gothic", "NanumGothic"],
    "ja": ["GoNotoKurrent-Regular", "NotoSansCJK-Regular", "Meiryo", "Yu Gothic"],
    "zh": ["GoNotoKurrent-Regular", "NotoSansCJK-Regular", "Microsoft YaHei", "SimHei"],
    "zh-cn": ["GoNotoKurrent-Regular", "NotoSansCJK-Regular", "Microsoft YaHei", "SimHei"],
    "zh-tw": ["GoNotoKurrent-Regular", "NotoSansCJK-Regular", "Microsoft JhengHei", "PMingLiU"],
    # 유럽
    "en": ["GoNotoKurrent-Regular", "DejaVu Sans", "Arial", "Times New Roman"],
    "de": ["GoNotoKurrent-Regular", "DejaVu Sans", "Arial"],
    "fr": ["GoNotoKurrent-Regular", "DejaVu Sans", "Arial"],
    "es": ["GoNotoKurrent-Regular", "DejaVu Sans", "Arial"],
    "ru": ["GoNotoKurrent-Regular", "DejaVu Sans", "Arial"],
    # 아랍/히브리
    "ar": ["GoNotoKurrent-Regular", "Noto Sans Arabic", "Arial"],
    "he": ["GoNotoKurrent-Regular", "Noto Sans Hebrew", "Arial"],
    # 동남아시아
    "th": ["GoNotoKurrent-Regular", "Noto Sans Thai", "Tahoma"],
    "vi": ["GoNotoKurrent-Regular", "DejaVu Sans", "Arial"],
}

# 시스템 폰트 경로
SYSTEM_FONT_PATHS = {
    "windows": [
        r"C:\Windows\Fonts",
    ],
    "linux": [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        str(Path.home() / ".fonts"),
        str(Path.home() / ".local/share/fonts"),
    ],
    "darwin": [
        "/System/Library/Fonts",
        "/Library/Fonts",
        str(Path.home() / "Library/Fonts"),
    ],
}

# 폰트 이름과 파일 매핑
FONT_FILE_NAMES = {
    # Windows
    "Malgun Gothic": ["malgun.ttf", "malgunsl.ttf", "malgunbd.ttf"],
    "NanumGothic": ["NanumGothic.ttf", "NanumGothicBold.ttf"],
    "Meiryo": ["meiryo.ttc", "meiryob.ttc"],
    "Microsoft YaHei": ["msyh.ttc", "msyhbd.ttc"],
    "Microsoft JhengHei": ["msjh.ttc", "msjhbd.ttc"],
    "SimHei": ["simhei.ttf"],
    "Yu Gothic": ["YuGothM.ttc", "YuGothB.ttc"],
    "PMingLiU": ["mingliu.ttc"],
    # Cross-platform
    "DejaVu Sans": ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"],
    "Arial": ["arial.ttf", "arialbd.ttf"],
    "Times New Roman": ["times.ttf", "timesbd.ttf"],
    "Tahoma": ["tahoma.ttf", "tahomabd.ttf"],
    # macOS
    "Apple SD Gothic Neo": ["AppleSDGothicNeo.ttc"],
    "Hiragino Sans": ["HiraginoSans-W3.ttc", "HiraginoSans-W6.ttc"],
}


class FontManager:
    """폰트 관리자 - 싱글톤"""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._cache_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._font_cache: Dict[str, FontInfo] = {}
        self._system_fonts: Dict[str, str] = {}

        # 시스템 폰트 스캔
        self._scan_system_fonts()

    def _scan_system_fonts(self):
        """시스템 폰트 스캔"""
        platform = self._get_platform()
        paths = SYSTEM_FONT_PATHS.get(platform, [])

        for base_path in paths:
            base = Path(base_path)
            if not base.exists():
                continue

            for font_file in base.rglob("*"):
                if font_file.suffix.lower() in (".ttf", ".ttc", ".otf"):
                    name = font_file.stem
                    self._system_fonts[name.lower()] = str(font_file)

    def _get_platform(self) -> str:
        """플랫폼 감지"""
        if sys.platform == "win32":
            return "windows"
        elif sys.platform == "darwin":
            return "darwin"
        else:
            return "linux"

    def get_font_path(self, language: str, weight: str = "Regular") -> Optional[str]:
        """
        언어에 맞는 최적의 폰트 경로 반환

        Args:
            language: 대상 언어 코드 (ko, ja, zh, en, etc.)
            weight: 폰트 굵기 (Regular, Bold)

        Returns:
            폰트 파일 경로 또는 None
        """
        lang = language.lower()

        # Go Noto Universal 폰트 확인 (가장 권장)
        go_noto_path = self._get_go_noto_font(weight)
        if go_noto_path:
            return go_noto_path

        # 언어별 우선순위 폰트 확인
        priority_fonts = LANGUAGE_FONT_PRIORITY.get(lang, LANGUAGE_FONT_PRIORITY["en"])

        for font_name in priority_fonts:
            path = self._find_font_by_name(font_name)
            if path:
                return path

        # 일반 시스템 폰트 폴백
        return self._get_fallback_font()

    def _get_go_noto_font(self, weight: str = "Regular") -> Optional[str]:
        """Go Noto Universal 폰트 경로 (다운로드 포함)"""
        font_name = f"GoNotoKurrent-{weight}"
        if font_name not in GO_NOTO_FONTS:
            font_name = "GoNotoKurrent-Regular"

        font_path = self._cache_dir / f"{font_name}.ttf"

        if font_path.exists():
            return str(font_path)

        # 다운로드 시도
        try:
            font_info = GO_NOTO_FONTS[font_name]
            return self._download_font(font_info["url"], font_path)
        except Exception:
            return None

    def _download_font(self, url: str, target_path: Path) -> Optional[str]:
        """폰트 다운로드"""
        try:
            print(f"폰트 다운로드 중: {url}")
            urllib.request.urlretrieve(url, str(target_path))
            print(f"폰트 저장됨: {target_path}")
            return str(target_path)
        except Exception as e:
            print(f"폰트 다운로드 실패: {e}")
            return None

    def _find_font_by_name(self, font_name: str) -> Optional[str]:
        """폰트 이름으로 경로 찾기"""
        # 캐시된 Go Noto 폰트 확인
        cached = self._cache_dir / f"{font_name}.ttf"
        if cached.exists():
            return str(cached)

        # 시스템 폰트 검색
        name_lower = font_name.lower()
        if name_lower in self._system_fonts:
            return self._system_fonts[name_lower]

        # 알려진 파일명으로 검색
        file_names = FONT_FILE_NAMES.get(font_name, [])
        platform = self._get_platform()
        paths = SYSTEM_FONT_PATHS.get(platform, [])

        for base_path in paths:
            base = Path(base_path)
            if not base.exists():
                continue

            for file_name in file_names:
                for font_file in base.rglob(file_name):
                    if font_file.exists():
                        return str(font_file)

        return None

    def _get_fallback_font(self) -> Optional[str]:
        """범용 폴백 폰트"""
        fallbacks = [
            "DejaVuSans.ttf",
            "arial.ttf",
            "FreeSans.ttf",
        ]

        platform = self._get_platform()
        paths = SYSTEM_FONT_PATHS.get(platform, [])

        for base_path in paths:
            base = Path(base_path)
            if not base.exists():
                continue

            for font_file in fallbacks:
                for found in base.rglob(font_file):
                    if found.exists():
                        return str(found)

        return None

    def ensure_fonts_available(self, languages: List[str]) -> Dict[str, str]:
        """
        지정된 언어들의 폰트가 사용 가능한지 확인하고
        필요시 다운로드

        Args:
            languages: 언어 코드 목록

        Returns:
            언어 -> 폰트 경로 매핑
        """
        result = {}

        for lang in languages:
            path = self.get_font_path(lang)
            if path:
                result[lang] = path

        return result

    def get_font_for_pymupdf(self, language: str) -> Tuple[str, Optional[str]]:
        """
        PyMuPDF용 폰트 정보 반환

        Args:
            language: 대상 언어 코드

        Returns:
            (fontname, fontfile) 튜플
            - fontname: PyMuPDF 내장 폰트 이름 (폰트 파일이 없을 경우)
            - fontfile: 외부 폰트 파일 경로 (있을 경우)
        """
        font_path = self.get_font_path(language)

        if font_path and Path(font_path).exists():
            return ("custom-font", font_path)

        # PyMuPDF 내장 CJK 폰트 폴백
        lang = language.lower()
        if lang in ("ko", "korean"):
            return ("korea1", None)
        elif lang in ("ja", "jp", "japanese"):
            return ("japan1", None)
        elif lang in ("zh", "zh-cn", "chinese"):
            return ("china-s", None)
        elif lang in ("zh-tw", "chinese-traditional"):
            return ("china-t", None)
        else:
            return ("helv", None)

    def get_font_for_pil(self, language: str, size: int = 12) -> 'ImageFont.FreeTypeFont':
        """
        PIL/Pillow용 폰트 객체 반환

        Args:
            language: 대상 언어 코드
            size: 폰트 크기

        Returns:
            PIL ImageFont 객체
        """
        from PIL import ImageFont

        font_path = self.get_font_path(language)

        if font_path and Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass

        # 폴백
        return ImageFont.load_default()

    def list_available_fonts(self) -> List[Dict[str, str]]:
        """사용 가능한 폰트 목록"""
        fonts = []

        # 캐시된 폰트
        for font_file in self._cache_dir.glob("*.ttf"):
            fonts.append({
                "name": font_file.stem,
                "path": str(font_file),
                "source": "cached"
            })

        # 시스템 폰트 (일부만)
        for name, path in list(self._system_fonts.items())[:50]:
            fonts.append({
                "name": name,
                "path": path,
                "source": "system"
            })

        return fonts

    def clear_cache(self):
        """폰트 캐시 삭제"""
        import shutil
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)


# 전역 폰트 관리자 인스턴스
font_manager = FontManager()


def get_font_path(language: str, weight: str = "Regular") -> Optional[str]:
    """
    언어에 맞는 폰트 경로 반환 (편의 함수)
    """
    return font_manager.get_font_path(language, weight)


def ensure_fonts(languages: List[str]) -> Dict[str, str]:
    """
    필요한 폰트 확보 (편의 함수)
    """
    return font_manager.ensure_fonts_available(languages)


def download_go_noto_fonts() -> bool:
    """
    Go Noto Universal 폰트 다운로드

    Returns:
        성공 여부
    """
    success = True

    for font_name, info in GO_NOTO_FONTS.items():
        path = font_manager._get_go_noto_font(
            "Bold" if "Bold" in font_name else "Regular"
        )
        if not path:
            success = False

    return success

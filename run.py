#!/usr/bin/env python3
"""
PDFMathTranslate - Local Runner Script
Run: python run.py

This script handles dependency installation and launches the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


# Required packages
REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "pdf2zh": "pdf2zh>=1.9.0",
    "fitz": "pymupdf",
    "PIL": "Pillow",
    "requests": "requests",
}

OPTIONAL_PACKAGES = {
    "openai": "openai",
    "onnxruntime": "onnxruntime",
}


def print_banner():
    """Print application banner."""
    print("=" * 60)
    print("  PDF Math Translator")
    print("  Based on PDFMathTranslate")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"[ERROR] Python 3.10+ required. Found: {version.major}.{version.minor}")
        print("Please upgrade your Python installation.")
        sys.exit(1)
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")


def check_package(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def install_package(pip_name: str) -> bool:
    """Install a package using pip."""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def install_dependencies(include_optional: bool = False):
    """Install required and optional dependencies."""
    print("[INFO] Checking dependencies...")

    # Install required packages
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if check_package(import_name):
            print(f"  [OK] {import_name}")
        else:
            print(f"  [INSTALLING] {pip_name}...")
            if install_package(pip_name):
                print(f"  [OK] {import_name} installed")
            else:
                print(f"  [ERROR] Failed to install {pip_name}")
                print(f"         Try: pip install {pip_name}")

    # Install optional packages
    if include_optional:
        print("\n[INFO] Installing optional packages...")
        for import_name, pip_name in OPTIONAL_PACKAGES.items():
            if check_package(import_name):
                print(f"  [OK] {import_name}")
            else:
                print(f"  [INSTALLING] {pip_name}...")
                install_package(pip_name)

    print()


def download_fonts():
    """Download required fonts for PDF generation."""
    print("[INFO] Checking fonts...")

    font_dir = Path.home() / ".cache" / "pdf2zh" / "fonts"
    font_dir.mkdir(parents=True, exist_ok=True)

    fonts = {
        "GoNotoKurrent-Regular.ttf": "https://github.com/satbyy/go-noto-universal/releases/download/v7.0/GoNotoKurrent-Regular.ttf",
        "NanumGothic.ttf": "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
    }

    for font_name, url in fonts.items():
        font_path = font_dir / font_name
        if font_path.exists():
            print(f"  [OK] {font_name}")
        else:
            print(f"  [DOWNLOADING] {font_name}...")
            try:
                import requests
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    font_path.write_bytes(response.content)
                    print(f"  [OK] {font_name} downloaded")
                else:
                    print(f"  [WARN] Failed to download {font_name}")
            except Exception as e:
                print(f"  [WARN] Font download error: {e}")

    print()


def check_ollama():
    """Check if Ollama is running (for local LLM support)."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print(f"[INFO] Ollama detected with {len(models)} model(s)")
                return True
    except Exception:
        pass
    return False


def run_streamlit(port: int = 8501, host: str = "localhost"):
    """Run the Streamlit application."""
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print("[ERROR] app.py not found!")
        sys.exit(1)

    print("[INFO] Starting Streamlit server...")
    print(f"[INFO] Open http://{host}:{port} in your browser")
    print("[INFO] Press Ctrl+C to stop")
    print("=" * 60)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            f"--server.port={port}",
            f"--server.address={host}",
            "--server.headless=false",
            "--browser.gatherUsageStats=false",
        ])
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


def run_cli(input_file: str, **kwargs):
    """Run translation from command line."""
    print(f"[INFO] Translating: {input_file}")

    try:
        from pdf2zh import translate

        result = translate(
            files=[input_file],
            lang_in=kwargs.get("source", "en"),
            lang_out=kwargs.get("target", "ko"),
            service=kwargs.get("service", "google"),
            thread=kwargs.get("threads", 4),
        )

        print("[OK] Translation complete!")
        print(f"     Output files in current directory")

    except ImportError:
        print("[ERROR] pdf2zh not installed. Run: pip install pdf2zh")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PDFMathTranslate - PDF Translation with Math Preservation"
    )

    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="Install/update all dependencies"
    )

    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501)"
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="Streamlit server host (default: localhost)"
    )

    parser.add_argument(
        "--cli",
        metavar="FILE",
        help="Run CLI translation on a PDF file"
    )

    parser.add_argument(
        "--source", "-s",
        default="en",
        help="Source language (default: en)"
    )

    parser.add_argument(
        "--target", "-t",
        default="ko",
        help="Target language (default: ko)"
    )

    parser.add_argument(
        "--service",
        default="google",
        choices=["google", "openai", "deepl", "ollama", "deepseek", "gemini"],
        help="Translation service (default: google)"
    )

    args = parser.parse_args()

    print_banner()
    check_python_version()

    # Install mode
    if args.install:
        install_dependencies(include_optional=True)
        download_fonts()
        print("[OK] Installation complete!")
        return

    # Always check basic dependencies
    install_dependencies()

    # Check Ollama
    check_ollama()

    # CLI mode
    if args.cli:
        run_cli(
            args.cli,
            source=args.source,
            target=args.target,
            service=args.service,
        )
        return

    # Default: run Streamlit
    download_fonts()
    run_streamlit(port=args.port, host=args.host)


if __name__ == "__main__":
    main()

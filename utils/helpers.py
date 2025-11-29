"""
Helper utilities for PDFMathTranslate.
"""
import os
import re
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import uuid


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_temp_filepath(filename: str, temp_dir: Optional[str] = None) -> Path:
    """Generate a unique temporary file path."""
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Generate unique filename
    base_name = Path(filename).stem
    extension = Path(filename).suffix
    unique_id = str(uuid.uuid4())[:8]

    return Path(temp_dir) / f"{base_name}_{unique_id}{extension}"


def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """Clean up old temporary files. Returns number of files deleted."""
    import time

    if not os.path.exists(temp_dir):
        return 0

    deleted_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(temp_dir):
        filepath = os.path.join(temp_dir, filename)
        try:
            file_age = current_time - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                deleted_count += 1
        except (OSError, IOError):
            continue

    return deleted_count


def parse_page_range(page_string: str, max_pages: int) -> List[int]:
    """
    Parse a page range string into a list of page numbers.

    Examples:
        "1-5" -> [1, 2, 3, 4, 5]
        "1,3,5" -> [1, 3, 5]
        "1-3,7,10-12" -> [1, 2, 3, 7, 10, 11, 12]
        "" or None -> all pages
    """
    if not page_string or page_string.strip() == "":
        return list(range(1, max_pages + 1))

    pages = set()
    parts = page_string.replace(" ", "").split(",")

    for part in parts:
        if "-" in part:
            try:
                start, end = part.split("-")
                start = int(start)
                end = int(end)
                # Clamp to valid range
                start = max(1, min(start, max_pages))
                end = max(1, min(end, max_pages))
                pages.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                page = int(part)
                if 1 <= page <= max_pages:
                    pages.add(page)
            except ValueError:
                continue

    return sorted(pages)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters."""
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, "_", filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")

    # Limit length
    max_length = 200
    if len(sanitized) > max_length:
        base, ext = os.path.splitext(sanitized)
        base = base[:max_length - len(ext)]
        sanitized = base + ext

    return sanitized or "untitled"


def get_output_filename(
    input_filename: str,
    target_lang: str,
    output_format: str = "dual"
) -> str:
    """Generate output filename based on input and settings."""
    base_name = Path(input_filename).stem
    extension = Path(input_filename).suffix or ".pdf"

    suffix = f"_{target_lang}"
    if output_format == "dual":
        suffix += "_dual"

    return f"{base_name}{suffix}{extension}"


def validate_pdf(file_path: Path) -> Tuple[bool, str]:
    """Validate that a file is a valid PDF."""
    if not file_path.exists():
        return False, "File does not exist"

    if not file_path.suffix.lower() == ".pdf":
        return False, "File is not a PDF"

    # Check PDF magic bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            if header != b"%PDF-":
                return False, "File does not appear to be a valid PDF"
    except IOError as e:
        return False, f"Error reading file: {e}"

    return True, "Valid PDF"


def estimate_translation_time(page_count: int, service: str) -> str:
    """Estimate translation time based on page count and service."""
    # Rough estimates per page (in seconds)
    time_per_page = {
        "google": 2,
        "deepl": 3,
        "openai": 5,
        "ollama": 8,
        "azure": 3,
        "deepseek": 5,
        "gemini": 4,
    }

    seconds_per_page = time_per_page.get(service, 5)
    total_seconds = page_count * seconds_per_page

    if total_seconds < 60:
        return f"~{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        return f"~{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        hours = total_seconds // 3600
        return f"~{hours} hour{'s' if hours > 1 else ''}"

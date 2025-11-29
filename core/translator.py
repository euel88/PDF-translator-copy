"""
PDF Translation module wrapping pdf2zh.
"""
import os
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Generator, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TranslationStatus(Enum):
    """Translation job status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TranslationProgress:
    """Progress information for translation."""
    status: TranslationStatus = TranslationStatus.PENDING
    current_page: int = 0
    total_pages: int = 0
    progress_percent: float = 0.0
    message: str = ""
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status in (TranslationStatus.COMPLETED, TranslationStatus.FAILED, TranslationStatus.CANCELLED)


@dataclass
class TranslationResult:
    """Result of a translation job."""
    success: bool
    mono_path: Optional[str] = None  # Translation only
    dual_path: Optional[str] = None  # Bilingual (original + translation)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFTranslator:
    """
    PDF Translator using pdf2zh library.

    This class provides a clean interface to the pdf2zh translation functionality,
    with support for progress tracking and multiple output formats.
    """

    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        os.makedirs(self.temp_dir, exist_ok=True)
        self._cancel_flag = threading.Event()

    def translate(
        self,
        input_path: str,
        source_lang: str = "en",
        target_lang: str = "ko",
        service: str = "google",
        output_format: str = "dual",
        pages: Optional[str] = None,
        thread_count: int = 4,
        progress_callback: Optional[Callable[[TranslationProgress], None]] = None,
        **service_params
    ) -> TranslationResult:
        """
        Translate a PDF file.

        Args:
            input_path: Path to the input PDF file
            source_lang: Source language code
            target_lang: Target language code
            service: Translation service to use
            output_format: Output format ("mono", "dual", or "both")
            pages: Page range string (e.g., "1-5,8,10-12")
            thread_count: Number of translation threads
            progress_callback: Optional callback for progress updates
            **service_params: Additional parameters for the translation service

        Returns:
            TranslationResult with paths to output files
        """
        self._cancel_flag.clear()

        # Update progress
        if progress_callback:
            progress_callback(TranslationProgress(
                status=TranslationStatus.IN_PROGRESS,
                message="Initializing translation..."
            ))

        try:
            # Import pdf2zh
            from pdf2zh import translate, translate_stream

            # Prepare output paths
            input_name = Path(input_path).stem
            output_dir = self.temp_dir

            mono_path = None
            dual_path = None

            # Build translation parameters
            params = {
                "lang_in": source_lang,
                "lang_out": target_lang,
                "service": service,
                "thread": thread_count,
                "output": output_dir,
            }

            if pages:
                params["pages"] = pages

            # Add service-specific parameters
            envs = service_params.pop("envs", {})
            if envs:
                params["envs"] = envs

            if "model" in service_params:
                params["model"] = service_params["model"]

            if "prompt" in service_params:
                params["prompt"] = service_params["prompt"]

            # Update progress
            if progress_callback:
                progress_callback(TranslationProgress(
                    status=TranslationStatus.IN_PROGRESS,
                    message=f"Translating with {service}..."
                ))

            # Execute translation
            # pdf2zh.translate returns paths to mono and dual PDFs
            mono_file, dual_file = translate(
                files=[input_path],
                **params
            )[0]  # Get result for first (only) file

            # Handle output format
            if output_format == "mono":
                mono_path = mono_file
            elif output_format == "dual":
                dual_path = dual_file
            else:  # "both"
                mono_path = mono_file
                dual_path = dual_file

            # Update progress
            if progress_callback:
                progress_callback(TranslationProgress(
                    status=TranslationStatus.COMPLETED,
                    progress_percent=100.0,
                    message="Translation completed!"
                ))

            return TranslationResult(
                success=True,
                mono_path=mono_path,
                dual_path=dual_path,
                metadata={
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "service": service,
                }
            )

        except ImportError:
            error_msg = "pdf2zh library not installed. Please install with: pip install pdf2zh"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(TranslationProgress(
                    status=TranslationStatus.FAILED,
                    error=error_msg
                ))
            return TranslationResult(success=False, error_message=error_msg)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Translation failed: {error_msg}")
            if progress_callback:
                progress_callback(TranslationProgress(
                    status=TranslationStatus.FAILED,
                    error=error_msg
                ))
            return TranslationResult(success=False, error_message=error_msg)

    def translate_stream(
        self,
        input_path: str,
        source_lang: str = "en",
        target_lang: str = "ko",
        service: str = "google",
        **kwargs
    ) -> Generator[Tuple[TranslationProgress, Optional[bytes]], None, None]:
        """
        Stream translation with progress updates.

        Yields progress updates and final PDF bytes.

        Args:
            input_path: Path to input PDF
            source_lang: Source language code
            target_lang: Target language code
            service: Translation service
            **kwargs: Additional translation parameters

        Yields:
            Tuples of (progress, pdf_bytes) - pdf_bytes is None until complete
        """
        try:
            from pdf2zh import translate_stream

            params = {
                "lang_in": source_lang,
                "lang_out": target_lang,
                "service": service,
                **kwargs
            }

            for progress, mono, dual in translate_stream(
                files=[input_path],
                **params
            ):
                prog = TranslationProgress(
                    status=TranslationStatus.IN_PROGRESS,
                    progress_percent=progress * 100 if progress else 0,
                    message=f"Translating... {int(progress * 100)}%" if progress else "Processing..."
                )
                yield prog, None

            # Final result
            yield TranslationProgress(
                status=TranslationStatus.COMPLETED,
                progress_percent=100.0,
                message="Complete!"
            ), dual

        except Exception as e:
            yield TranslationProgress(
                status=TranslationStatus.FAILED,
                error=str(e)
            ), None

    def cancel(self):
        """Cancel ongoing translation."""
        self._cancel_flag.set()

    def is_cancelled(self) -> bool:
        """Check if translation was cancelled."""
        return self._cancel_flag.is_set()


def translate_pdf(
    input_path: str,
    output_dir: Optional[str] = None,
    source_lang: str = "en",
    target_lang: str = "ko",
    service: str = "google",
    **kwargs
) -> TranslationResult:
    """
    Convenience function to translate a PDF file.

    Args:
        input_path: Path to input PDF
        output_dir: Output directory (uses temp if not specified)
        source_lang: Source language code
        target_lang: Target language code
        service: Translation service
        **kwargs: Additional parameters

    Returns:
        TranslationResult
    """
    translator = PDFTranslator(temp_dir=output_dir)
    return translator.translate(
        input_path=input_path,
        source_lang=source_lang,
        target_lang=target_lang,
        service=service,
        **kwargs
    )

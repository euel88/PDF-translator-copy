"""
Services module - Translation service providers and interfaces.
"""
from .base import TranslationServiceBase, TranslationResult
from .providers import (
    get_translation_service,
    get_available_services,
    validate_service_config,
    SUPPORTED_SERVICES,
)

__all__ = [
    "TranslationServiceBase",
    "TranslationResult",
    "get_translation_service",
    "get_available_services",
    "validate_service_config",
    "SUPPORTED_SERVICES",
]

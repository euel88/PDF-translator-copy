"""
Base classes for translation services.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class ServiceStatus(Enum):
    """Translation service status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    REQUIRES_API_KEY = "requires_api_key"
    ERROR = "error"


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    success: bool
    translated_text: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def success_result(cls, text: str, metadata: Optional[Dict] = None) -> "TranslationResult":
        return cls(success=True, translated_text=text, metadata=metadata)

    @classmethod
    def error_result(cls, message: str) -> "TranslationResult":
        return cls(success=False, error_message=message)


class TranslationServiceBase(ABC):
    """Abstract base class for translation services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._status = ServiceStatus.UNAVAILABLE

    @property
    @abstractmethod
    def name(self) -> str:
        """Service display name."""
        pass

    @property
    @abstractmethod
    def service_id(self) -> str:
        """Service identifier (used in pdf2zh)."""
        pass

    @property
    def requires_api_key(self) -> bool:
        """Whether this service requires an API key."""
        return False

    @property
    def api_key_env_var(self) -> Optional[str]:
        """Environment variable name for the API key."""
        return None

    @property
    def supported_models(self) -> List[str]:
        """List of supported models (if applicable)."""
        return []

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate service configuration."""
        pass

    @abstractmethod
    def check_availability(self) -> ServiceStatus:
        """Check if the service is available."""
        pass

    def get_status(self) -> ServiceStatus:
        """Get current service status."""
        return self._status

    def get_pdf2zh_params(self) -> Dict[str, Any]:
        """Get parameters for pdf2zh translation."""
        return {"service": self.service_id}

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        """
        Get UI configuration fields for this service.
        Returns list of field definitions for Streamlit.
        """
        return []


class FreeTranslationService(TranslationServiceBase):
    """Base class for free translation services (no API key required)."""

    @property
    def requires_api_key(self) -> bool:
        return False

    def validate_config(self) -> bool:
        return True


class PaidTranslationService(TranslationServiceBase):
    """Base class for paid translation services (API key required)."""

    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = api_key

    @property
    def requires_api_key(self) -> bool:
        return True

    def validate_config(self) -> bool:
        if not self.api_key:
            self._status = ServiceStatus.REQUIRES_API_KEY
            return False
        return True

    def get_pdf2zh_params(self) -> Dict[str, Any]:
        params = super().get_pdf2zh_params()
        if self.api_key and self.api_key_env_var:
            params["envs"] = {self.api_key_env_var: self.api_key}
        return params

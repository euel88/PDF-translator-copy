"""
Translation service provider implementations.
"""
import os
from typing import Dict, Any, List, Optional, Type
from .base import (
    TranslationServiceBase,
    FreeTranslationService,
    PaidTranslationService,
    ServiceStatus,
)


class GoogleTranslateService(FreeTranslationService):
    """Google Translate (free, no API key required)."""

    @property
    def name(self) -> str:
        return "Google Translate"

    @property
    def service_id(self) -> str:
        return "google"

    def check_availability(self) -> ServiceStatus:
        self._status = ServiceStatus.AVAILABLE
        return self._status


class DeepLService(PaidTranslationService):
    """DeepL Translation service."""

    @property
    def name(self) -> str:
        return "DeepL"

    @property
    def service_id(self) -> str:
        return "deepl"

    @property
    def api_key_env_var(self) -> str:
        return "DEEPL_API_KEY"

    def check_availability(self) -> ServiceStatus:
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env_var)
        if self.api_key:
            self._status = ServiceStatus.AVAILABLE
        else:
            self._status = ServiceStatus.REQUIRES_API_KEY
        return self._status

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "label": "DeepL API Key",
                "type": "password",
                "required": True,
                "help": "Get your API key from https://www.deepl.com/pro-api",
            }
        ]


class OpenAIService(PaidTranslationService):
    """OpenAI GPT translation service."""

    DEFAULT_MODEL = "gpt-4o-mini"
    SUPPORTED_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(api_key, config)
        self.model = model
        self.base_url = base_url

    @property
    def name(self) -> str:
        return "OpenAI GPT"

    @property
    def service_id(self) -> str:
        return "openai"

    @property
    def api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def supported_models(self) -> List[str]:
        return self.SUPPORTED_MODELS

    def check_availability(self) -> ServiceStatus:
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env_var)
        if self.api_key:
            self._status = ServiceStatus.AVAILABLE
        else:
            self._status = ServiceStatus.REQUIRES_API_KEY
        return self._status

    def get_pdf2zh_params(self) -> Dict[str, Any]:
        params = super().get_pdf2zh_params()
        params["model"] = self.model
        if self.base_url:
            params["envs"] = params.get("envs", {})
            params["envs"]["OPENAI_BASE_URL"] = self.base_url
        return params

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "label": "OpenAI API Key",
                "type": "password",
                "required": True,
                "help": "Get your API key from https://platform.openai.com/api-keys",
            },
            {
                "key": "model",
                "label": "Model",
                "type": "select",
                "options": self.SUPPORTED_MODELS,
                "default": self.DEFAULT_MODEL,
                "help": "Select the GPT model to use",
            },
            {
                "key": "base_url",
                "label": "Base URL (Optional)",
                "type": "text",
                "required": False,
                "help": "Custom API endpoint (for compatible services)",
            },
        ]


class OllamaService(FreeTranslationService):
    """Ollama local LLM service."""

    DEFAULT_MODEL = "gemma2"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.model = model
        self.host = host

    @property
    def name(self) -> str:
        return "Ollama (Local)"

    @property
    def service_id(self) -> str:
        return "ollama"

    def check_availability(self) -> ServiceStatus:
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                self._status = ServiceStatus.AVAILABLE
            else:
                self._status = ServiceStatus.UNAVAILABLE
        except Exception:
            self._status = ServiceStatus.UNAVAILABLE
        return self._status

    def get_pdf2zh_params(self) -> Dict[str, Any]:
        return {
            "service": self.service_id,
            "model": self.model,
            "envs": {"OLLAMA_HOST": self.host},
        }

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "host",
                "label": "Ollama Host",
                "type": "text",
                "default": self.DEFAULT_HOST,
                "help": "Ollama server URL",
            },
            {
                "key": "model",
                "label": "Model Name",
                "type": "text",
                "default": self.DEFAULT_MODEL,
                "help": "Model name (e.g., gemma2, llama3, mistral)",
            },
        ]


class DeepSeekService(PaidTranslationService):
    """DeepSeek translation service."""

    DEFAULT_MODEL = "deepseek-chat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(api_key, config)
        self.model = model

    @property
    def name(self) -> str:
        return "DeepSeek"

    @property
    def service_id(self) -> str:
        return "deepseek"

    @property
    def api_key_env_var(self) -> str:
        return "DEEPSEEK_API_KEY"

    def check_availability(self) -> ServiceStatus:
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env_var)
        if self.api_key:
            self._status = ServiceStatus.AVAILABLE
        else:
            self._status = ServiceStatus.REQUIRES_API_KEY
        return self._status

    def get_pdf2zh_params(self) -> Dict[str, Any]:
        params = super().get_pdf2zh_params()
        params["model"] = self.model
        return params

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "label": "DeepSeek API Key",
                "type": "password",
                "required": True,
                "help": "Get your API key from https://platform.deepseek.com/",
            },
            {
                "key": "model",
                "label": "Model",
                "type": "text",
                "default": self.DEFAULT_MODEL,
            },
        ]


class GeminiService(PaidTranslationService):
    """Google Gemini translation service."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(api_key, config)

    @property
    def name(self) -> str:
        return "Google Gemini"

    @property
    def service_id(self) -> str:
        return "gemini"

    @property
    def api_key_env_var(self) -> str:
        return "GEMINI_API_KEY"

    def check_availability(self) -> ServiceStatus:
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env_var)
        if self.api_key:
            self._status = ServiceStatus.AVAILABLE
        else:
            self._status = ServiceStatus.REQUIRES_API_KEY
        return self._status

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "label": "Gemini API Key",
                "type": "password",
                "required": True,
                "help": "Get your API key from https://aistudio.google.com/",
            },
        ]


class AzureService(PaidTranslationService):
    """Azure Translator service."""

    @property
    def name(self) -> str:
        return "Azure Translator"

    @property
    def service_id(self) -> str:
        return "azure"

    @property
    def api_key_env_var(self) -> str:
        return "AZURE_APIKEY"

    def check_availability(self) -> ServiceStatus:
        if not self.api_key:
            self.api_key = os.getenv(self.api_key_env_var)
        if self.api_key:
            self._status = ServiceStatus.AVAILABLE
        else:
            self._status = ServiceStatus.REQUIRES_API_KEY
        return self._status

    def get_config_ui_fields(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "api_key",
                "label": "Azure API Key",
                "type": "password",
                "required": True,
            },
        ]


# Registry of all supported services
SUPPORTED_SERVICES: Dict[str, Type[TranslationServiceBase]] = {
    "google": GoogleTranslateService,
    "openai": OpenAIService,
    "deepl": DeepLService,
    "ollama": OllamaService,
    "deepseek": DeepSeekService,
    "gemini": GeminiService,
    "azure": AzureService,
}


def get_translation_service(
    service_id: str,
    **kwargs
) -> Optional[TranslationServiceBase]:
    """Get a translation service instance by ID."""
    service_class = SUPPORTED_SERVICES.get(service_id)
    if service_class:
        return service_class(**kwargs)
    return None


def get_available_services() -> Dict[str, TranslationServiceBase]:
    """Get all available translation services."""
    services = {}
    for service_id, service_class in SUPPORTED_SERVICES.items():
        try:
            service = service_class()
            services[service_id] = service
        except Exception:
            continue
    return services


def validate_service_config(service_id: str, config: Dict[str, Any]) -> bool:
    """Validate configuration for a specific service."""
    service = get_translation_service(service_id, **config)
    if service:
        return service.validate_config()
    return False

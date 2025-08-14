# ===================================================================
# FILE: app/core/__init__.py (FINAL – Production-Ready)
# ===================================================================
"""
Core application components for Chatbot FILKOM.

Berisi konfigurasi utama, logging setup, dan custom exceptions
yang sering digunakan di seluruh modul aplikasi.
"""

from .config import settings
from .logging import setup_logging
from .exceptions import (
    ChatbotException,
    ModelLoadError,
    KnowledgeBaseError,
    IntentClassificationError,
    NERError,
    SearchError,
    ResponseGenerationError,
    ValidationError,
    UnknownError,
)

__all__ = [
    "settings",
    "setup_logging",
    "ChatbotException",
    "ModelLoadError",
    "KnowledgeBaseError",
    "IntentClassificationError",
    "NERError",
    "SearchError",
    "ResponseGenerationError",
    "ValidationError",
    "UnknownError",
]

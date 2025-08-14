# ===================================================================
# FILE: app/core/exceptions.py (FINAL – Step 9 Synced)
# ===================================================================
from fastapi import HTTPException
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

# ===================================================================
# BASE EXCEPTION
# ===================================================================

class ChatbotException(Exception):
    """Base exception untuk semua error di Chatbot FILKOM"""
    def __init__(self, message: str, details: Optional[Any] = None):
        self.message = message
        self.details = details
        logger.error(f"[ChatbotException] {message} | Details: {details}")
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} | Details: {self.details}" if self.details else self.message

# ===================================================================
# SPECIFIC CHATBOT EXCEPTIONS
# ===================================================================

class ModelLoadError(ChatbotException):
    """Error saat memuat model (Intent, NER, Semantic Search)"""
    pass

class KnowledgeBaseError(ChatbotException):
    """Error saat inisialisasi atau akses Knowledge Base"""
    pass

class IntentClassificationError(ChatbotException):
    """Error saat proses Intent Classification"""
    pass

class NERError(ChatbotException):
    """Error saat Named Entity Recognition"""
    pass

class SearchError(ChatbotException):
    """Error saat Semantic Search"""
    pass

class ResponseGenerationError(ChatbotException):
    """Error saat proses generate jawaban"""
    pass

class ValidationError(ChatbotException):
    """Error validasi input atau entitas tidak sesuai"""
    pass

class UnknownError(ChatbotException):
    """Error tidak terduga (unexpected exception)"""
    pass

# ===================================================================
# HTTP EXCEPTIONS (untuk API Layer)
# ===================================================================

class HTTPValidationError(HTTPException):
    def __init__(self, detail: str):
        logger.warning(f"[HTTPValidationError] {detail}")
        super().__init__(status_code=422, detail=detail)

class HTTPNotFoundError(HTTPException):
    def __init__(self, detail: str = "Resource not found"):
        logger.warning(f"[HTTPNotFoundError] {detail}")
        super().__init__(status_code=404, detail=detail)

class HTTPInternalServerError(HTTPException):
    def __init__(self, detail: str = "Internal server error"):
        logger.error(f"[HTTPInternalServerError] {detail}")
        super().__init__(status_code=500, detail=detail)

# ===================================================================
# EXPORTS
# ===================================================================
__all__ = [
    "ChatbotException",
    "ModelLoadError",
    "KnowledgeBaseError",
    "IntentClassificationError",
    "NERError",
    "SearchError",
    "ResponseGenerationError",
    "ValidationError",
    "UnknownError",
    "HTTPValidationError",
    "HTTPNotFoundError",
    "HTTPInternalServerError"
]

# ===================================================================
# FILE: app/models/__init__.py (FINAL)
# ===================================================================
"""
Models package initializer.
Menyediakan akses global ke semua Pydantic models utama
(ChatRequest, ChatResponse, ProcessingStatus, dll)
untuk memudahkan import di seluruh project.
"""

# Chatbot interaction models
from .chat_models import (
    ChatRequest,
    ChatResponse,
    ProcessingStatus,
    StatusEnum
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ProcessingStatus",
    "StatusEnum"
]

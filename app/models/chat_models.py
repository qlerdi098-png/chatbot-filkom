# ===================================================================
# FILE: app/models/chat_models.py (FINAL PATCHED VERSION)
# ===================================================================
"""
Pydantic models untuk request/response chatbot dan status pemrosesan.
Digunakan pada semua endpoint utama, termasuk chat interaksi,
health check, dan monitoring pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# ===================================================================
# BASE MODEL FIX UNTUK PROTECTED NAMESPACE
# ===================================================================

class SafeBaseModel(BaseModel):
    """
    BaseModel dengan konfigurasi untuk menghilangkan warning
    'Field "model_info" has conflict with protected namespace "model_".'
    """
    model_config = {
        "protected_namespaces": ()
    }

# ===================================================================
# ENUMS
# ===================================================================

class StatusEnum(str, Enum):
    """
    Enum untuk status pemrosesan chatbot.
    Membantu standarisasi status di seluruh pipeline.
    """
    processing = "processing"  # Sedang diproses
    completed = "completed"    # Selesai dengan sukses
    error = "error"            # Terjadi kesalahan

# ===================================================================
# REQUEST & RESPONSE MODELS
# ===================================================================

class ChatRequest(SafeBaseModel):
    """
    Model untuk request chat dari user.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pesan dari user (1-500 karakter)."
    )
    user_id: Optional[str] = Field(
        None,
        description="ID user (opsional, untuk personalisasi)."
    )
    session_id: Optional[str] = Field(
        None,
        description="ID sesi (opsional, untuk pelacakan percakapan)."
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Konteks tambahan (riwayat percakapan, preferensi, dll)."
    )

class ChatResponse(SafeBaseModel):
    """
    Model untuk response chatbot ke user.
    """
    response: str = Field(
        ...,
        description="Pesan balasan dari bot."
    )
    intent: Optional[str] = Field(
        None,
        description="Intent yang terdeteksi (contoh: JADWAL_KULIAH, INFO_DOSEN)."
    )
    entities: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Hasil ekstraksi entitas (hasil NER)."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Skor keyakinan model (0.0 - 1.0)."
    )
    source: Optional[str] = Field(
        None,
        description="Sumber jawaban (contoh: template, semantic_search, kb_lookup)."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Waktu response dihasilkan."
    )
    session_id: Optional[str] = Field(
        None,
        description="ID sesi (untuk pelacakan percakapan)."
    )

class ProcessingStatus(SafeBaseModel):
    """
    Model untuk status pemrosesan pipeline chatbot.
    Berguna untuk monitoring progres dan debugging.
    """
    stage: str = Field(
        ...,
        description="Tahap pemrosesan saat ini "
                    "(contoh: intent_classification, ner_extraction)."
    )
    status: StatusEnum = Field(
        ...,
        description="Status pemrosesan (processing, completed, error)."
    )
    message: Optional[str] = Field(
        None,
        description="Pesan status atau detail error jika terjadi kesalahan."
    )
    progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Persentase progres (0 - 100)."
    )

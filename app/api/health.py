# ===================================================================
# FILE: app/api/health.py (FINAL – Production-Ready & Fully Synced)
# ===================================================================
"""
Health & Status API untuk monitoring layanan Chatbot FILKOM.
Memberikan status ringkas dan detail terkait layanan:
- Knowledge Base
- Intent Classification
- Named Entity Recognition (NER)
- Semantic Search
"""

from fastapi import APIRouter
from app.core.config import settings
import logging

from app.services.kb_service import is_kb_ready, kb_loader
from app.services.intent_service import intent_service
from app.services.ner_service import ner_service
from app.services.semantic_search_service import search_service

router = APIRouter()
logger = logging.getLogger(__name__)

# ===================================================================
# ✅ HEALTH CHECK (Ringkas)
# ===================================================================
@router.get("/health", operation_id="health_check_status")
async def health_check():
    """
    Extended health check → status ringkas layanan utama.
    """
    try:
        kb_ready = is_kb_ready()
        intent_ready = intent_service.is_loaded
        ner_ready = ner_service.is_loaded
        search_ready = search_service.is_loaded

        all_ready = all([kb_ready, intent_ready, ner_ready, search_ready])

        return {
            "status": "healthy" if all_ready else "initializing",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG,
            "device": settings.DEVICE,
            "services": {
                "knowledge_base": "ready" if kb_ready else "not_ready",
                "intent_service": "ready" if intent_ready else "not_ready",
                "ner_service": "ready" if ner_ready else "not_ready",
                "semantic_search": "ready" if search_ready else "not_ready",
            },
            "all_services_ready": all_ready
        }
    except Exception as e:
        logger.error(f"[HEALTH CHECK ERROR] {str(e)}")
        return {
            "status": "error",
            "message": "Health check failed",
            "error": str(e) if settings.DEBUG else "Internal error"
        }

# ===================================================================
# ✅ DETAILED STATUS (Lengkap)
# ===================================================================
@router.get("/status", operation_id="detailed_health_status")
async def detailed_status():
    """
    Detailed system status → monitoring lengkap pipeline chatbot.
    """
    try:
        kb_status = {
            "loaded": kb_loader.is_loaded,
            "entries": {
                "dosen": len(kb_loader.dosen_data),
                "mata_kuliah": len(kb_loader.matakuliah_data),
                "jadwal": len(kb_loader.jadwal_data),
                "skripsi": len(kb_loader.skripsi_data),
                "regulasi_sks": len(kb_loader.regulasi_sks_data)
            } if kb_loader.is_loaded else {}
        }

        intent_status = intent_service.get_model_info() if intent_service.is_loaded else {"loaded": False}
        ner_status = ner_service.get_status() if ner_service.is_loaded else {"loaded": False}
        search_status = search_service.get_status() if search_service.is_loaded else {"loaded": False}

        return {
            "app_info": {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "debug": settings.DEBUG,
                "device": settings.DEVICE
            },
            "services_status": {
                "knowledge_base": kb_status,
                "intent_classification": intent_status,
                "named_entity_recognition": ner_status,
                "semantic_search": search_status
            },
            "system_info": {
                "models_dir": str(settings.MODELS_DIR),
                "data_dir": str(settings.DATA_DIR),
                "max_length": settings.MAX_LENGTH,
                "batch_size": settings.BATCH_SIZE
            }
        }
    except Exception as e:
        logger.error(f"[DETAILED STATUS ERROR] {str(e)}")
        return {
            "status": "error",
            "message": "Status check failed",
            "error": str(e) if settings.DEBUG else "Internal error"
        }

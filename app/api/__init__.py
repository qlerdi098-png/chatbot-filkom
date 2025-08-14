# ===================================================================
# FILE: app/api/__init__.py (FINAL – Phase 3: Production-Ready)
# ===================================================================
"""
API routes and endpoints registration.

Routers:
- main_router        : Knowledge Base (status, jadwal, regulasi, skripsi, templates)
- health_router      : Health check and system status
- intent_router      : Intent Classification Service (Phase 2)
- ner_router         : Named Entity Recognition (Phase 2)
- search_router      : Semantic Search Service (Phase 3)
- chat_router        : Chat Processing Pipeline (Phase 3)

Version: 1.3.1 (Fully Synced)
"""

from fastapi import APIRouter

# Import routers
from .routes import api_router as main_router
from .health import router as health_router
from .intent_api import router as intent_router
from .ner_api import router as ner_router
from .search_api import router as search_router
from .chat_api import router as chat_router  # ✅ Sinkron dengan Chat Pipeline

__version__ = "1.3.1"
__all__ = [
    "main_router",
    "health_router",
    "intent_router",
    "ner_router",
    "search_router",
    "chat_router",
]

# ===================================================================
# ✅ COMBINED API ROUTER
# ===================================================================
api_router = APIRouter()

api_router.include_router(health_router, prefix="/api/v1", tags=["Health Check"])
api_router.include_router(main_router, prefix="/api/v1", tags=["Knowledge Base"])
api_router.include_router(intent_router, prefix="/api/v1", tags=["Intent Classification"])
api_router.include_router(ner_router, prefix="/api/v1", tags=["Named Entity Recognition"])
api_router.include_router(search_router, prefix="/api/v1", tags=["Semantic Search"])
api_router.include_router(chat_router, prefix="/api/v1", tags=["Chat Pipeline"])

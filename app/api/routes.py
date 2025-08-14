# ===================================================================
# FILE: app/api/routes.py (FINAL – Hotfix 1.9, Python 3.9 Compatible)
# ===================================================================

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Routers lain
from app.api import health, intent_api, ner_api, chat_api

# Service KB
from app.services.kb_service import get_kb_query, is_kb_ready
from app.core.exceptions import KnowledgeBaseError

logger = logging.getLogger(__name__)

# Main API Router
api_router = APIRouter()

# Templates untuk Web Interface (Home, Chat, Status)
templates = Jinja2Templates(directory="frontend/templates")

# ✅ Cache untuk Demo Search
_cached_demo_results: Optional[Dict[str, Any]] = None

# ===================================================================
# ✅ INCLUDE ALL ROUTERS
# ===================================================================
api_router.include_router(health.router)
api_router.include_router(intent_api.router)
api_router.include_router(ner_api.router)
api_router.include_router(chat_api.router)

# ===================================================================
# ✅ CHAT WEB INTERFACE
# ===================================================================
@api_router.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    logger.info("💬 Serving Chat Web Interface")
    return templates.TemplateResponse("chat.html", {"request": request})

# ===================================================================
# ✅ KNOWLEDGE BASE STATUS
# ===================================================================
@api_router.get("/kb/status", operation_id="kb_status_get", response_model=Dict[str, Any])
async def kb_status() -> Dict[str, Any]:
    try:
        if is_kb_ready():
            kb = get_kb_query()
            logger.info("✅ KB status checked successfully")
            return {
                "status": "ready",
                "is_loaded": kb.kb.is_loaded,
                "load_time": f"{kb.kb.load_time:.2f}s",
                "data_summary": {
                    "dosen": len(kb.kb.dosen_data),
                    "mata_kuliah": len(kb.kb.matakuliah_data),
                    "jadwal": len(kb.kb.jadwal_data),
                    "kalender": len(kb.kb.kalender_data),
                    "skripsi": len(kb.kb.skripsi_data),
                    "regulasi_sks": len(kb.kb.regulasi_sks_data),
                    "templates": len(kb.kb.response_templates)
                }
            }
        logger.warning("⚠️ KB not initialized when checking status")
        return {"status": "not_ready", "is_loaded": False, "message": "Knowledge base not initialized"}
    except Exception as e:
        logger.error(f"❌ Error checking KB status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================================================================
# ✅ DEMO SEARCH KB (Optimized with Cache)
# ===================================================================
@api_router.get("/demo/search-demo", operation_id="kb_demo_search_get", response_model=Dict[str, Any])
async def demo_search() -> Dict[str, Any]:
    global _cached_demo_results
    try:
        if _cached_demo_results:
            logger.debug("⚡ Returning cached demo search results")
            return _cached_demo_results

        if not is_kb_ready():
            logger.warning("⚠️ KB not ready during demo search")
            raise HTTPException(status_code=503, detail="Knowledge base not ready")

        kb = get_kb_query()
        results = {
            "dosen_machine_learning": [d.nama_lengkap for d in kb.find_dosen_by_matkul("machine learning")[:5]],
            "jadwal_senin": [f"{j.mata_kuliah} - {j.jam}" for j in kb.find_jadwal_by_hari("senin")[:5]],
            "regulasi_sks_contoh": (
                lambda r: f"SKS Max: {r.sks_maksimal}, Min: {r.sks_minimal}"
            )(kb.get_batas_sks("2", 3.5, "Teknik Informatika"))
            if kb.get_batas_sks("2", 3.5, "Teknik Informatika") else None,
            "syarat_skripsi_ti": [
                f"Min SKS: {s.sks_minimum}, Min IPK: {s.ipk_minimum}"
                for s in kb.get_syarat_skripsi("Teknik Informatika")[:2]
            ]
        }

        _cached_demo_results = {
            "message": "Demo query results (cached)",
            "results": results,
            "cached": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info("✅ Demo search results generated & cached")
        return _cached_demo_results

    except Exception as e:
        logger.error(f"❌ Error in demo search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

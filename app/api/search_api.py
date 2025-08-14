# ===================================================================
# FILE: app/api/search_api.py (FINAL – Hotfix 1.9, Python 3.9 Compatible)
# ===================================================================

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.services.semantic_search_service import search_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Semantic Search"])

# ✅ Cache untuk Demo Search
_cached_demo_results: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query", example="mata kuliah machine learning")
    search_type: str = Field("hybrid", description="Search type: bm25, semantic, or hybrid")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)

class SearchResponse(BaseModel):
    success: bool
    query: Optional[str] = None
    search_type: Optional[str] = None
    results: List[Dict[str, Any]] = []
    total_found: int = 0
    error: Optional[str] = None

@router.on_event("startup")
async def initialize_search():
    try:
        if search_service.load_models():
            logger.info("✅ Semantic search models loaded successfully")
        else:
            logger.warning("⚠️ Semantic search models not found. Please rebuild if needed.")
    except Exception as e:
        logger.error(f"❌ Error initializing semantic search service: {str(e)}")

@router.get("/demo", operation_id="search_demo_get")
async def search_demo():
    """
    Cache + limit top result → response super cepat (<0.1s)
    """
    global _cached_demo_results
    try:
        if _cached_demo_results:
            logger.debug("⚡ Returning cached search demo results")
            return _cached_demo_results

        if not search_service.is_loaded:
            return {"error": "Search service not loaded", "status": "Models need to be loaded first"}

        demo_queries = [
            "syarat skripsi teknik informatika",
            "jadwal mata kuliah machine learning",
            "dosen pengampu basis data",
            "batas sks semester 3",
            "mata kuliah prasyarat web programming"
        ]

        results = {}
        for query in demo_queries:
            search_result = search_service.search(query, "hybrid", 1)
            results[query] = {
                "found": search_result["total_found"],
                "top_result": search_result["results"][0] if search_result["results"] else None
            }

        _cached_demo_results = {
            "service": "Semantic Search Demo",
            "total_documents": len(search_service.documents),
            "demo_results": results,
            "cached": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": search_service.get_status()
        }

        logger.info("✅ Search demo results generated & cached")
        return _cached_demo_results

    except Exception as e:
        logger.error(f"❌ Error in search demo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", operation_id="search_status_get")
async def get_search_status():
    """
    ✅ Cek status Semantic Search Service (Hotfix 1.9 – Cached Support)
    """
    try:
        if not search_service.is_loaded:
            return {
                "status": "not_loaded",
                "loaded": False,
                "documents_indexed": 0,
                "cached": False,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        status = search_service.get_status()  # Sudah tersedia di semantic_search_service
        return {
            "status": "healthy" if status.get("loaded", False) else "not_loaded",
            "loaded": status.get("loaded", False),
            "model_name": status.get("model_name", "Unknown"),
            "documents_indexed": len(search_service.documents),
            "cached": True,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"[SEARCH STATUS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/query", operation_id="search_query_get")
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    search_type: str = Query("hybrid", description="Search type: bm25, semantic, or hybrid"),
    top_k: int = Query(5, description="Number of results to return", ge=1, le=20)
):
    """
    ✅ Search documents (GET Method – cepat & untuk testing)
    """
    try:
        if not search_service.is_loaded:
            raise HTTPException(status_code=503, detail="Search service not loaded")

        result = search_service.search(query=q, search_type=search_type, top_k=top_k)
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

        return {
            "success": True,
            "query": q,
            "search_type": search_type,
            "total_found": result.get("total_found", 0),
            "results": result.get("results", []),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        logger.error(f"[SEARCH QUERY ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


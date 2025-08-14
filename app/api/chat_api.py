# ======================================================
# FILE: app/api/chat_api.py (Hotfix 1.9 – Cached Support & Python 3.9 Compatible)
# ======================================================

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
import logging

from app.services.chat_pipeline import ChatPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat Pipeline"])

# ======================================================
# Global Pipeline Instance
# ======================================================
chat_pipeline: Optional[ChatPipeline] = None

def get_chat_pipeline() -> ChatPipeline:
    global chat_pipeline
    if chat_pipeline is None:
        chat_pipeline = ChatPipeline()
        logger.info("✅ ChatPipeline loaded in chat_api")
    return chat_pipeline

# ======================================================
# Request & Response Models
# ======================================================
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    search_results: Optional[List[Dict]] = None
    template_used: Optional[str] = None
    fallback_reason: Optional[str] = None
    processing_time: Optional[float] = None
    cached: bool = False           # ✅ Tambahan untuk support cache
    timestamp: str

class HistoryResponse(BaseModel):
    conversation_history: List[Dict[str, Any]]
    total_messages: int

# ======================================================
# Endpoints
# ======================================================

@router.post("/process", operation_id="chat_process_post")
async def process_chat(
    request: ChatRequest,
    pipeline: ChatPipeline = Depends(get_chat_pipeline)
):
    """Proses chat user → intent → NER → Template/KB → Response"""
    try:
        # ✅ Tidak perlu await karena process_message sinkron
        result = pipeline.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )

        return ChatResponse(
            response=result.response,
            intent=result.intent,
            entities=result.entities,
            confidence=result.confidence,
            search_results=result.search_results,
            template_used=result.template_used,
            fallback_reason=result.fallback_reason,
            processing_time=result.processing_time,
            cached=result.cached,  # ✅ Flag cache ditampilkan di response
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        logger.error(f"[CHAT API ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.get("/history/{user_id}/{session_id}", operation_id="chat_history_get")
async def get_chat_history(
    user_id: str,
    session_id: str,
    pipeline: ChatPipeline = Depends(get_chat_pipeline)
):
    """Ambil history percakapan berdasarkan user & session"""
    try:
        history = pipeline.get_conversation_history(user_id, session_id)
        return HistoryResponse(conversation_history=history, total_messages=len(history))
    except Exception as e:
        logger.error(f"[CHAT HISTORY ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval error: {str(e)}")


@router.delete("/history/{user_id}/{session_id}", operation_id="chat_history_delete")
async def clear_chat_history(
    user_id: str,
    session_id: str,
    pipeline: ChatPipeline = Depends(get_chat_pipeline)
):
    """Hapus history percakapan"""
    try:
        pipeline.clear_context(user_id, session_id)
        return {"message": "✅ Chat history cleared successfully"}
    except Exception as e:
        logger.error(f"[CLEAR HISTORY ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Clear history error: {str(e)}")


@router.get("/status", operation_id="chat_status_get")
async def chat_status():
    """Cek status Chat Pipeline"""
    try:
        pipeline = get_chat_pipeline()
        return {
            "status": "ready",
            "services": {
                "intent_service": "✅ loaded" if pipeline.intent_service.is_loaded else "❌ not_loaded",
                "ner_service": "✅ loaded" if pipeline.ner_service.is_loaded else "❌ not_loaded",
                "knowledge_base": "✅ loaded" if pipeline.kb_service.kb.is_loaded else "❌ not_loaded",
                "semantic_search": "✅ loaded" if pipeline.semantic_search.is_loaded else "❌ not_loaded",
            },
            "templates_loaded": len(pipeline.template_service.templates),
            "active_contexts": len(pipeline.contexts),
            "intent_threshold": pipeline.intent_threshold,
            "fallback_threshold": pipeline.fallback_threshold
        }
    except Exception as e:
        logger.error(f"[STATUS ERROR] {e}")
        return {"status": "error", "error": str(e)}


@router.post("/demo", operation_id="chat_demo_post")
async def chat_demo(pipeline: ChatPipeline = Depends(get_chat_pipeline)):
    """Demo chat dengan pertanyaan predefined"""
    demo_queries = [
        "Siapa dosen pengampu mata kuliah Machine Learning?",
        "Jadwal kuliah Basis Data hari apa?",
        "Berapa SKS mata kuliah Algoritma dan Pemrograman?",
        "Kontak dosen Hendry Fonda",
        "Syarat untuk mengambil skripsi"
    ]

    results = []
    for query in demo_queries:
        # ✅ Sinkron, tidak pakai await
        result = pipeline.process_message(query, "demo_user", "demo_session")
        results.append({
            "query": query,
            "response": result.response,
            "intent": result.intent,
            "confidence": result.confidence,
            "entities": result.entities,
            "cached": result.cached
        })

    return {"demo_results": results}

@router.get("/process", operation_id="chat_process_get")
async def process_chat_get(
    message: str,
    user_id: str = "default",
    session_id: str = "default",
    pipeline: ChatPipeline = Depends(get_chat_pipeline)
):
    """Versi GET hanya untuk testing cepat di browser"""
    result = pipeline.process_message(
        message=message,
        user_id=user_id,
        session_id=session_id
    )
    return {
        "response": result.response,
        "intent": result.intent,
        "confidence": result.confidence,
        "entities": result.entities,
        "cached": result.cached
    }

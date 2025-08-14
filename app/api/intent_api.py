# ===================================================================
# FILE: app/api/intent_api.py (FINAL – Production-Ready & Fully Synced)
# ===================================================================

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, List
import logging

from app.services.intent_service import get_intent_service, is_intent_service_ready
from app.core.exceptions import IntentClassificationError

logger = logging.getLogger(__name__)
router = APIRouter()

# ===================================================================
# ✅ REQUEST/RESPONSE MODELS
# ===================================================================

class IntentRequest(BaseModel):
    text: str
    return_probabilities: bool = False

class TopPrediction(BaseModel):
    intent: str
    confidence: float
    model_config = ConfigDict(protected_namespaces=())  # ✅ Hindari warning

class IntentResponse(BaseModel):
    text: str
    predicted_intent: str
    confidence: float
    is_confident: bool
    probabilities: Optional[Dict[str, float]] = None
    model_config = ConfigDict(protected_namespaces=())

class IntentServiceStatus(BaseModel):
    status: str
    is_loaded: bool
    model_details: Optional[Dict] = None
    model_config = ConfigDict(protected_namespaces=())

class DetailedIntentResponse(BaseModel):
    text: str
    predicted_intent: str
    confidence: float
    is_confident: bool
    top_3_predictions: List[TopPrediction]
    all_probabilities: Optional[Dict[str, float]] = None
    model_config = ConfigDict(protected_namespaces=())

# ===================================================================
# ✅ ENDPOINTS
# ===================================================================

@router.get("/intent/status", operation_id="intent_service_status")
async def intent_service_status():
    """
    Cek status Intent Classification Service
    """
    try:
        if is_intent_service_ready():
            svc = get_intent_service()
            return IntentServiceStatus(
                status="ready",
                is_loaded=True,
                model_details=svc.get_model_info()
            )
        return IntentServiceStatus(status="not_ready", is_loaded=False)
    except Exception as e:
        logger.error(f"[INTENT STATUS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/intent/predict", operation_id="intent_predict_post")
async def predict_intent(request: IntentRequest):
    """
    Prediksi intent secara detail (Top-3 + Probabilitas)
    """
    try:
        if not is_intent_service_ready():
            raise HTTPException(status_code=503, detail="Intent service not ready")

        svc = get_intent_service()
        intent, confidence, probs = svc.predict_intent_with_probabilities(request.text)

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3 = [TopPrediction(intent=i, confidence=p) for i, p in sorted_probs]

        return DetailedIntentResponse(
            text=request.text,
            predicted_intent=intent,
            confidence=confidence,
            is_confident=svc.is_confident_prediction(confidence),
            top_3_predictions=top_3,
            all_probabilities=probs if request.return_probabilities else None
        )
    except IntentClassificationError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"[INTENT PREDICT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intent/predict", operation_id="intent_predict_get")
async def predict_intent_get(
    text: str = Query(..., description="Text to classify"),
    probabilities: bool = Query(False, description="Return all probabilities"),
    detailed: bool = Query(True, description="Return detailed response")
):
    """
    Prediksi intent via GET (opsi ringkas atau detail)
    """
    request = IntentRequest(text=text, return_probabilities=probabilities)
    if detailed:
        return await predict_intent(request)

    try:
        svc = get_intent_service()
        intent, confidence = svc.predict_intent(text)
        return IntentResponse(
            text=text,
            predicted_intent=intent,
            confidence=confidence,
            is_confident=svc.is_confident_prediction(confidence)
        )
    except Exception as e:
        logger.error(f"[INTENT PREDICT (GET) ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intent/classes", operation_id="intent_classes_get")
async def get_intent_classes():
    """
    Ambil daftar intent yang tersedia (Hotfix 1.9 – Cached Support)
    """
    try:
        if not is_intent_service_ready():
            raise HTTPException(status_code=503, detail="Intent service not ready")
        svc = get_intent_service()
        classes = svc.get_cached_classes()  # ✅ Ambil dari cache

        return {
            "total_classes": classes.get("total_classes", 0),
            "intent_classes": classes.get("intent_classes", []),
            "cached": classes.get("cached", False),
            "confidence_threshold": svc.confidence_threshold,
            "model_name": svc.model_name,
            "device": str(svc.device)
        }
    except Exception as e:
        logger.error(f"[INTENT CLASSES ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

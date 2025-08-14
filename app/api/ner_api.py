# ===================================================================
# FILE: app/api/ner_api.py (CLEAN FIXED – Hotfix 1.7)
# Optimasi full cache untuk /ner/status & /ner/labels
# ===================================================================

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
import logging

from app.services.ner_service import get_ner_service, is_ner_ready

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ner", tags=["Named Entity Recognition"])

# ===================================================================
# ✅ Pydantic Models
# ===================================================================
class EntityInfo(BaseModel):
    entity: str
    value: str
    start: int
    end: int
    confidence: float
    model_config = ConfigDict(protected_namespaces=())

class NERRequest(BaseModel):
    text: str = Field(..., description="Teks untuk diekstraksi entitasnya", max_length=512)
    model_config = ConfigDict(protected_namespaces=())

class NERResponse(BaseModel):
    text: str
    entities: List[EntityInfo]
    entity_count: int
    prediction_time: float
    model_config = ConfigDict(protected_namespaces=())

class KeyEntitiesResponse(BaseModel):
    text: str
    key_entities: Dict[str, List[str]]
    prediction_time: float
    model_config = ConfigDict(protected_namespaces=())

# ===================================================================
# ✅ ENDPOINTS – Optimasi Cache
# ===================================================================

@router.get("/status", operation_id="ner_status")
async def get_ner_status():
    """
    ✅ Cek status NER service (super cepat via cache bawaan NERService)
    """
    try:
        ner_service = get_ner_service()
        return ner_service.get_status()  # ✅ Cached
    except Exception as e:
        logger.error(f"[NER STATUS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/labels", operation_id="ner_labels")
async def get_ner_labels():
    """
    ✅ Ambil daftar label entitas dari cache (tanpa hitungan ulang)
    """
    try:
        if not is_ner_ready():
            raise HTTPException(status_code=503, detail="NER service not ready")
        ner_service = get_ner_service()
        return ner_service.get_labels_cached()  # ✅ Cached
    except Exception as e:
        logger.error(f"[NER LABELS ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", operation_id="ner_predict_post")
async def predict_entities(request: NERRequest):
    """
    Prediksi Named Entities dari input text (POST method)
    """
    try:
        if not is_ner_ready():
            raise HTTPException(status_code=503, detail="NER service not ready")
        ner_service = get_ner_service()
        result = ner_service.predict(request.text)
        return NERResponse(
            text=result["text"],
            entities=[EntityInfo(**ent) for ent in result["entities"]],
            entity_count=result["entity_count"],
            prediction_time=result["prediction_time"]
        )
    except Exception as e:
        logger.error(f"[NER PREDICT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict", operation_id="ner_predict_get")
async def predict_entities_get(
    text: str = Query(..., description="Teks untuk diekstraksi entitasnya", max_length=512)
):
    """Prediksi Named Entities (GET method – testing cepat)"""
    request = NERRequest(text=text)
    return await predict_entities(request)

@router.post("/extract", operation_id="ner_extract_post")
async def extract_key_entities(request: NERRequest):
    """
    Ekstraksi key entities dalam format key-value untuk chatbot pipeline
    """
    try:
        if not is_ner_ready():
            raise HTTPException(status_code=503, detail="NER service not ready")
        ner_service = get_ner_service()
        result = ner_service.predict(request.text)
        key_entities = ner_service.extract_key_entities(request.text)
        return KeyEntitiesResponse(
            text=result["text"],
            key_entities=key_entities,
            prediction_time=result["prediction_time"]
        )
    except Exception as e:
        logger.error(f"[NER EXTRACT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/extract", operation_id="ner_extract_get")
async def extract_key_entities_get(
    text: str = Query(..., description="Teks untuk extract key entities", max_length=512)
):
    """Ekstraksi key entities (GET method – testing cepat)"""
    request = NERRequest(text=text)
    return await extract_key_entities(request)

@router.get("/demo", operation_id="ner_demo_get")
async def ner_demo():
    """
    Demo NER dengan query contoh seperti pada training dataset
    """
    try:
        if not is_ner_ready():
            raise HTTPException(status_code=503, detail="NER service not ready")
        ner_service = get_ner_service()
        demo_queries = [
            "siapa dosen yang ngajar mata kuliah algoritma pemrograman?",
            "jadwal kuliah basis data hari senin jam berapa?",
            "berapa sks mata kuliah machine learning?",
            "prasyarat untuk mengambil mata kuliah kecerdasan buatan apa saja?",
            "kontak pak herman bisa dihubungi dimana?",
            "semester berapa boleh ambil skripsi?",
            "info dosen pengampu jaringan komputer dong"
        ]
        results = {}
        for i, q in enumerate(demo_queries, 1):
            prediction = ner_service.predict(q)
            key_entities = ner_service.extract_key_entities(q)
            results[f"example_{i}"] = {
                "query": q,
                "entities": prediction["entities"],
                "key_entities": key_entities,
                "prediction_time": prediction["prediction_time"]
            }
        return {
            "message": "NER Demo Results",
            "model_details": ner_service.get_status().get("model_info", {}),
            "results": results
        }
    except Exception as e:
        logger.error(f"[NER DEMO ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", operation_id="ner_performance_get")
async def get_ner_performance():
    """Metrik performa model (berdasarkan hasil training)"""
    try:
        if not is_ner_ready():
            raise HTTPException(status_code=503, detail="NER service not ready")
        return {
            "overall_metrics": {
                "test_f1": 0.9266,
                "test_accuracy": 0.9800,
                "best_val_f1": 0.9336
            },
            "top_entities": {
                "DOSEN": {"f1": 0.9520},
                "MATA_KULIAH": {"f1": 0.9773},
                "HARI": {"f1": 0.9753},
                "SEMESTER": {"f1": 0.9500},
                "PRODI": {"f1": 0.9848}
            },
            "model_architecture": "IndoBERT+BiLSTM+CRF"
        }
    except Exception as e:
        logger.error(f"[NER PERFORMANCE ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

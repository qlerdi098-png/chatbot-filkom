# ===================================================================
# FILE: app/core/config.py (FINAL – FIXED & CONSISTENT – Step 9 Synced)
# ===================================================================
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path
import torch
import multiprocessing
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    APP_NAME: str = "Chatbot FILKOM"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ✅ Tambahan agar sesuai dengan .env
    ENV: str = os.getenv("ENV", "dev")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CUDA_AVAILABLE: bool = os.getenv("CUDA_AVAILABLE", "false").lower() == "true"

    # ✅ Device auto detect
    DEVICE: str = (
        "cuda" if torch.cuda.is_available() and CUDA_AVAILABLE else "cpu"
    )

    # ✅ Directories (absolute resolved)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = (BASE_DIR / "models").resolve()
    DATA_DIR: Path = (BASE_DIR / "data").resolve()

    # ✅ Intent Classifier Paths
    INTENT_MODEL_PATH: Path = MODELS_DIR / "intent_classifier" / "best_model.pth"
    INTENT_CONFIG_PATH: Path = MODELS_DIR / "intent_classifier" / "model_config.json"
    INTENT_MAPPING_PATH: Path = MODELS_DIR / "intent_classifier" / "intent_mapping.json"

    # ✅ NER Model Paths
    NER_MODEL_PATH: Path = MODELS_DIR / "ner_model" / "best_model.pt"
    NER_CONFIG_PATH: Path = MODELS_DIR / "ner_model" / "model_config.json"
    NER_LABEL_ENCODER_PATH: Path = MODELS_DIR / "ner_model" / "label_encoder.pkl"

    # ✅ Semantic Search Paths
    SEARCH_INDEX_PATH: Path = MODELS_DIR / "semantic_search" / "faiss_index.bin"
    SEARCH_EMBEDDINGS_PATH: Path = MODELS_DIR / "semantic_search" / "embeddings.npy"
    SEARCH_BM25_PATH: Path = MODELS_DIR / "semantic_search" / "bm25_model.pkl"
    SEARCH_INFO_PATH: Path = MODELS_DIR / "semantic_search" / "model_info.json"

    # ✅ Knowledge Base & Templates
    KB_PATH: Path = DATA_DIR / "kb_processed.json"
    TEMPLATES_PATH: Path = DATA_DIR / "response_templates.json"

    # ✅ Thresholds & Hyperparameters
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 16
    INTENT_THRESHOLD: float = 0.85
    NER_THRESHOLD: float = 0.80
    SEARCH_THRESHOLD: float = 0.70
    FALLBACK_THRESHOLD: float = 0.40

    # ✅ CORS & Cache
    ALLOWED_ORIGINS: List[str] = ["*"]
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL: int = 3600
    ENABLE_CACHE: bool = True

    # ✅ Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ✅ System & Monitoring
    MAX_WORKERS: int = multiprocessing.cpu_count()
    ENABLE_MONITORING: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # ✅ Izinkan field tambahan di .env


# ✅ Global instance (langsung dipanggil dengan `from app.core.config import settings`)
settings = Settings()

# ✅ Alias untuk kompatibilitas Step 9 Final
def get_settings() -> Settings:
    logger.debug("🔧 get_settings() dipanggil – menggunakan global settings instance")
    return settings

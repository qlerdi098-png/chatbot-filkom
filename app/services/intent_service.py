# ===================================================================
# FILE: app/services/intent_service.py (FINAL – Hotfix 1.9, Python 3.9 Compatible)
# ===================================================================
import logging
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from functools import lru_cache

from app.core.config import settings
from app.core.exceptions import IntentClassificationError, ModelLoadError

logger = logging.getLogger(__name__)

# ======================= INTENT CLASSIFIER =========================
class IntentClassifier(nn.Module):
    """IndoBERT-based Intent Classification Model (sesuai training)"""
    def __init__(self, model_name: str, num_classes: int, dropout_rate: float = 0.3):
        super(IntentClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# ======================= INTENT SERVICE ============================
class IntentClassificationService:
    def __init__(self):
        self.model: Optional[IntentClassifier] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.intent_mapping: Dict[int, str] = {}
        self.reverse_mapping: Dict[str, int] = {}

        self.model_name = "indolem/indobert-base-uncased"
        self.max_length = settings.MAX_LENGTH
        self.device = torch.device(settings.DEVICE)
        self.confidence_threshold = settings.INTENT_THRESHOLD

        self.model_path = settings.INTENT_MODEL_PATH
        self.mapping_path = settings.INTENT_MAPPING_PATH

        self.is_loaded = False
        self.load_time = 0.0

        # ✅ Cache Hotfix 1.9
        self._cached_status: Optional[Dict[str, Any]] = None
        self._cached_classes: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------
    def load_model(self) -> bool:
        try:
            import time
            start_time = time.time()

            logger.info("🎯 Loading Intent Classification Model...")
            self._load_intent_mapping()
            self._load_tokenizer()
            self._load_model_weights()

            self.load_time = round(time.time() - start_time, 3)
            self.is_loaded = True
            self._set_cache()

            logger.info(f"✅ Intent Model loaded in {self.load_time:.2f}s | Classes: {len(self.intent_mapping)}")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading Intent Classification Model: {str(e)}")
            self.is_loaded = False
            raise ModelLoadError(f"Failed to load intent classification model: {str(e)}")

    def _load_intent_mapping(self):
        correct_mapping = {
            0: "BATAS_SKS", 1: "CLARIFICATION", 2: "DOSEN_PENGAMPU", 3: "GOODBYE", 4: "GREETING",
            5: "HELP", 6: "INFO_DOSEN_UMUM", 7: "INFO_MATAKULIAH", 8: "JADWAL_DOSEN", 9: "JADWAL_HARI",
            10: "JADWAL_KRS", 11: "JADWAL_MATAKULIAH", 12: "JADWAL_PRODI", 13: "JADWAL_RUANGAN",
            14: "JADWAL_SEMESTER", 15: "KONTAK_DOSEN", 16: "NIDN_DOSEN", 17: "OUT_OF_SCOPE",
            18: "PANDUAN_KRS", 19: "PRASYARAT_MATKUL", 20: "PROSEDUR_CUTI", 21: "SKS_MATKUL",
            22: "SYARAT_SKRIPSI"
        }
        self.intent_mapping = correct_mapping
        self.reverse_mapping = {v: k for k, v in correct_mapping.items()}
        logger.info(f"📋 Loaded intent mapping ({len(correct_mapping)} classes)")

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        logger.info("🔤 Tokenizer loaded")

    def _load_model_weights(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        self.model = IntentClassifier(self.model_name, num_classes=len(self.intent_mapping))
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("🏗️ Model weights loaded successfully")

    # ---------------------------------------------------------------
    def _set_cache(self):
        self._cached_status = {
            "status": "ready" if self.is_loaded else "not_loaded",
            "model_info": {
                "model_name": self.model_name,
                "num_classes": len(self.intent_mapping),
                "device": str(self.device),
                "confidence_threshold": self.confidence_threshold,
                "load_time": self.load_time
            },
            "cached": True
        }
        self._cached_classes = {
            "intent_classes": list(self.reverse_mapping.keys()),
            "total_classes": len(self.reverse_mapping),
            "cached": True
        }
        logger.info("✅ Cached intent status & classes updated")

    # ---------------------------------------------------------------
    def _preprocess_text(self, text: str) -> str:
        return ' '.join(text.strip().split()) if isinstance(text, str) else ""

    @lru_cache(maxsize=512)
    def _cached_predict(self, processed_text: str) -> Tuple[str, float, Dict[str, float]]:
        inputs = self.tokenizer(
            processed_text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt", add_special_tokens=True, return_token_type_ids=False
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        confidence, predicted_class = torch.max(probs, dim=-1)
        prob_dict = {self.intent_mapping[i]: probs[i].item() for i in range(len(self.intent_mapping))}
        return self.intent_mapping.get(predicted_class.item(), "UNKNOWN"), confidence.item(), prob_dict

    def predict_intent(self, text: str) -> Tuple[str, float]:
        if not self.is_loaded:
            raise IntentClassificationError("Model not loaded.")
        processed_text = self._preprocess_text(text)
        intent, confidence, _ = self._cached_predict(processed_text)
        return intent, confidence

    def predict_intent_with_probabilities(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_loaded:
            raise IntentClassificationError("Model not loaded.")
        processed_text = self._preprocess_text(text)
        return self._cached_predict(processed_text)

    def is_confident_prediction(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold

    def get_model_info(self) -> Dict[str, Any]:
        return self._cached_status if self._cached_status else {
            "status": "not_loaded", "model_info": {}
        }

    def get_cached_classes(self) -> Dict[str, Any]:
        return self._cached_classes if self._cached_classes else {
            "intent_classes": [], "total_classes": 0
        }

# ======================= GLOBAL INSTANCE ===========================
intent_service = IntentClassificationService()

def initialize_intent_service() -> bool:
    try:
        return intent_service.load_model()
    except Exception:
        return False

def get_intent_service() -> IntentClassificationService:
    if not intent_service.is_loaded:
        raise IntentClassificationError("Intent service not initialized.")
    return intent_service

def is_intent_service_ready() -> bool:
    return intent_service.is_loaded

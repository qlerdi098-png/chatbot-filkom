# ===================================================================
# FILE: app/services/ner_service.py (FINAL – Hotfix 1.9, Python 3.9 Compatible)
# ===================================================================

import json
import logging
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional
import time

from app.core.config import settings

logger = logging.getLogger(__name__)

class IndoBERTBiLSTMCRF(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(IndoBERTBiLSTMCRF, self).__init__()
        self.bert = AutoModel.from_pretrained(config["model_name"])
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=config["lstm_hidden_dim"],
            num_layers=config["lstm_num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=config["lstm_dropout"] if config["lstm_num_layers"] > 1 else 0
        )
        self.linear = nn.Linear(config["lstm_hidden_dim"] * 2, config["num_labels"])
        self.dropout = nn.Dropout(config["lstm_dropout"])
        self.crf = CRF(config["num_labels"], batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)
        lstm_output = self.dropout(lstm_output)
        logits = self.linear(lstm_output)
        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.bool())
            return loss, logits
        predictions = self.crf.decode(logits, mask=attention_mask.bool())
        return logits, predictions

class NERService:
    def __init__(self):
        self.model: Optional[IndoBERTBiLSTMCRF] = None
        self.tokenizer = None
        self.config: Optional[Dict[str, Any]] = None
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[str, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.CUDA_AVAILABLE else "cpu")
        self.is_loaded = False
        self.load_time: Optional[float] = None

        self.model_path = settings.NER_MODEL_PATH
        self.config_path = settings.NER_CONFIG_PATH
        self.confidence_threshold: float = settings.NER_THRESHOLD

        self._cached_status: Optional[Dict[str, Any]] = None
        self._cached_labels: Optional[Dict[str, Any]] = None

    def load_model(self) -> bool:
        try:
            start_time = time.time()
            logger.info("🤖 Loading NER Model (IndoBERT+BiLSTM+CRF)...")

            if not self.config_path.exists():
                logger.error(f"❌ Config file not found: {self.config_path}")
                return False

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            self.label2id = self.config.get("label2id", {})
            self.id2label = self.config.get("id2label", {})
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

            self.model = IndoBERTBiLSTMCRF(self.config)
            if not self.model_path.exists():
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False

            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            if "classifier.weight" in state_dict:
                state_dict["linear.weight"] = state_dict.pop("classifier.weight")
            if "classifier.bias" in state_dict:
                state_dict["linear.bias"] = state_dict.pop("classifier.bias")

            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            self.load_time = round(time.time() - start_time, 2)
            self.is_loaded = True
            self._set_cache()

            logger.info(f"✅ NER Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load NER model: {str(e)}")
            return False

    def _set_cache(self):
        entity_types = {
            (label.split("-")[1] if "-" in label else label)
            for label in self.id2label.values() if label != "O"
        }
        self._cached_labels = {
            "total_labels": len(self.id2label),
            "entity_types": sorted(list(entity_types)),
            "full_labels": list(self.id2label.values()),
            "bio_format": True,
            "model_architecture": "IndoBERT+BiLSTM+CRF",
            "labels_cached": True
        }
        self._cached_status = {
            "status": "ready" if self.is_loaded else "not_loaded",
            "model_info": {
                "model_name": self.config.get("model_name") if self.config else None,
                "labels": list(self.id2label.values()),
                "device": str(self.device),
                "load_time": self.load_time,
                "labels_cached": True
            }
        }
        logger.info("✅ Cached NER status & labels updated")

    def get_status(self) -> Dict[str, Any]:
        return self._cached_status if self._cached_status else {"status": "not_loaded", "model_info": {}}

    def get_labels_cached(self) -> Dict[str, Any]:
        return self._cached_labels if self._cached_labels else {
            "total_labels": 0, "entity_types": [], "full_labels": []
        }

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("NER model not loaded.")
        try:
            start_time = time.time()
            encoding = self.tokenizer(
                text, truncation=True, padding=True,
                max_length=self.config["max_len"], return_tensors="pt",
                return_offsets_mapping=True
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            offset_mapping = encoding["offset_mapping"][0]

            with torch.no_grad():
                logits, predictions = self.model(input_ids, attention_mask)

            predicted_labels = [self.id2label[str(p)] for p in predictions[0]]
            entities = self._convert_predictions_to_entities(text, predicted_labels, logits[0], offset_mapping)
            return {
                "text": text,
                "entities": entities,
                "entity_count": len(entities),
                "prediction_time": round(time.time() - start_time, 4)
            }
        except Exception as e:
            logger.error(f"❌ NER prediction error: {str(e)}")
            return {"entities": [], "error": str(e)}

    def _convert_predictions_to_entities(self, text: str, labels: List[str], logits: torch.Tensor, offset_mapping):
        entities, current_entity = [], None
        for i, (label, offset) in enumerate(zip(labels, offset_mapping)):
            if offset[0] == offset[1]:
                continue
            confidence = torch.softmax(logits[i], dim=-1)[self.label2id.get(label, 0)].item()

            if label == "O" or confidence < self.confidence_threshold:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": label[2:], "value": text[offset[0]:offset[1]],
                    "start": offset[0].item(), "end": offset[1].item(), "confidence": confidence
                }
            elif label.startswith("I-") and current_entity and current_entity["entity"] == label[2:]:
                current_entity["end"] = offset[1].item()
                current_entity["value"] = text[current_entity["start"]:offset[1]]
                current_entity["confidence"] = (current_entity["confidence"] + confidence) / 2

        if current_entity:
            entities.append(current_entity)
        return entities

    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        result = self.predict(text)
        grouped = {}
        for ent in result.get("entities", []):
            grouped.setdefault(ent["entity"], []).append(ent["value"])
        return grouped

ner_service = NERService()

def initialize_ner_service() -> bool:
    return ner_service.load_model()

def get_ner_service() -> NERService:
    return ner_service

def is_ner_ready() -> bool:
    return ner_service.is_loaded

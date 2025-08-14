# ===================================================================
# FILE: app/services/chat_pipeline.py
# ===================================================================

import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
from rapidfuzz import process, fuzz

from app.services.intent_service import intent_service
from app.services.ner_service import ner_service
from app.services.kb_service import get_kb_query
from app.services.semantic_search_service import search_service
from app.services.template_service import TemplateService
from app.core.exceptions import ChatbotException

logger = logging.getLogger(__name__)

# ======================
# DATA CLASSES
# ======================

@dataclass
class ChatContext:
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    last_intent: Optional[str] = None
    last_entities: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class PipelineResult:
    response: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    search_results: Optional[List[Dict]] = None
    template_used: Optional[str] = None
    fallback_reason: Optional[str] = None
    processing_time: Optional[float] = None
    cached: bool = False

# ======================
# CHAT PIPELINE
# ======================

class ChatPipeline:
    def __init__(self):
        self.intent_service = intent_service
        self.ner_service = ner_service
        self.kb_service = get_kb_query()
        self.semantic_search = search_service
        self.template_service = TemplateService()

        self.contexts: Dict[str, ChatContext] = {}
        self.intent_threshold = 0.85
        self.fallback_threshold = 0.5

        self._cache: Dict[str, PipelineResult] = {}

        logger.info("✅ Chat Pipeline initialized successfully (Hotfix 2.4.4)")

    def get_or_create_context(self, user_id: str, session_id: str) -> ChatContext:
        context_key = f"{user_id}_{session_id}"
        if context_key not in self.contexts:
            self.contexts[context_key] = ChatContext(
                user_id=user_id,
                session_id=session_id,
                conversation_history=[]
            )
        return self.contexts[context_key]

    def process_message(
        self,
        message: str,
        user_id: str = "default",
        session_id: str = "default"
    ) -> PipelineResult:
        """Main pipeline processing (safe & alias matched)"""
        start_time = time.time()

        if not message or not isinstance(message, str):
            raise ChatbotException("Invalid input text.", details={"message": message})

        context = self.get_or_create_context(user_id, session_id)

        cache_key = f"{message.strip().lower()}::{user_id}::{session_id}"
        if cache_key in self._cache:
            logger.debug(f"⚡ [CACHE HIT] Returning cached result for: {message}")
            cached_result = self._cache[cache_key]
            cached_result.cached = True
            cached_result.processing_time = round(time.time() - start_time, 4)
            return cached_result

        logger.debug(f"⚡ [CACHE MISS] Processing new message: {message}")

        try:
            intent, intent_confidence = self.intent_service.predict_intent(message)
            logger.info(f"[PIPELINE] Intent: {intent} ({intent_confidence:.3f})")

            entities = self.ner_service.extract_key_entities(message)
            logger.info(f"[PIPELINE] Entities (raw): {entities}")

            # ✅ Normalisasi alias & list-safe
            entities = self._normalize_entities_alias(intent, entities)
            logger.info(f"[PIPELINE] Entities (alias-normalized): {entities}")

            response, search_results, fallback_reason = "", None, None

            if intent in ["GREETING", "HELP", "GOODBYE", "CLARIFICATION"]:
                response = self.template_service.fill_template(intent, entities)
            elif intent in ["PANDUAN_KRS", "PROSEDUR_CUTI"]:
                response, search_results = self._generate_search_response(message, entities)
                fallback_reason = f"Intent panjang ({intent}), langsung semantic search"
            elif intent_confidence >= self.intent_threshold:
                response = self._generate_template_response(intent, entities, message)
            elif intent_confidence >= self.fallback_threshold and self.semantic_search.is_loaded:
                response, search_results = self._generate_search_response(message, entities)
                fallback_reason = f"Low intent confidence ({intent_confidence:.3f}), using semantic search"
            else:
                response = self._generate_fallback_response(message)
                fallback_reason = f"Very low intent confidence ({intent_confidence:.3f})"

            self._update_context(context, message, intent, entities, response)

            processing_time = round(time.time() - start_time, 4)
            result = PipelineResult(
                response=response,
                intent=intent,
                entities=entities,
                confidence=round(intent_confidence, 4),
                search_results=search_results,
                template_used=intent if intent in self.template_service.templates else None,
                fallback_reason=fallback_reason,
                processing_time=processing_time,
                cached=False
            )

            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"[PIPELINE ERROR]: {e}")
            return PipelineResult(
                response="Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi.",
                intent="ERROR",
                entities={},
                confidence=0.0,
                fallback_reason=str(e)
            )

    # ======================
    # ENTITY NORMALIZATION (ALIAS SUPPORT)
    # ======================

    def _normalize_entities_alias(self, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Final fix: konversi list ke string & fuzzy match ke key+alias KB"""
        try:
            normalized = {}
            kb = self.kb_service.kb

            def to_str(val: Any) -> str:
                if isinstance(val, list) and val:
                    return str(val[0])
                return str(val or "")

            all_matkul_keys = list(kb.matakuliah_data.keys())
            for mk, info in kb.matakuliah_data.items():
                if info.alias and "mata_kuliah" in info.alias:
                    all_matkul_keys.extend([a.lower() for a in info.alias["mata_kuliah"]])

            all_dosen_keys = list(kb.dosen_data.keys())
            for ds, info in kb.dosen_data.items():
                if info.alias and "nama_lengkap" in info.alias:
                    all_dosen_keys.extend([a.lower() for a in info.alias["nama_lengkap"]])

            for k, v in entities.items():
                val = to_str(v).strip()
                if not val:
                    normalized[k] = ""
                    continue

                if k == "MATA_KULIAH":
                    best = process.extractOne(val.lower(), all_matkul_keys, scorer=fuzz.ratio)
                    normalized[k] = best[0] if best and best[1] >= 75 else val
                elif k == "DOSEN":
                    clean = val.lower().replace("dosen", "").replace("pak", "").replace("bu", "").strip()
                    best = process.extractOne(clean, all_dosen_keys, scorer=fuzz.ratio)
                    normalized[k] = best[0] if best and best[1] >= 75 else val
                else:
                    normalized[k] = val

            return normalized
        except Exception as e:
            logger.warning(f"[ALIAS NORMALIZE ERROR] {e}")
            return entities

    # ======================
    # RESPONSE GENERATION
    # ======================

    def _generate_template_response(self, intent: str, entities: Dict[str, Any], original_message: str) -> str:
        try:
            return self.template_service.fill_template(intent=intent, entities=entities, search_results=None)
        except Exception as e:
            logger.warning(f"[TEMPLATE FALLBACK] {e}")
            return self._generate_search_response(original_message, entities)[0]

    def _generate_search_response(self, message: str, entities: Dict[str, Any]) -> Tuple[str, List[Dict]]:
        try:
            search_output = self.semantic_search.search(query=message, search_type="hybrid", top_k=3)
            search_results = search_output.get("results", [])
            if search_results:
                best_result = search_results[0]
                response = f"Berdasarkan informasi yang saya temukan: {best_result['text']}"
                if entities:
                    entity_info = ", ".join([f"{k}: {v}" for k, v in entities.items()])
                    response += f"\n\n(Entitas terdeteksi: {entity_info})"
                return response, search_results
            return self._generate_fallback_response(message), []
        except Exception as e:
            logger.error(f"[SEARCH ERROR] {e}")
            return self._generate_fallback_response(message), []

    def _generate_fallback_response(self, message: str) -> str:
        fallback_responses = [
            "Maaf, saya belum memahami pertanyaan Anda. Bisa dijelaskan lebih spesifik?",
            "Saya belum bisa menjawab pertanyaan tersebut. Coba tanyakan tentang jadwal kuliah, dosen, atau mata kuliah.",
            "Pertanyaan Anda di luar pemahaman saya saat ini. Silakan tanyakan informasi akademik FILKOM.",
            "Maaf, saya membutuhkan informasi yang lebih jelas. Coba tanyakan tentang kurikulum, jadwal, atau dosen."
        ]
        hash_val = int(hashlib.md5(message.encode()).hexdigest(), 16)
        return fallback_responses[hash_val % len(fallback_responses)]

    # ======================
    # CONTEXT HANDLING
    # ======================

    def _update_context(self, context: ChatContext, message: str, intent: str, entities: Dict[str, Any], response: str):
        context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "bot_response": response,
            "intent": intent,
            "entities": entities
        })
        context.last_intent = intent
        context.last_entities = entities
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]

    def get_conversation_history(self, user_id: str, session_id: str) -> List[Dict]:
        context_key = f"{user_id}_{session_id}"
        return self.contexts.get(context_key, ChatContext(user_id, session_id, [])).conversation_history

    def clear_context(self, user_id: str, session_id: str):
        context_key = f"{user_id}_{session_id}"
        self.contexts.pop(context_key, None)

# ===================================================================
# FILE: app/services/__init__.py (FINAL – Production-Ready)
# ===================================================================
"""
Services package initializer.

Menyediakan akses global ke semua service utama (KB, Intent, NER, Semantic Search).
Direkomendasikan hanya untuk import di level aplikasi utama (main.py) atau API,
bukan di dalam service lain agar menghindari circular import.
"""

# Knowledge Base Service
from .kb_service import kb_loader, initialize_knowledge_base, get_kb_query

# Intent Classification Service
from .intent_service import intent_service, initialize_intent_service

# Named Entity Recognition (NER) Service
from .ner_service import ner_service, initialize_ner_service

# Semantic Search Service
from .semantic_search_service import search_service

__all__ = [
    "kb_loader",
    "initialize_knowledge_base",
    "get_kb_query",
    "intent_service",
    "initialize_intent_service",
    "ner_service",
    "initialize_ner_service",
    "search_service",
]

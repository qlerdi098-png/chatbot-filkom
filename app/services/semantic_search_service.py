# ===================================================================
# FILE: app/services/semantic_search_service.py (CLEAN – No Warning Version)
# ===================================================================

import json
import pickle
import numpy as np
import faiss
import re
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
from pathlib import Path
import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# Download stopwords hanya jika belum ada
try:
    _ = stopwords.words("indonesian")
except LookupError:
    nltk.download("stopwords", quiet=True)


class SemanticSearchService:
    """
    Hybrid Semantic Search Service menggunakan BM25 + MiniLM
    Sinkron dengan hasil training di Colab (FAISS + BM25).
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path: Path = Path(config_path or "models/semantic_search")
        self.is_loaded: bool = False

        # Model components
        self.sentence_model: Optional[SentenceTransformer] = None
        self.bm25_model: Optional[BM25Okapi] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

        # Stopwords & preprocessing
        self.stop_words: set[str] = set(stopwords.words("indonesian"))

        # Config sinkron dengan training
        self.config: Dict[str, Any] = {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "max_length": 512,
            "top_k_bm25": 10,
            "top_k_semantic": 10,
            "hybrid_weights": [0.4, 0.6],
            "confidence_threshold": 0.3
        }

    # ================================================================
    # ✅ Preprocessing Text
    # ================================================================
    def preprocess_text(self, text: str) -> str:
        """Lowercase, remove special chars, dan filter stopwords."""
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return " ".join([t for t in text.split() if t and t not in self.stop_words])

    # ================================================================
    # ✅ Load Models
    # ================================================================
    def load_models(self) -> bool:
        try:
            if not self.config_path.exists():
                logger.error(f"❌ Config path not found: {self.config_path}")
                return False

            # 1. Load SentenceTransformer
            logger.info("🔄 Loading SentenceTransformer model...")
            self.sentence_model = SentenceTransformer(self.config["model_name"])

            # 2. Load BM25
            bm25_path = self.config_path / "bm25_model.pkl"
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    self.bm25_model = pickle.load(f)
            else:
                logger.error(f"❌ BM25 model not found at {bm25_path}")
                return False

            # 3. Load FAISS
            faiss_path = self.config_path / "faiss_index.bin"
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
            else:
                logger.error(f"❌ FAISS index not found at {faiss_path}")
                return False

            # 4. Load embeddings (optional)
            embeddings_path = self.config_path / "embeddings.npy"
            if embeddings_path.exists():
                self.embeddings = np.load(embeddings_path)
            else:
                logger.warning(f"⚠️ Embeddings not found, using FAISS only.")

            # 5. Load documents & metadata
            docs_path = self.config_path / "documents.json"
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.documents = [doc.get("konten", "") for doc in data]
                    self.metadata = data
                else:
                    self.documents = data.get("documents", [])
                    self.metadata = data.get("metadata", [])
            else:
                logger.warning("⚠️ documents.json not found, returning only indices.")

            # 6. Load custom config jika ada
            config_file = self.config_path / "search_config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    self.config.update(json.load(f))

            self.is_loaded = True
            logger.info(f"✅ Semantic Search Service loaded ({len(self.documents)} documents, FAISS: {self.faiss_index.ntotal})")
            return True

        except Exception as e:
            logger.error(f"❌ Error loading semantic search models: {e}")
            self.is_loaded = False
            return False

    # ================================================================
    # ✅ Search Functions
    # ================================================================
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.bm25_model:
            return []
        tokens = self.preprocess_text(query).split()
        scores = self.bm25_model.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in indices if scores[i] > 0]

    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.sentence_model or not self.faiss_index:
            return []
        query_emb = self.sentence_model.encode([self.preprocess_text(query)])
        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb.astype("float32"), top_k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0])) if scores[0][i] > 0]

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            return []
        try:
            bm25_res = self.bm25_search(query, self.config["top_k_bm25"])
            semantic_res = self.semantic_search(query, self.config["top_k_semantic"])
            scores: Dict[int, float] = {}

            if bm25_res:
                max_bm25 = max(s for _, s in bm25_res)
                for idx, sc in bm25_res:
                    scores[idx] = scores.get(idx, 0) + (sc / max_bm25 if max_bm25 > 0 else 0) * self.config["hybrid_weights"][0]

            if semantic_res:
                max_sem = max(s for _, s in semantic_res)
                for idx, sc in semantic_res:
                    scores[idx] = scores.get(idx, 0) + (sc / max_sem if max_sem > 0 else 0) * self.config["hybrid_weights"][1]

            return [
                {
                    "document_id": idx,
                    "text": self.documents[idx] if self.documents else f"Document #{idx}",
                    "metadata": self.metadata[idx] if self.metadata else {},
                    "score": float(score),
                    "confidence": min(score, 1.0)
                }
                for idx, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
                if score >= self.config["confidence_threshold"]
            ][:top_k]

        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return []

    # ================================================================
    # ✅ Public Interface
    # ================================================================
    def search(self, query: str, search_type: str = "hybrid", top_k: int = 5) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Models not loaded", "results": []}
        try:
            if search_type == "bm25":
                raw = self.bm25_search(query, top_k)
            elif search_type == "semantic":
                raw = self.semantic_search(query, top_k)
            else:
                return {
                    "success": True,
                    "query": query,
                    "search_type": "hybrid",
                    "results": self.hybrid_search(query, top_k),
                    "total_found": len(self.hybrid_search(query, top_k))
                }

            results = [
                {
                    "document_id": i,
                    "text": self.documents[i] if self.documents else f"Document #{i}",
                    "metadata": self.metadata[i] if self.metadata else {},
                    "score": float(sc),
                    "confidence": min(sc / 10 if search_type == "bm25" else sc, 1.0)
                } for i, sc in raw
            ]
            return {"success": True, "query": query, "search_type": search_type, "results": results, "total_found": len(results)}

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return {"success": False, "error": str(e), "results": []}

    def get_status(self) -> Dict[str, Any]:
        return {
            "service": "Semantic Search Service",
            "loaded": self.is_loaded,
            "model_path": str(self.config_path),
            "documents_count": len(self.documents),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "config": self.config
        }


# ======================= GLOBAL INSTANCE ===========================
search_service = SemanticSearchService()

def initialize_search_service() -> bool:
    return search_service.load_models()

def is_search_ready() -> bool:
    return search_service.is_loaded

# ===================================================================
# FILE: scripts/setup_semantic_search.py (FINAL)
# ===================================================================
"""
Script untuk setup dan training Semantic Search models
Sinkron dengan kb_service (flat=True) & semantic_search_service.py
"""

import os
import sys
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
import re
import nltk
from nltk.corpus import stopwords

# Project root
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ✅ FIXED: Import langsung kb_loader (bukan kb_service)
from app.services.kb_service import kb_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords jika belum ada
nltk.download("stopwords", quiet=True)


class SemanticSearchTrainer:
    """
    Training pipeline untuk Semantic Search (BM25 + MiniLM Hybrid)
    """

    def __init__(self, output_dir: str = "models/semantic_search"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "max_length": 512,
            "top_k_bm25": 10,
            "top_k_semantic": 10,
            "hybrid_weights": [0.4, 0.6],
            "confidence_threshold": 0.3,
            "version": "1.0.0",
            "created_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        }

        self.stop_words = set(stopwords.words("indonesian"))
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.sentence_model = None
        self.bm25_model = None
        self.faiss_index = None

    # ================================================================
    # ✅ Preprocessing (sinkron semantic_search_service.py)
    # ================================================================
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t not in self.stop_words]
        return " ".join(tokens)

    # ================================================================
    # ✅ Load & Process KB Data
    # ================================================================
    def load_knowledge_base(self):
        logger.info("📚 Loading knowledge base...")
        if not kb_loader.is_loaded:
            kb_loader.load_knowledge_base()

        kb_data = kb_loader.get_all_data(flat=True)
        if not kb_data:
            raise ValueError("❌ No knowledge base data found!")
        logger.info(f"✅ Loaded KB with {len(kb_data)} entries")
        return kb_data

    def process_documents(self, kb_data: list):
        logger.info("📄 Processing documents for training...")
        documents, metadata = [], []

        for item in kb_data:
            text = self.preprocess_text(
                f"{item.get('judul', '')} | {item.get('konten', '')}"
            )
            meta_info = {
                "id": item.get("id"),
                "kategori": item.get("kategori", "UNKNOWN"),
                "source": item.get("source", "")
            }
            documents.append(text)
            metadata.append(meta_info)

        self.documents = documents
        self.metadata = metadata
        logger.info(f"✅ Processed {len(documents)} documents")
        return documents, metadata

    # ================================================================
    # ✅ Train BM25
    # ================================================================
    def train_bm25(self):
        logger.info("🔍 Training BM25 model...")
        tokenized_docs = [doc.split() for doc in self.documents]
        self.bm25_model = BM25Okapi(tokenized_docs)
        logger.info("✅ BM25 model trained")

    # ================================================================
    # ✅ Train Semantic Model
    # ================================================================
    def train_semantic(self):
        logger.info("🤖 Loading SentenceTransformer model...")
        self.sentence_model = SentenceTransformer(self.config["model_name"])
        logger.info("🔮 Creating embeddings...")
        self.embeddings = self.sentence_model.encode(
            self.documents, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info(f"✅ Created embeddings: {self.embeddings.shape}")

    # ================================================================
    # ✅ Create FAISS Index
    # ================================================================
    def create_faiss_index(self):
        logger.info("🗂️ Creating FAISS index...")
        normalized_embeddings = self.embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        dimension = normalized_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(normalized_embeddings.astype("float32"))
        logger.info(f"✅ FAISS index created with {self.faiss_index.ntotal} vectors")

    # ================================================================
    # ✅ Save Models
    # ================================================================
    def save_models(self):
        logger.info("💾 Saving models...")

        with open(self.output_dir / "bm25_model.pkl", "wb") as f:
            pickle.dump(self.bm25_model, f)
        faiss.write_index(self.faiss_index, str(self.output_dir / "faiss_index.bin"))
        np.save(self.output_dir / "embeddings.npy", self.embeddings)

        with open(self.output_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(
                {"documents": self.documents, "metadata": self.metadata},
                f, ensure_ascii=False, indent=2
            )

        self.config["embedding_dimension"] = self.embeddings.shape[1]
        self.config["total_documents"] = len(self.documents)
        with open(self.output_dir / "search_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"✅ All models saved in {self.output_dir}")

    # ================================================================
    # ✅ Train All Pipeline
    # ================================================================
    def train_all(self):
        logger.info("🚀 Starting Semantic Search Training Pipeline...")
        try:
            kb_data = self.load_knowledge_base()
            self.process_documents(kb_data)
            self.train_bm25()
            self.train_semantic()
            self.create_faiss_index()
            self.save_models()
            logger.info("🎉 Training completed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return False


# ================================================================
# ✅ Main Function
# ================================================================
def main():
    print("🤖 SEMANTIC SEARCH MODEL TRAINING")
    print("=" * 40)

    trainer = SemanticSearchTrainer()

    model_files = [
        "bm25_model.pkl", "faiss_index.bin", "embeddings.npy", "documents.json"
    ]
    existing = [f for f in model_files if (trainer.output_dir / f).exists()]

    if existing:
        print(f"⚠️ Found existing model files: {existing}")
        response = input("Do you want to retrain? (y/N): ").lower().strip()
        if response != "y":
            print("❌ Training cancelled")
            return

    success = trainer.train_all()
    if success:
        print("\n✅ SUCCESS!")
        print(f"Models location: {trainer.output_dir}")
        print("1. Start FastAPI server")
        print("2. Test endpoints: /api/v1/search/status, /api/v1/search/demo, /api/v1/search/query")
    else:
        print("\n❌ FAILED! Check logs for details")


if __name__ == "__main__":
    main()

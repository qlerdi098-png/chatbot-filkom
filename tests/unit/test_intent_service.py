# tests/unit/test_intent_service.py

import pytest
import time
from app.services.intent_service import IntentClassificationService

class TestIntentService:

    @pytest.fixture(scope="class")
    def intent_service(self):
        service = IntentClassificationService()
        service.load_model()
        return service

    def test_intent_model_loading(self, intent_service):
        """Test apakah model dan tokenizer berhasil dimuat"""
        assert intent_service.model is not None
        assert intent_service.tokenizer is not None
        assert len(intent_service.intent_mapping) == 23
        print("✅ test_intent_model_loading PASSED")

    def test_intent_prediction_confidence(self, intent_service):
        """Test confidence dan label hasil prediksi"""
        result = intent_service.predict_intent_with_probabilities("Jadwal kuliah machine learning kapan?")
        assert 0.0 <= result[1] <= 1.0
        assert result[0] in intent_service.intent_mapping.values()
        print(f"✅ test_intent_prediction_confidence PASSED (confidence: {result[1]:.3f})")

    def test_intent_batch_processing(self, intent_service):
        """Test batch prediction"""
        queries = [
            "Siapa dosen pengampu Machine Learning?",
            "Syarat mengambil skripsi apa saja?",
            "Berapa SKS Algoritma dan Pemrograman?"
        ]
        start_time = time.time()
        results = [intent_service.predict_intent(q) for q in queries]
        duration = time.time() - start_time

        assert len(results) == len(queries)
        assert all(isinstance(r[0], str) and 0.0 <= r[1] <= 1.0 for r in results)
        print(f"✅ test_intent_batch_processing PASSED ({duration:.3f}s)")

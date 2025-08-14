# ===================================================================
# FILE: tests/test_integration.py (FINAL – Production-Ready)
# ===================================================================
"""
Integration Test untuk Chatbot FILKOM (Intent + NER + KB + Semantic Search).

Menjalankan pengujian end-to-end pada ChatPipeline untuk memastikan:
1. Intent terdeteksi sesuai harapan.
2. Entitas terdeteksi sesuai harapan.
3. Confidence di atas threshold minimal.
4. Response dihasilkan tanpa error.

Run Command:
(venv) PS D:\chatbot-filkom> python tests/test_integration.py
"""

import sys, os
import asyncio
from typing import Dict, Any

# ✅ Pastikan path aplikasi terdeteksi (untuk menghindari ImportError)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===================================================================
# MAIN TEST FUNCTION
# ===================================================================
async def run_integration_tests():
    """Comprehensive integration test untuk chat pipeline"""
    print("🧪 RUNNING INTEGRATION TESTS")
    print("=" * 50)

    from app.services.chat_pipeline import ChatPipeline

    pipeline = ChatPipeline()

    test_cases = [
        {
            "name": "Intent Classification Test",
            "query": "Siapa dosen pengampu Machine Learning?",
            "expected_intent": "DOSEN_PENGAMPU",
            "expected_entities": ["MATA_KULIAH"]
        },
        {
            "name": "Schedule Query Test",
            "query": "Jadwal kuliah Basis Data hari apa?",
            "expected_intent": "JADWAL_MATAKULIAH",
            "expected_entities": ["MATA_KULIAH"]
        },
        {
            "name": "Contact Information Test",
            "query": "Kontak dosen Hendry Fonda",
            "expected_intent": "KONTAK_DOSEN",
            "expected_entities": ["DOSEN"]
        },
        {
            "name": "Course Information Test",
            "query": "Berapa SKS mata kuliah Algoritma dan Pemrograman?",
            "expected_intent": "SKS_MATKUL",
            "expected_entities": ["MATA_KULIAH"]
        },
        {
            "name": "Requirements Test",
            "query": "Syarat untuk mengambil skripsi",
            "expected_intent": "SYARAT_SKRIPSI",
            "expected_entities": []
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[TEST {i}] {test_case['name']}")
        print(f"Query: {test_case['query']}")

        try:
            result = await pipeline.process_message(
                test_case["query"],
                "integration_user",
                f"session_{i}"
            )

            # ✅ Intent check (case-insensitive)
            intent_correct = result.intent.upper() == test_case["expected_intent"].upper()

            # ✅ Entities check (case-insensitive)
            entities_detected = [k.upper() for k in result.entities.keys()]
            entities_correct = all(ent.upper() in entities_detected for ent in test_case["expected_entities"])

            # ✅ Success condition (intent + entities + confidence threshold)
            success = intent_correct and entities_correct and result.confidence > 0.5

            results.append({
                "test_name": test_case["name"],
                "success": success,
                "intent_correct": intent_correct,
                "entities_correct": entities_correct,
                "confidence": result.confidence,
                "processing_time": result.processing_time
            })

            status = "✅ PASS" if success else "❌ FAIL"
            print(f"Status        : {status}")
            print(f"Intent        : {result.intent} ({'✓' if intent_correct else '✗'})")
            print(f"Entities      : {entities_detected} ({'✓' if entities_correct else '✗'})")
            print(f"Confidence    : {result.confidence:.3f}")
            print(f"Response      : {result.response[:100]}...")
            print(f"Processing    : {result.processing_time:.3f}s")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                "test_name": test_case["name"],
                "success": False,
                "error": str(e)
            })

    # ✅ Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.get("success", False))

    avg_confidence = (
        sum(r.get("confidence", 0) for r in results if r.get("success")) / max(passed_tests, 1)
    )
    avg_processing_time = (
        sum(r.get("processing_time", 0) for r in results if r.get("success")) / max(passed_tests, 1)
    )

    print(f"Total Tests   : {total_tests}")
    print(f"Passed        : {passed_tests}")
    print(f"Failed        : {total_tests - passed_tests}")
    print(f"Success Rate  : {(passed_tests/total_tests)*100:.1f}%")
    print(f"Avg Confidence: {avg_confidence:.3f}")
    print(f"Avg Proc Time : {avg_processing_time:.3f}s")

    return results

# ===================================================================
# ENTRY POINT
# ===================================================================
if __name__ == "__main__":
    asyncio.run(run_integration_tests())

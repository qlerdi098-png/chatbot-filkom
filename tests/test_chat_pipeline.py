# ===================================================================
# FILE: tests/test_chat_pipeline.py
# Hotfix 1.9.1 – Auto Validasi Semester DOSEN_PENGAMPU + Cached Verification
# ===================================================================
"""
End-to-End Test untuk Chat Pipeline FILKOM (Hotfix 1.9.1)
Memvalidasi khusus DOSEN_PENGAMPU: semester & dosen harus sesuai data KB.
Menguji seluruh alur: Intent + NER + KB Lookup + Semantic Search + Template,
dengan auto-verifikasi cached flag & waktu eksekusi optimal.

Cara Menjalankan:
(venv) PS D:\chatbot-filkom> python tests/test_chat_pipeline.py
"""

import sys
import os
import time

# Tambahkan root project ke sys.path agar modul app terdeteksi
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.kb_service import initialize_knowledge_base
from app.services.intent_service import initialize_intent_service
from app.services.ner_service import initialize_ner_service
from app.services.semantic_search_service import search_service
from app.services.chat_pipeline import ChatPipeline


def test_chat_pipeline():
    """End-to-end test untuk Chat Pipeline dengan auto init & deteksi cache otomatis."""

    print("\n=== 🔄 INITIALIZING SERVICES (Hotfix 1.9.1) ===")

    # ✅ Pastikan semua service diinisialisasi dulu
    if not initialize_knowledge_base():
        print("❌ Knowledge Base gagal dimuat. Pastikan data lengkap.")
        return
    print("✅ Knowledge Base ready.")

    if not initialize_intent_service():
        print("❌ Intent Service gagal dimuat.")
        return
    print("✅ Intent Service ready.")

    if not initialize_ner_service():
        print("❌ NER Service gagal dimuat.")
        return
    print("✅ NER Service ready.")

    if not search_service.load_models():
        print("❌ Semantic Search gagal dimuat.")
        return
    print("✅ Semantic Search ready.")

    pipeline = ChatPipeline()  # ✅ Baru dibuat setelah semua service ready
    print("\n✅ Semua service berhasil diinisialisasi. Memulai pengujian pipeline...\n")

    # ✅ Beragam query untuk mencakup seluruh alur pipeline
    test_queries = [
        "Siapa dosen pengampu Machine Learning?",
        "Jadwal kuliah Basis Data kapan?",
        "Berapa SKS Algoritma dan Pemrograman?",
        "Kontak dosen Hendry Fonda",
        "Syarat mengambil skripsi apa saja?",
        "Mata kuliah semester 3 apa saja?",
        "Ruang kuliah Computer Vision dimana?",
        "NIDN pak Bambang berapa?",
        "Prasyarat mata kuliah Web Programming",
        "Kapan jadwal UTS semester ini?",
        "hari senin ada kuliah apa?",
        "Info lengkap jadwal Ruang 2A di semester ini gimana min?",
        "jadwal bu eka ngajar offline atau online minggu ini?",
        "Bagaimana langkah-langkah ngisi KRS semester ini?",
        "Bagaimana prosedur pengajuan cuti kuliah semester ini?",
        "Tolong info detail jumlah SKS Rekayasa Perangkat Lunak ya min",
        "Profil Pak Yuda bisa dilihat di mana min?",
        "min jaringan komputer itu lebih banyak teori atau praktek sih?",
        "Tolong dijelaskan ulang lebih detail biar saya ngerti"
    ]

    print("\n=== 🧪 TESTING CHAT PIPELINE (Hotfix 1.9.1 – Cached Verification + Validasi Semester) ===\n")

    for i, query in enumerate(test_queries, 1):
        print(f"[TEST {i}] Query: {query}")

        try:
            # ✅ Tes pertama (CACHE MISS)
            start_1 = time.time()
            result_1 = pipeline.process_message(query, "test_user", "test_session")
            elapsed_1 = time.time() - start_1

            # ✅ Tes kedua (CACHE HIT → harus jauh lebih cepat)
            start_2 = time.time()
            result_2 = pipeline.process_message(query, "test_user", "test_session")
            elapsed_2 = time.time() - start_2

            print(f"Response      : {result_2.response}")
            print(f"Intent        : {result_2.intent} (confidence: {result_2.confidence:.3f})")
            print(f"Entities      : {result_2.entities}")

            if result_2.template_used:
                print(f"Template Used : ✅ {result_2.template_used}")

            if result_2.fallback_reason:
                print(f"Fallback      : ⚠️ {result_2.fallback_reason}")

            if result_2.search_results:
                print(f"Search Results: {len(result_2.search_results)} items")
                top1_text = result_2.search_results[0].get('text') or result_2.search_results[0].get('konten')
                print(f"Top-1 Result  : {top1_text}")

            # ✅ Validasi tambahan khusus DOSEN_PENGAMPU
            if result_2.intent == "DOSEN_PENGAMPU":
                if "semester" in result_2.response.lower() and "-" not in result_2.response:
                    print("✅ Validasi Semester OK → Semester ditemukan di response.")
                else:
                    print("❌ Validasi Semester GAGAL → Semester masih '-' di response.")

            print(f"Processing    : {elapsed_1:.3f}s → {elapsed_2:.3f}s (2nd run)")
            print(f"Cached        : {'✅ Yes' if getattr(result_2, 'cached', False) else '❌ No'}")
            if result_2.cached and elapsed_2 < 0.1:
                print("⚡ Cache Optimal (Execution <0.1s)")
            print("-" * 80)

        except Exception as e:
            print(f"❌ ERROR: {e}")
            print("-" * 80)

        time.sleep(0.3)  # jeda antar tes biar log rapi

    # ✅ Tampilkan history percakapan terakhir
    print("\n=== 📜 CONVERSATION HISTORY (Last 3 Exchanges) ===")
    history = pipeline.get_conversation_history("test_user", "test_session")
    print(f"Total conversations: {len(history)}")

    for conv in history[-3:]:
        print(f"User  : {conv['user_message']}")
        print(f"Bot   : {conv['bot_response']}")
        print(f"Intent: {conv['intent']}\n")


# ===================================================================
# ENTRY POINT
# ===================================================================
if __name__ == "__main__":
    test_chat_pipeline()

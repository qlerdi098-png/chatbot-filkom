# ===================================================================
# FILE: tests/test_all_endpoints.py
# Hotfix 1.9 – Full Cached Detection + Chat Process Test (Optimal Timing)
# ===================================================================

import requests
import time

BASE_URL = "http://127.0.0.1:8000"

ENDPOINTS = [
    (f"{BASE_URL}/health", "GET"),
    (f"{BASE_URL}/kb/status", "GET"),
    (f"{BASE_URL}/demo/search-demo", "GET"),
    (f"{BASE_URL}/intent/status", "GET"),
    (f"{BASE_URL}/intent/classes", "GET"),
    (f"{BASE_URL}/intent/predict?text=jadwal+machine+learning", "GET"),
    (f"{BASE_URL}/ner/status", "GET"),
    (f"{BASE_URL}/ner/labels", "GET"),
    (f"{BASE_URL}/ner/performance", "GET"),
    (f"{BASE_URL}/ner/predict?text=jadwal+machine+learning", "GET"),
    (f"{BASE_URL}/search/status", "GET"),
    (f"{BASE_URL}/search/demo", "GET"),
    (f"{BASE_URL}/search/query?q=syarat+skripsi", "GET"),
    (f"{BASE_URL}/api/v1/chat/status", "GET"),
    (f"{BASE_URL}/api/v1/chat/history/test_user/test_session", "GET"),
    # ✅ Tambahan: Tes otomatis Chat Pipeline (POST)
    (f"{BASE_URL}/api/v1/chat/process", "POST"),
]

def run_tests():
    print("\n=== 🔥 RUNNING FULL ENDPOINT TESTS (Hotfix 1.9 – Full Cached Detection) ===\n")

    for url, method in ENDPOINTS:
        try:
            start = time.time()

            # ✅ Tambahkan payload khusus untuk /api/v1/chat/process
            if "/api/v1/chat/process" in url and method == "POST":
                res = requests.post(
                    url,
                    json={
                        "message": "Siapa dosen pengampu Machine Learning?",
                        "user_id": "test_user",
                        "session_id": "test_session"
                    },
                    timeout=10
                )
            else:
                res = requests.get(url, timeout=10) if method == "GET" else requests.post(url, timeout=10)

            elapsed = round(time.time() - start, 3)

            if res.status_code == 200:
                status_note = ""
                json_data = {}
                try:
                    json_data = res.json()
                except Exception:
                    pass

                # ✅ Auto-detect cache flags
                if "/ner/status" in url and json_data.get("model_details", {}).get("labels_cached"):
                    status_note = "(cached)"
                elif "/ner/labels" in url and json_data.get("labels_cached", False):
                    status_note = "(cached)"
                elif "/intent/status" in url and json_data.get("cached", False):
                    status_note = "(cached)"
                elif "/intent/classes" in url and json_data.get("cached", False):
                    status_note = "(cached)"
                elif "/demo/search-demo" in url and json_data.get("cached", False):
                    status_note = "(cached)"
                elif "/search/demo" in url and json_data.get("cached", False):
                    status_note = "(cached)"
                elif "/api/v1/chat/process" in url and json_data.get("cached", False):
                    status_note = "(cached)"
                elif "/api/v1/chat/status" in url and json_data.get("active_contexts", 0) > 0:
                    status_note = f"(contexts: {json_data.get('active_contexts')})"

                # ✅ NER predict entity count
                if "/ner/predict" in url and json_data.get("entities"):
                    entity_count = json_data.get("entity_count", 0)
                    status_note += f" ({entity_count} entities detected)"

                # ✅ Chat process intent info
                if "/api/v1/chat/process" in url:
                    intent = json_data.get("intent", "UNKNOWN")
                    conf = json_data.get("confidence", 0.0)
                    status_note += f" [intent: {intent} ({conf:.3f})]"

                # ✅ Waktu eksekusi <0.1s untuk endpoint cached
                if "cached" in status_note and elapsed < 0.1:
                    status_note += " ⚡"

                print(f"[✅ PASS] {method} {url} → 200 OK {status_note} | {elapsed}s")

            else:
                print(f"[❌ FAIL] {method} {url} → {res.status_code} | {res.text[:100]} | {elapsed}s")

        except Exception as e:
            print(f"[❌ ERROR] {method} {url} → {e}")

if __name__ == "__main__":
    run_tests()

# ===================================================================
# FILE: deploy_chat.py (FINAL - Step 10 Synced)
# ===================================================================

import os
import sys
from pathlib import Path

def deploy_chat_pipeline():
    """Deploy Chatbot FILKOM - Production Prep"""

    print("🚀 DEPLOYING CHATBOT FILKOM - CHAT PIPELINE")
    print("=" * 60)

    # 1. Check Dependencies
    print("1. Checking dependencies...")
    try:
        import fastapi, torch, transformers, sentence_transformers, faiss
        print("✅ Core dependencies OK")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

    # 2. Check Model Files
    print("\n2. Checking model files...")
    model_dir = Path("models")
    intent_model = model_dir / "intent_classifier" / "best_model.pth"
    ner_model = model_dir / "ner_model" / "best_model.pt"

    print("✅ Intent model found" if intent_model.exists() else "❌ Intent model missing")
    print("✅ NER model found" if ner_model.exists() else "❌ NER model missing")

    # 3. Check Data Files
    print("\n3. Checking data files...")
    data_dir = Path("data")
    kb_file = data_dir / "kb_processed.json"
    templates_file = data_dir / "response_templates.json"

    print("✅ Knowledge base data found" if kb_file.exists() else "❌ Knowledge base data missing")
    print("✅ Response templates found" if templates_file.exists() else "❌ Response templates missing")

    # 4. Test Import Pipeline
    print("\n4. Testing pipeline import...")
    try:
        sys.path.append(str(Path.cwd()))
        from app.services.chat_pipeline import ChatPipeline
        _ = ChatPipeline()
        print("✅ Chat pipeline imported successfully")
    except Exception as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

    # 5. Test API
    print("\n5. Testing API endpoints...")
    try:
        from app.main import app
        print("✅ FastAPI app imported successfully")
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False

    # 6. Create Startup Script
    print("\n6. Creating startup script...")
    startup_script = """#!/bin/bash
# Chatbot FILKOM Startup Script

echo "🤖 Starting Chatbot FILKOM..."

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start FastAPI server (Production Mode)
uvicorn app.main:app --host 0.0.0.0 --port 8000

echo "✅ Chatbot FILKOM started on http://localhost:8000"
"""
    with open("start_chatbot.sh", "w") as f:
        f.write(startup_script)
    os.chmod("start_chatbot.sh", 0o755)
    print("✅ Startup script created: start_chatbot.sh")

    # 7. Create Requirements File
    print("\n7. Creating requirements file...")
    requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
torch==2.1.0
transformers==4.35.0
TorchCRF==1.1.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
rank-bm25==0.2.2
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
pydantic==2.5.0
jinja2==3.1.2
python-multipart==0.0.6
aiofiles==23.2.1
nltk==3.8.1
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Requirements file created")

    print("\n🎉 DEPLOYMENT PREPARATION COMPLETE!")
    print("Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Start server: ./start_chatbot.sh")
    print("3. Open browser: http://localhost:8000 or /api/v1/")
    print("4. Test chat interface in browser")

    return True

if __name__ == "__main__":
    deploy_chat_pipeline()

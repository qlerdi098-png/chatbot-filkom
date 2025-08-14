1. Struktur Folder Final Backend
   bash
   Copy
   Edit
   chatbot-filkom/
   │
   ├── app/ # Main Application Package
   │ ├── **init**.py # Info versi & shortcut settings/logging
   │ │
   │ ├── api/ # FastAPI API Endpoints
   │ │ ├── **init**.py # Router aggregator (health, intent, ner, search)
   │ │ ├── health.py # Health & status check
   │ │ ├── intent_api.py # Intent classification endpoints
   │ │ ├── ner_api.py # NER endpoints
   │ │ ├── search_api.py # Semantic search endpoints
   │ │ ├── chat_api.py # Chat pipeline endpoints
   │ │ └── routes.py # Combine all router + Chat Web Interface
   │ │
   │ ├── core/ # Core Config & Logging
   │ │ ├── **init**.py
   │ │ ├── config.py # Settings & env
   │ │ ├── exceptions.py # Custom exception classes
   │ │ └── logging.py # Loguru-based logging setup
   │ │
   │ ├── services/ # Business Logic Services
   │ │ ├── **init**.py
   │ │ ├── kb_service.py # Knowledge base loader & query
   │ │ ├── intent_service.py # IndoBERT-based intent classification
   │ │ ├── ner_service.py # IndoBERT+BiLSTM+CRF NER
   │ │ ├── semantic_search_service.py # BM25+MiniLM hybrid semantic search
   │ │ ├── template_service.py # Template filler + KB Lookup
   │ │ └── chat_pipeline.py # Integrasi Intent+NER+KB+Search
   │ │
   │ └── main.py # FastAPI entry point
   │
   ├── frontend/ # Web Frontend
   │ ├── templates/ # HTML (Jinja2)
   │ │ ├── base.html
   │ │ ├── index.html
   │ │ ├── chat.html
   │ │ └── status.html
   │ └── static/
   │ ├── css/
   │ │ └── style.css
   │ └── images/ # logo.png dll
   │
   ├── models/ # Pre-trained models
   │ ├── intent_classifier/
   │ ├── ner_model/
   │ └── semantic_search/
   │
   ├── data/ # Knowledge base & templates
   │ ├── kb_processed.json
   │ └── response_templates.json
   │
   ├── tests/ # Testing
   │ ├── test_chat_pipeline.py
   │ └── test_integration.py
   │
   ├── .env
   ├── requirements.txt
   └── README.md (akan kita buat berikutnya)
2. Service Status – Production Ready
   Service Status File Terkait
   Knowledge Base ✅ Ready kb_service.py
   Intent Classification ✅ Ready (IndoBERT Fine-tuned) intent_service.py
   NER ✅ Ready (IndoBERT+BiLSTM+CRF) ner_service.py
   Semantic Search ✅ Ready (BM25 + MiniLM Hybrid) semantic_search_service.py
   Template Filler ✅ Ready template_service.py
   Chat Pipeline ✅ Integrated & tested chat_pipeline.py
   Frontend Chat Web ✅ Responsive + integrated index.html, chat.html, status.html

3. Testing & Monitoring
   ✅ Integration Testing – tests/test_integration.py
   ✅ End-to-End Chat Pipeline Testing – tests/test_chat_pipeline.py
   ✅ Health Monitoring – endpoint /api/v1/health + /status

4. Siap Deployment
   Local Development: uvicorn app.main:app --reload

Production: uvicorn app.main:app --host 0.0.0.0 --port 8000 atau via Docker.

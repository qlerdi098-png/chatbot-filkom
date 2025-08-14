# 🤖 Chatbot FILKOM

Intelligent chatbot system untuk Fakultas Ilmu Komputer dengan kemampuan Natural Language Processing dan knowledge base integration.

## 📋 Deskripsi

Chatbot FILKOM adalah sistem chatbot cerdas yang dirancang khusus untuk menjawab pertanyaan seputar Fakultas Ilmu Komputer. Sistem ini menggunakan teknologi Machine Learning untuk klasifikasi intent, Named Entity Recognition (NER), dan semantic search untuk memberikan respons yang akurat dan relevan.

## ✨ Fitur Utama

- **Intent Classification**: Klasifikasi otomatis maksud pengguna menggunakan IndoBERT
- **Named Entity Recognition**: Ekstraksi entitas penting dari teks menggunakan IndoBERT + BiLSTM + CRF
- **Semantic Search**: Pencarian semantik menggunakan BM25 + MiniLM hybrid model
- **Knowledge Base**: Basis pengetahuan terintegrasi tentang FILKOM
- **Web Interface**: Interface chat yang responsif dan user-friendly
- **Real-time Processing**: Pemrosesan chat secara real-time
- **Health Monitoring**: Monitoring kesehatan sistem dan API endpoints

## 🏗️ Arsitektur Sistem

```
chatbot-filkom/
├── app/                          # Main Application Package
│   ├── api/                      # FastAPI API Endpoints
│   │   ├── chat_api.py          # Chat pipeline endpoints
│   │   ├── health.py            # Health & status check
│   │   ├── intent_api.py        # Intent classification endpoints
│   │   ├── ner_api.py           # NER endpoints
│   │   ├── routes.py            # Route aggregator
│   │   └── search_api.py        # Semantic search endpoints
│   ├── core/                    # Core Config & Logging
│   │   ├── config.py            # Settings & environment
│   │   ├── exceptions.py        # Custom exception classes
│   │   └── logging.py           # Logging setup
│   └── services/                # Business Logic Services
│       ├── chat_pipeline.py     # Main chat pipeline
│       ├── intent_service.py    # Intent classification service
│       ├── kb_service.py        # Knowledge base service
│       ├── ner_service.py       # NER service
│       ├── semantic_search_service.py # Semantic search service
│       └── template_service.py  # Template processing service
├── frontend/                    # Web Frontend
│   ├── static/                  # Static assets (CSS, JS, images)
│   └── templates/               # HTML templates
├── models/                      # Pre-trained Models
│   ├── intent_classifier/       # Intent classification model
│   ├── ner_model/              # NER model
│   └── semantic_search/        # Semantic search model
├── data/                       # Knowledge Base & Templates
├── tests/                      # Testing Suite
└── monitoring/                 # Logs & Monitoring
```

## 🚀 Instalasi

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Setup Environment

1. **Clone Repository**
   ```bash
   git clone https://github.com/erdiansyahm/chatbot-filkom.git
   cd chatbot-filkom
   ```

2. **Buat Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env sesuai konfigurasi Anda
   ```

## 🏃‍♂️ Cara Menjalankan

### Development Mode
```bash
# Local development dengan auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Production deployment
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build dan run dengan Docker
docker build -t chatbot-filkom .
docker run -p 8000:8000 chatbot-filkom
```

Akses aplikasi di: **http://localhost:8000**

## 🔧 API Endpoints

### Chat Endpoints
- `POST /api/v1/chat` - Main chat endpoint
- `GET /api/v1/health` - Health check
- `GET /api/v1/status` - Service status

### Service Endpoints
- `POST /api/v1/intent/classify` - Intent classification
- `POST /api/v1/ner/extract` - Named entity recognition  
- `POST /api/v1/search/semantic` - Semantic search
- `GET /api/v1/kb/search` - Knowledge base search

## 🧪 Testing

### Menjalankan Tests
```bash
# Semua tests
pytest

# Integration tests
pytest tests/test_integration.py

# Chat pipeline tests  
pytest tests/test_chat_pipeline.py

# Coverage report
pytest --cov=app tests/
```

### Test Categories
- ✅ **Integration Testing** - `tests/test_integration.py`
- ✅ **End-to-End Chat Pipeline** - `tests/test_chat_pipeline.py`
- ✅ **Health Monitoring** - Endpoint `/api/v1/health`

## 📊 Service Status

| Service | Status | Description |
|---------|--------|-------------|
| Knowledge Base | ✅ Ready | `kb_service.py` - Loaded and ready |
| Intent Classification | ✅ Ready | `intent_service.py` - IndoBERT Fine-tuned |
| NER | ✅ Ready | `ner_service.py` - IndoBERT + BiLSTM + CRF |
| Semantic Search | ✅ Ready | `semantic_search_service.py` - BM25 + MiniLM Hybrid |
| Template Filler | ✅ Ready | `template_service.py` - Template processing |
| Chat Pipeline | ✅ Integrated | `chat_pipeline.py` - Fully integrated and tested |
| Frontend | ✅ Responsive | Web interface with real-time chat |

## 🛠️ Teknologi Stack

- **Backend**: FastAPI, Python 3.8+
- **Machine Learning**: PyTorch, Transformers (HuggingFace)
- **Models**: IndoBERT, BiLSTM, CRF, BM25, MiniLM
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: JSON-based knowledge base
- **Deployment**: Docker, Uvicorn
- **Testing**: Pytest
- **Monitoring**: Custom health endpoints

## 📝 Development

### Menambah Intent Baru
1. Update `models/intent_classifier/intent_mapping.json`
2. Retrain model intent classification
3. Update knowledge base di `data/`
4. Test dengan `pytest tests/test_intent_service.py`

### Menambah Template Respons
1. Edit `data/response_templates.json`
2. Test template dengan endpoint `/api/v1/chat`

### Monitoring & Logs
- Logs tersimpan di `monitoring/logs/`
- Health check: `GET /api/v1/health`
- Service status: `GET /api/v1/status`

## 🤝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push ke branch (`git push origin feature/amazing-feature`)
5. Buat Pull Request

## 📄 License

Project ini menggunakan MIT License. Lihat file `LICENSE` untuk detail.

## 📧 Contact

**Developer**: Erdiansyah M  
**GitHub**: [@qlerdi098-png](https://github.com/qlerdi098-png)  
**Repository**: [chatbot-filkom](https://github.com/qlerdi098-png/chatbot-filkom)

---

⭐ **Star repository ini jika membantu!** ⭐

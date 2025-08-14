# ===================================================================
# FILE: app/main.py (FINAL – Clean Fixed & Hotfix 1.4)
# ===================================================================

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Core imports
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.exceptions import ChatbotException

# Routers (prefix sudah ada di masing-masing router → tidak perlu prefix ganda di sini)
from app.api.health import router as health_router
from app.api.routes import api_router
from app.api.intent_api import router as intent_router
from app.api.ner_api import router as ner_router
from app.api.search_api import router as search_router
from app.api.chat_api import router as chat_router

# Services
from app.services.kb_service import initialize_knowledge_base, kb_loader
from app.services.intent_service import initialize_intent_service, intent_service
from app.services.ner_service import initialize_ner_service, ner_service
from app.services.semantic_search_service import search_service
from app.services.chat_pipeline import ChatPipeline

# ===================================================================
# Logging Setup
# ===================================================================
logger = setup_logging(log_level=settings.LOG_LEVEL)

# ===================================================================
# FastAPI Application
# ===================================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "🤖 Chatbot FILKOM - AI Assistant untuk Fakultas Ilmu Komputer.\n"
        "Membantu mahasiswa mendapatkan informasi akademik dengan akurat dan cepat."
    ),
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    debug=settings.DEBUG
)

# ===================================================================
# CORS Middleware
# ===================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# Static Files & Templates
# ===================================================================
static_path = Path("frontend/static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info("📁 Static files mounted at /static")

templates_path = Path("frontend/templates")
templates = Jinja2Templates(directory=str(templates_path)) if templates_path.exists() else None
if templates:
    logger.info("📄 Templates loaded from frontend/templates")
else:
    logger.warning("⚠️ Templates directory not found")

# ===================================================================
# Global Instances (Lazy Init ChatPipeline)
# ===================================================================
global_chat_pipeline: Optional[ChatPipeline] = None

# ===================================================================
# Global Exception Handlers
# ===================================================================
@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    logger.error(f"Chatbot error: {exc.message} - Details: {exc.details}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Chatbot Error",
            "message": exc.message,
            "details": exc.details if settings.DEBUG else None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Terjadi kesalahan internal. Silakan coba lagi."
            if not settings.DEBUG else str(exc)
        }
    )

# ===================================================================
# Startup Event
# ===================================================================
@app.on_event("startup")
async def startup_event():
    global global_chat_pipeline

    logger.info("🚀 Starting Chatbot FILKOM Application")
    logger.info(f"📊 Version: {settings.APP_VERSION} | Debug: {settings.DEBUG}")
    logger.info(f"💻 Device: {settings.DEVICE}")
    logger.info(f"📁 Models Directory: {settings.MODELS_DIR}")
    logger.info(f"📚 Data Directory: {settings.DATA_DIR}")

    # 1. Knowledge Base Initialization
    try:
        if not initialize_knowledge_base():
            logger.warning("⚠️ Knowledge base initialization failed - running in limited mode")
        else:
            logger.info("✅ Knowledge base initialized successfully")
    except Exception as e:
        logger.error(f"❌ Knowledge base initialization error: {str(e)}")

    # 2. Intent Service Initialization
    try:
        if not initialize_intent_service():
            logger.warning("⚠️ Intent service initialization failed - running in limited mode")
        else:
            logger.info("✅ Intent service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Intent service initialization error: {str(e)}")

    # 3. NER Service Initialization
    try:
        if not initialize_ner_service():
            logger.warning("⚠️ NER service initialization failed - running in limited mode")
        else:
            logger.info("✅ NER service initialized successfully")
    except Exception as e:
        logger.error(f"❌ NER service initialization error: {str(e)}")

    # 4. Semantic Search Initialization
    try:
        logger.info("🔍 Loading Semantic Search Models...")
        if search_service.load_models():
            logger.info("✅ Semantic Search Models loaded successfully")
        else:
            logger.warning("⚠️ Semantic search models not found. Please rebuild if needed.")
    except Exception as e:
        logger.error(f"❌ Semantic Search initialization error: {str(e)}")

    # 5. Chat Pipeline Initialization (Lazy Init After All Services Ready)
    try:
        global_chat_pipeline = ChatPipeline()
        logger.info("✅ Chat Pipeline initialized successfully after all services ready")
    except Exception as e:
        logger.error(f"❌ Chat Pipeline initialization error: {str(e)}")

    # Summary Status
    services_status = {
        "Knowledge Base": "✅" if kb_loader.is_loaded else "❌",
        "Intent Classification": "✅" if intent_service.is_loaded else "⚠️",
        "NER": "✅" if ner_service.is_loaded else "⚠️",
        "Semantic Search": "✅" if search_service.is_loaded else "⚠️",
        "Chat Pipeline": "✅" if global_chat_pipeline else "⚠️"
    }
    logger.info("📊 SERVICES STATUS:")
    for service, status in services_status.items():
        logger.info(f"   {status} {service}")
    logger.info("🎉 Chatbot FILKOM is ready!")

# ===================================================================
# Shutdown Event
# ===================================================================
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Chatbot FILKOM Application")
    logger.info("✅ Application shutdown completed")

# ===================================================================
# Basic Endpoints (Frontend Pages)
# ===================================================================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if templates:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "app_name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "current_year": datetime.now().year
            }
        )
    return {"message": f"🤖 {settings.APP_NAME} is running!", "version": settings.APP_VERSION}

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    if templates:
        return templates.TemplateResponse("chat.html", {"request": request, "app_name": settings.APP_NAME})
    return {"message": "Chat interface template not found."}

@app.get("/status", response_class=HTMLResponse)
async def status_page(request: Request):
    if templates:
        return templates.TemplateResponse(
            "status.html",
            {
                "request": request,
                "app_name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "current_year": datetime.now().year
            }
        )
    return {"message": "Status monitoring template not found."}

@app.get("/ping")
async def ping():
    return {"message": "pong", "timestamp": time.time()}

# ===================================================================
# Include API Routers (NO PREFIX – already set in router files)
# ===================================================================
app.include_router(health_router)
app.include_router(api_router)
app.include_router(intent_router)
app.include_router(ner_router)
app.include_router(search_router)
app.include_router(chat_router)

# ===================================================================
# Main Entry Point
# ===================================================================
if __name__ == "__main__":
    logger.info(f"🌐 Starting development server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

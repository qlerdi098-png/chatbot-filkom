# ===================================================================
# FILE: app/__init__.py (FINAL – Production-Ready)
# ===================================================================
"""
Chatbot FILKOM Application
AI Assistant untuk Fakultas Ilmu Komputer

Versi ini digunakan untuk inisialisasi global package,
sehingga modul core seperti settings & logging bisa langsung diimport.
"""

__version__ = "1.0.0"
__author__ = "FILKOM Development Team"
__license__ = "MIT"

# Shortcut imports untuk kemudahan akses di seluruh aplikasi
from app.core.config import settings
from app.core.logging import setup_logging

__all__ = [
    "settings",
    "setup_logging",
]

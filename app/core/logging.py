# ===================================================================
# FILE: app/core/logging.py (CLEAN - No Warning Version)
# ===================================================================
import logging
import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup application logging dengan loguru (Clean & Production-Ready).

    :param log_level: Logging level (default: INFO)
    :return: Configured logger instance
    """
    # ✅ Adjust log level based on DEBUG flag
    log_level = "DEBUG" if settings.DEBUG else log_level

    # ✅ Remove default handlers to avoid duplicate logs
    logger.remove()

    # ✅ Console logging (utama saat development & production)
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True
    )

    # ✅ Log directory (base dir aware)
    log_dir: Path = settings.BASE_DIR / "monitoring" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if settings.ENABLE_MONITORING:
        # ✅ File logging
        logger.add(
            log_dir / "app.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="100 MB",
            retention="30 days",
            compression="gz"
        )

        # ✅ Error logging
        logger.add(
            log_dir / "error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="50 MB",
            retention="60 days",
            compression="gz"
        )

    # ✅ Redirect standard logging (FastAPI & Uvicorn)
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelname, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=log_level)

    return logger

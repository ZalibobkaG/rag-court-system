import sys

from loguru import logger

from app.core.config import settings


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    logger.add("logs/app.log", level=settings.log_level, rotation="10 MB", retention=5, encoding="utf-8")

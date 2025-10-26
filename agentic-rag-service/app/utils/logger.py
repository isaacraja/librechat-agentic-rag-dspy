"""Logging configuration using loguru."""

import sys
from loguru import logger
from app.config import settings


def setup_logging():
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()

    # Add custom handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
        colorize=True,
    )

    # Add file handler for persistent logs
    logger.add(
        "logs/agentic-rag-{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="7 days",  # Keep logs for 7 days
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    return logger

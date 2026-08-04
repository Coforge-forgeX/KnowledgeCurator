"""Centralized logging with structlog"""
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from core.config import settings


def setup_logging() -> None:
    """Configure structured logging with environment-aware formatting"""

    # Choose renderer based on LOG_FORMAT setting
    # "console" = human-readable with colors (best for development)
    # "json" = structured JSON (best for production/log aggregation)
    if settings.LOG_FORMAT.lower() == "console":
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        renderer = structlog.processors.JSONRenderer()

    # Common processors for all environments
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        renderer,
    ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    )

    # Reduce noise from third-party libraries
    if settings.is_development:
        # Set third-party loggers to WARNING to reduce clutter
        for noisy_logger in [
            "httpx",
            "httpcore",
            "urllib3",
            "azure",
            "azure.core.pipeline.policies.http_logging_policy",
            "azure.storage",
            "boto3",
            "botocore",
            "s3transfer",
            "neo4j",
            "pymongo",
            "redis",
            "lightrag",
            "openai",
            "anthropic",
            "langchain",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class Logger:
    """Structured logger"""

    def __init__(self, name: str = None):
        self.logger = structlog.get_logger(name or "indexer-service")

    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add common context"""
        return {
            "app": "indexer-service",
            "environment": settings.ENVIRONMENT,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }

    def info(self, message: str, **kwargs) -> None:
        self.logger.info(message, **self._add_context(**kwargs))

    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(message, **self._add_context(**kwargs))

    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        context = self._add_context(**kwargs)
        if error:
            context.update(
                {
                    "error_message": str(error),
                    "error_type": type(error).__name__,
                }
            )
        self.logger.error(message, **context)

    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, **self._add_context(**kwargs))

    def critical(
        self, message: str, error: Optional[Exception] = None, **kwargs
    ) -> None:
        context = self._add_context(**kwargs)
        if error:
            context.update(
                {
                    "error_message": str(error),
                    "error_type": type(error).__name__,
                }
            )
        self.logger.critical(message, **context)

    def exception(self, message: str, **kwargs) -> None:
        self.logger.exception(message, **self._add_context(**kwargs))


def get_logger(name: str = None) -> Logger:
    """
    Get a logger instance for the specified module.

    Usage:
        from src.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
        logger.error("Something went wrong", error=exception)

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return Logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all subsequent log messages"""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Unbind context variables"""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables"""
    structlog.contextvars.clear_contextvars()


# Initialize logging on module import
setup_logging()

# Alias for backward compatibility
configure_logging = setup_logging

# Global logger
logger = Logger(__name__)

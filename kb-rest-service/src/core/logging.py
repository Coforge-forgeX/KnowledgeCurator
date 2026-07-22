"""Centralized logging with structlog"""
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog


def setup_logging() -> None:
    """Configure structured logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    from core.config import settings

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    )


class Logger:
    """Structured logger"""

    def __init__(self, name: str = None):
        self.logger = structlog.get_logger(name or "kb-rest")

    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add common context"""
        from core.config import settings

        return {
            "app": "kb-rest-service",
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

    def log_request(
        self, method: str, path: str, user_id: Optional[str] = None, **kwargs
    ) -> None:
        self.info(
            "HTTP Request",
            request_method=method,
            request_path=path,
            user_id=user_id,
            **kwargs,
        )

    def log_response(self, status_code: int, response_time_ms: float, **kwargs) -> None:
        self.info(
            "HTTP Response",
            status_code=status_code,
            response_time_ms=response_time_ms,
            **kwargs,
        )


def get_logger(name: str = None) -> Logger:
    """
    Get a logger instance for the specified module.

    Usage:
        from core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
        logger.error("Something went wrong", error=exception)

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return Logger(name)


# Global logger
logger = Logger(__name__)

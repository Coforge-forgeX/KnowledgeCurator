"""Dependency health checks for the kb-rest-service /health endpoint."""
from typing import Any, Dict, Tuple

from sqlalchemy import text

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


async def _check_postgres() -> Dict[str, Any]:
    try:
        from .database import get_database

        db = get_database()
        async with db.get_session() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Postgres health check failed", error=e)
        return {"status": "unhealthy", "error": str(e)}


async def _check_neo4j() -> Dict[str, Any]:
    if not settings.database.NEO4J_URI:
        return {"status": "not_configured"}
    try:
        from .neo4j_driver import get_neo4j_driver

        driver = get_neo4j_driver()
        if driver._driver is None:
            await driver.connect()
        else:
            await driver.verify_connectivity()
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Neo4j health check failed", error=e)
        return {"status": "unhealthy", "error": str(e)}


async def _check_redis() -> Dict[str, Any]:
    try:
        from .redis import redis_manager

        if not settings.database.REDIS_HOST:
            return {"status": "not_configured"}

        redis_manager.initialize()
        if not redis_manager.is_available:
            return {"status": "unhealthy", "error": "not connected"}
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Redis health check failed", error=e)
        return {"status": "unhealthy", "error": str(e)}


async def _check_mongodb() -> Dict[str, Any]:
    uri = settings.database.MONGODB_URI
    if not uri:
        return {"status": "not_configured"}
    try:
        from src.services.mongodb_service import get_mongodb_service

        service = get_mongodb_service()
        await service.initialize()
        await service.db.client.admin.command("ping")
        return {"status": "healthy"}
    except Exception as e:
        logger.error("MongoDB health check failed", error=e)
        return {"status": "unhealthy", "error": str(e)}


async def run_health_checks() -> Tuple[Dict[str, Any], str]:
    """Run all dependency checks and return (checks, overall_status)."""
    checks = {
        "postgres": await _check_postgres(),
        "neo4j": await _check_neo4j(),
        "redis": await _check_redis(),
        "mongodb": await _check_mongodb(),
    }

    required_unhealthy = any(
        c["status"] == "unhealthy"
        for name, c in checks.items()
        if name != "redis"
    )
    overall_status = "unhealthy" if required_unhealthy else "healthy"

    return checks, overall_status

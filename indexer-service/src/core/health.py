"""Dependency health checks for the indexer-service /health endpoint."""
from typing import Any, Dict, Tuple

from sqlalchemy import text

from .database import db_manager
from .logging import get_logger

logger = get_logger(__name__)


async def _ensure_initialized() -> None:
    if not db_manager._initialized:
        await db_manager.initialize()


async def _check_postgres() -> Dict[str, Any]:
    from .config import settings

    if not settings.database.postgresql_url:
        return {"status": "not_configured"}
    try:
        await _ensure_initialized()
        async with db_manager.get_session() as session:
            await session.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Postgres health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}


async def _check_neo4j() -> Dict[str, Any]:
    from .config import settings

    if not settings.database.neo4j_uri:
        return {"status": "not_configured"}
    try:
        await _ensure_initialized()
        await db_manager.neo4j_driver.verify_connectivity()
        return {"status": "healthy"}
    except Exception as e:
        logger.error("Neo4j health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}


async def run_health_checks() -> Tuple[Dict[str, Any], str]:
    """Run all dependency checks and return (checks, overall_status)."""
    checks = {
        "postgres": await _check_postgres(),
        "neo4j": await _check_neo4j(),
    }

    overall_status = "unhealthy" if any(
        c["status"] == "unhealthy" for c in checks.values()
    ) else "healthy"

    return checks, overall_status

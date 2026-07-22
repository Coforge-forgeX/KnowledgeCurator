"""
State Manager

Persists and retrieves indexing job state for retry/resume functionality.
"""
import json
import os
from pathlib import Path
from typing import Optional

from core.logging import get_logger
from shared.indexing_state import IndexingJobState, IndexingState

logger = get_logger(__name__)


class StateManager:
    """
    Manages persistence of indexing job state.

    Stores state in:
    1. PostgreSQL database (primary)
    2. Local filesystem cache (fallback for crash recovery)
    """

    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize state manager.

        Args:
            state_dir: Directory for local state cache (from settings)
        """
        from core.config import settings

        self.state_dir = Path(state_dir or settings.processing.INDEXER_STATE_DIR)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        logger.info("State manager initialized", state_dir=str(self.state_dir))

    async def save_state(self, job_state: IndexingJobState) -> None:
        """
        Save job state to both database and local cache.

        Args:
            job_state: Current job state
        """
        try:
            # Save to database
            await self._save_to_database(job_state)

            # Save to local cache (for crash recovery)
            await self._save_to_file(job_state)

            logger.debug(
                "State saved",
                job_id=job_state.job_id,
                state=job_state.state,
            )

        except Exception as e:
            logger.error("Failed to save state", job_id=job_state.job_id, error=e)
            raise

    async def load_state(self, job_id: str) -> Optional[IndexingJobState]:
        """
        Load job state from database or local cache.

        Args:
            job_id: Job ID

        Returns:
            IndexingJobState if found, None otherwise
        """
        try:
            # Try database first
            job_state = await self._load_from_database(job_id)

            if job_state:
                logger.debug("State loaded from database", job_id=job_id)
                return job_state

            # Fallback to local cache
            job_state = await self._load_from_file(job_id)

            if job_state:
                logger.debug("State loaded from local cache", job_id=job_id)
                return job_state

            logger.warning("No state found", job_id=job_id)
            return None

        except Exception as e:
            logger.error("Failed to load state", job_id=job_id, error=e)
            return None

    async def delete_state(self, job_id: str) -> None:
        """
        Delete job state after successful completion.

        Args:
            job_id: Job ID
        """
        try:
            # Delete from database
            await self._delete_from_database(job_id)

            # Delete from local cache
            await self._delete_from_file(job_id)

            logger.debug("State deleted", job_id=job_id)

        except Exception as e:
            logger.warning("Failed to delete state", job_id=job_id, error=e)

    async def _save_to_database(self, job_state: IndexingJobState) -> None:
        """Save state to PostgreSQL database"""
        from core.database import db_manager
        from sqlalchemy import text

        try:
            async with db_manager.get_session() as session:
                # Upsert into indexing_jobs table
                query = text("""
                    INSERT INTO indexing_jobs (
                        job_id, workspace_id, document_url, kb_id,
                        state, checkpoint_data, retry_count, last_error,
                        created_at, started_at, completed_at, updated_at
                    ) VALUES (
                        :job_id, :workspace_id, :document_url, :kb_id,
                        :state, :checkpoint_data, :retry_count, :last_error,
                        :created_at, :started_at, :completed_at, :updated_at
                    )
                    ON CONFLICT (job_id) DO UPDATE SET
                        state = EXCLUDED.state,
                        checkpoint_data = EXCLUDED.checkpoint_data,
                        retry_count = EXCLUDED.retry_count,
                        last_error = EXCLUDED.last_error,
                        started_at = EXCLUDED.started_at,
                        completed_at = EXCLUDED.completed_at,
                        updated_at = EXCLUDED.updated_at
                """)

                await session.execute(
                    query,
                    {
                        "job_id": job_state.job_id,
                        "workspace_id": job_state.workspace_id,
                        "document_url": job_state.document_url,
                        "kb_id": job_state.kb_id,
                        "state": job_state.state,
                        "checkpoint_data": job_state.checkpoint.model_dump_json(),
                        "retry_count": job_state.retry_count,
                        "last_error": job_state.last_error,
                        "created_at": job_state.created_at,
                        "started_at": job_state.started_at,
                        "completed_at": job_state.completed_at,
                        "updated_at": job_state.updated_at,
                    },
                )

                await session.commit()

        except Exception as e:
            logger.warning("Failed to save to database", job_id=job_state.job_id, error=e)
            # Don't raise - local cache is fallback

    async def _load_from_database(self, job_id: str) -> Optional[IndexingJobState]:
        """Load state from PostgreSQL database"""
        from core.database import db_manager
        from sqlalchemy import text

        try:
            async with db_manager.get_session() as session:
                query = text("SELECT * FROM indexing_jobs WHERE job_id = :job_id")
                result = await session.execute(query, {"job_id": job_id})
                row = result.fetchone()

                if not row:
                    return None

                # Convert row to dict
                row_dict = dict(row._mapping)

                # Parse checkpoint data
                checkpoint_data = json.loads(row_dict.get("checkpoint_data", "{}"))

                return IndexingJobState(
                    job_id=row_dict["job_id"],
                    workspace_id=row_dict["workspace_id"],
                    document_url=row_dict["document_url"],
                    kb_id=row_dict.get("kb_id"),
                    state=row_dict["state"],
                    checkpoint=checkpoint_data,
                    retry_count=row_dict.get("retry_count", 0),
                    last_error=row_dict.get("last_error"),
                    created_at=row_dict.get("created_at"),
                    started_at=row_dict.get("started_at"),
                    completed_at=row_dict.get("completed_at"),
                    updated_at=row_dict.get("updated_at"),
                )

        except Exception as e:
            logger.warning("Failed to load from database", job_id=job_id, error=e)
            return None

    async def _delete_from_database(self, job_id: str) -> None:
        """Delete state from database"""
        from core.database import db_manager
        from sqlalchemy import text

        try:
            async with db_manager.get_session() as session:
                query = text("DELETE FROM indexing_jobs WHERE job_id = :job_id")
                await session.execute(query, {"job_id": job_id})
                await session.commit()

        except Exception as e:
            logger.warning("Failed to delete from database", job_id=job_id, error=e)

    async def _save_to_file(self, job_state: IndexingJobState) -> None:
        """Save state to local filesystem cache"""
        try:
            state_file = self.state_dir / f"{job_state.job_id}.json"
            state_file.write_text(job_state.model_dump_json(indent=2))

        except Exception as e:
            logger.warning("Failed to save to file", job_id=job_state.job_id, error=e)

    async def _load_from_file(self, job_id: str) -> Optional[IndexingJobState]:
        """Load state from local filesystem cache"""
        try:
            state_file = self.state_dir / f"{job_id}.json"

            if not state_file.exists():
                return None

            state_data = json.loads(state_file.read_text())
            return IndexingJobState(**state_data)

        except Exception as e:
            logger.warning("Failed to load from file", job_id=job_id, error=e)
            return None

    async def _delete_from_file(self, job_id: str) -> None:
        """Delete state from local filesystem cache"""
        try:
            state_file = self.state_dir / f"{job_id}.json"
            if state_file.exists():
                state_file.unlink()

        except Exception as e:
            logger.warning("Failed to delete file", job_id=job_id, error=e)


# Singleton instance
_state_manager_instance: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get or create singleton state manager"""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
    return _state_manager_instance

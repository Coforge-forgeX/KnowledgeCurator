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
            logger.error("Failed to save state", job_id=job_state.job_id, error_msg=e)
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
            logger.error("Failed to load state", job_id=job_id, error_msg=e)
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
            logger.warning("Failed to delete state", job_id=job_id, error_msg=e)

    async def _save_to_database(self, job_state: IndexingJobState) -> None:
        """Save state to PostgreSQL database using SQLAlchemy ORM"""
        from core.database import get_async_session, IndexingJob
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert

        try:
            async with get_async_session() as session:
                # Upsert using PostgreSQL insert...on conflict
                stmt = insert(IndexingJob).values(
                    job_id=job_state.job_id,
                    workspace_id=job_state.workspace_id,
                    document_url=job_state.document_url,
                    kb_id=job_state.kb_id,
                    state=job_state.state,
                    checkpoint_data=job_state.checkpoint.model_dump() if job_state.checkpoint else {},
                    retry_count=job_state.retry_count,
                    last_error=job_state.last_error,
                    created_at=job_state.created_at,
                    started_at=job_state.started_at,
                    completed_at=job_state.completed_at,
                    updated_at=job_state.updated_at,
                )

                stmt = stmt.on_conflict_do_update(
                    index_elements=['job_id'],
                    set_={
                        'state': stmt.excluded.state,
                        'checkpoint_data': stmt.excluded.checkpoint_data,
                        'retry_count': stmt.excluded.retry_count,
                        'last_error': stmt.excluded.last_error,
                        'started_at': stmt.excluded.started_at,
                        'completed_at': stmt.excluded.completed_at,
                        'updated_at': stmt.excluded.updated_at,
                    }
                )

                await session.execute(stmt)

        except Exception as e:
            logger.warning("Failed to save to database", job_id=job_state.job_id, error_msg=e)
            # Don't raise - local cache is fallback

    async def _load_from_database(self, job_id: str) -> Optional[IndexingJobState]:
        """Load state from PostgreSQL database using SQLAlchemy ORM"""
        from core.database import get_async_session, IndexingJob
        from sqlalchemy import select

        try:
            async with get_async_session() as session:
                stmt = select(IndexingJob).where(IndexingJob.job_id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()

                if not job:
                    return None

                return IndexingJobState(
                    job_id=job.job_id,
                    workspace_id=job.workspace_id,
                    document_url=job.document_url,
                    kb_id=job.kb_id,
                    state=job.state,
                    checkpoint=job.checkpoint_data or {},
                    retry_count=job.retry_count or 0,
                    last_error=job.last_error,
                    created_at=job.created_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    updated_at=job.updated_at,
                )

        except Exception as e:
            logger.warning("Failed to load from database", job_id=job_id, error_msg=e)
            return None

    async def _delete_from_database(self, job_id: str) -> None:
        """Delete state from database using SQLAlchemy ORM"""
        from core.database import get_async_session, IndexingJob
        from sqlalchemy import delete

        try:
            async with get_async_session() as session:
                stmt = delete(IndexingJob).where(IndexingJob.job_id == job_id)
                await session.execute(stmt)

        except Exception as e:
            logger.warning("Failed to delete from database", job_id=job_id, error_msg=e)

    async def _save_to_file(self, job_state: IndexingJobState) -> None:
        """Save state to local filesystem cache"""
        try:
            state_file = self.state_dir / f"{job_state.job_id}.json"
            state_file.write_text(job_state.model_dump_json(indent=2))

        except Exception as e:
            logger.warning("Failed to save to file", job_id=job_state.job_id, error_msg=e)

    async def _load_from_file(self, job_id: str) -> Optional[IndexingJobState]:
        """Load state from local filesystem cache"""
        try:
            state_file = self.state_dir / f"{job_id}.json"

            if not state_file.exists():
                return None

            state_data = json.loads(state_file.read_text())
            return IndexingJobState(**state_data)

        except Exception as e:
            logger.warning("Failed to load from file", job_id=job_id, error_msg=e)
            return None

    async def _delete_from_file(self, job_id: str) -> None:
        """Delete state from local filesystem cache"""
        try:
            state_file = self.state_dir / f"{job_id}.json"
            if state_file.exists():
                state_file.unlink()

        except Exception as e:
            logger.warning("Failed to delete file", job_id=job_id, error_msg=e)


# Singleton instance
_state_manager_instance: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get or create singleton state manager"""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
    return _state_manager_instance

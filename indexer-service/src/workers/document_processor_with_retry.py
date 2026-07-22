"""
Document Processing Pipeline with Retry and Resume

Complete document ingestion with:
- State tracking at each step
- Checkpoint-based resume
- Exponential backoff retry
- Cached intermediate results
"""
import asyncio
import hashlib
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from core.database import db_manager
from core.logging import get_logger
from core.state_manager import get_state_manager
from processors.factory import get_document_processor_factory
from shared.indexing_state import CheckpointData, IndexingJobState, IndexingState
from storage.factory import get_storage_adapter

logger = get_logger(__name__)


class DocumentProcessorWithRetry:
    """
    Document processor with checkpoint-based retry and resume.

    Features:
    - Saves state after each step
    - Resumes from last successful checkpoint
    - Exponential backoff retry
    - Caches extracted text to avoid re-extraction
    """

    def __init__(self):
        from core.config import settings

        self.state_manager = get_state_manager()
        self.cache_dir = Path(settings.processing.INDEXER_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def process_document(
        self,
        job_id: str,
        workspace_id: int,
        document_url: str,
        kb_id: Optional[int] = None,
        max_retries: int = 3,
    ) -> Dict:
        """
        Process document with retry and resume capability.

        Args:
            job_id: Unique job identifier
            workspace_id: Workspace ID
            document_url: URL or blob path to document
            kb_id: Knowledge base ID (optional)
            max_retries: Maximum retry attempts

        Returns:
            Dict with processing results
        """
        start_time = time.time()

        # Load existing state or create new
        job_state = await self._load_or_create_state(
            job_id, workspace_id, document_url, kb_id, max_retries
        )

        logger.info(
            "Starting document processing",
            job_id=job_id,
            state=job_state.state,
            retry_count=job_state.retry_count,
        )

        try:
            # Resume from checkpoint or start fresh
            if job_state.should_resume():
                logger.info(
                    "Resuming from checkpoint",
                    job_id=job_id,
                    state=job_state.state,
                )
                result = await self._resume_from_checkpoint(job_state)
            else:
                result = await self._process_from_start(job_state)

            # Mark as completed
            job_state.mark_completed()
            await self.state_manager.save_state(job_state)

            duration = time.time() - start_time

            logger.info(
                "Document processing completed",
                job_id=job_id,
                duration_seconds=duration,
                retries=job_state.retry_count,
            )

            # Clean up state after successful completion
            await self.state_manager.delete_state(job_id)
            await self._cleanup_cache(job_id)

            return {
                "success": True,
                "duration_seconds": duration,
                "chunks_processed": job_state.checkpoint.chunks_processed,
                "doc_id": job_state.checkpoint.doc_id,
                "retry_count": job_state.retry_count,
            }

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "Document processing failed",
                job_id=job_id,
                error=str(e),
                state=job_state.state,
                retry_count=job_state.retry_count,
                exc_info=True,
            )

            # Record error
            job_state.record_error(str(e), job_state.state)

            # Check if we can retry
            if job_state.can_retry():
                job_state.increment_retry()
                await self.state_manager.save_state(job_state)

                retry_delay = job_state.get_retry_delay()

                logger.info(
                    "Will retry processing",
                    job_id=job_id,
                    retry_count=job_state.retry_count,
                    delay_seconds=retry_delay,
                )

                # Schedule retry (caller should handle this)
                return {
                    "success": False,
                    "error": str(e),
                    "retry_scheduled": True,
                    "retry_delay_seconds": retry_delay,
                    "retry_count": job_state.retry_count,
                }
            else:
                # Max retries exceeded
                job_state.mark_failed()
                await self.state_manager.save_state(job_state)

                logger.error(
                    "Max retries exceeded, marking as failed",
                    job_id=job_id,
                    retry_count=job_state.retry_count,
                )

                return {
                    "success": False,
                    "error": str(e),
                    "retry_scheduled": False,
                    "retry_count": job_state.retry_count,
                    "max_retries_exceeded": True,
                }

    async def _load_or_create_state(
        self,
        job_id: str,
        workspace_id: int,
        document_url: str,
        kb_id: Optional[int],
        max_retries: int,
    ) -> IndexingJobState:
        """Load existing state or create new"""
        job_state = await self.state_manager.load_state(job_id)

        if job_state:
            logger.info("Loaded existing state", job_id=job_id, state=job_state.state)
            return job_state

        # Create new state
        job_state = IndexingJobState(
            job_id=job_id,
            workspace_id=workspace_id,
            document_url=document_url,
            kb_id=kb_id,
            max_retries=max_retries,
        )

        await self.state_manager.save_state(job_state)

        return job_state

    async def _process_from_start(self, job_state: IndexingJobState) -> Dict:
        """Process document from the beginning"""
        job_state.mark_started()
        job_state.state = IndexingState.DOWNLOADING
        await self.state_manager.save_state(job_state)

        # Step 1: Download
        content, file_name, content_type = await self._download_file(job_state)

        # Step 2: Extract text
        extracted_text, extraction_method, page_count = await self._extract_text(
            job_state, content, file_name, content_type
        )

        # Step 3: Index with LightRAG
        doc_id, chunks = await self._index_document(
            job_state, extracted_text, file_name, extraction_method
        )

        # Step 4: Update metadata
        await self._update_metadata(
            job_state, doc_id, file_name, len(content), page_count, extraction_method
        )

        return {"doc_id": doc_id, "chunks": chunks}

    async def _resume_from_checkpoint(self, job_state: IndexingJobState) -> Dict:
        """Resume processing from last checkpoint"""
        checkpoint = job_state.checkpoint

        logger.info(
            "Resuming from checkpoint",
            job_id=job_state.job_id,
            checkpoint_state=job_state.state,
        )

        # Resume based on last successful state
        if job_state.state == IndexingState.DOWNLOADED:
            # Download completed, resume from extraction
            # Need to re-download since we don't cache binary files
            content, file_name, content_type = await self._download_file(job_state)

            extracted_text, extraction_method, page_count = await self._extract_text(
                job_state, content, file_name, content_type
            )

            doc_id, chunks = await self._index_document(
                job_state, extracted_text, file_name, extraction_method
            )

            await self._update_metadata(
                job_state, doc_id, file_name, len(content), page_count, extraction_method
            )

        elif job_state.state == IndexingState.EXTRACTED:
            # Extraction completed, resume from indexing
            # Load cached extracted text
            extracted_text = await self._load_cached_text(job_state.job_id)

            if not extracted_text:
                raise Exception("Cached extracted text not found, cannot resume")

            doc_id, chunks = await self._index_document(
                job_state,
                extracted_text,
                checkpoint.extracted_text_path or "unknown",
                checkpoint.extraction_method or "unknown",
            )

            await self._update_metadata(
                job_state,
                doc_id,
                checkpoint.extracted_text_path or "unknown",
                checkpoint.file_size or 0,
                checkpoint.page_count or 0,
                checkpoint.extraction_method or "unknown",
            )

        elif job_state.state == IndexingState.INDEXED:
            # Indexing completed, resume from metadata update
            await self._update_metadata(
                job_state,
                checkpoint.doc_id,
                "unknown",
                checkpoint.file_size or 0,
                checkpoint.page_count or 0,
                checkpoint.extraction_method or "unknown",
            )

        else:
            # Invalid state for resume, start from beginning
            logger.warning(
                "Cannot resume from state, starting fresh",
                state=job_state.state,
            )
            return await self._process_from_start(job_state)

        return {
            "doc_id": checkpoint.doc_id,
            "chunks": checkpoint.chunks_processed,
        }

    async def _download_file(self, job_state: IndexingJobState) -> tuple[bytes, str, str]:
        """Download file from storage"""
        try:
            logger.info("Downloading file", job_id=job_state.job_id)

            storage_adapter = get_storage_adapter()

            # Parse file path from URL
            if "://" in job_state.document_url:
                file_path = job_state.document_url.split("/")[-3:]
                file_path = "/".join(file_path)
            else:
                file_path = job_state.document_url

            blob_content = await storage_adapter.download_file(file_path)
            content = blob_content.data
            file_name = blob_content.metadata.name
            content_type = blob_content.metadata.content_type

            # Update checkpoint
            job_state.checkpoint.file_downloaded = True
            job_state.checkpoint.file_size = len(content)
            job_state.checkpoint.content_type = content_type
            job_state.state = IndexingState.DOWNLOADED

            await self.state_manager.save_state(job_state)

            logger.info(
                "File downloaded",
                job_id=job_state.job_id,
                file_name=file_name,
                size=len(content),
            )

            return content, file_name, content_type

        except Exception as e:
            logger.error("Download failed", job_id=job_state.job_id, error=e)
            raise

    async def _extract_text(
        self,
        job_state: IndexingJobState,
        content: bytes,
        file_name: str,
        content_type: str,
    ) -> tuple[str, str, int]:
        """Extract text from document"""
        try:
            job_state.state = IndexingState.EXTRACTING
            await self.state_manager.save_state(job_state)

            logger.info("Extracting text", job_id=job_state.job_id, file_name=file_name)

            processor_factory = get_document_processor_factory()

            processed_doc = await processor_factory.process_document(
                content=content,
                file_name=file_name,
                content_type=content_type,
            )

            extracted_text = processed_doc.text
            extraction_method = processed_doc.metadata.get("extraction_method", "unknown")
            page_count = processed_doc.page_count or 0

            # Cache extracted text for resume
            await self._cache_extracted_text(job_state.job_id, extracted_text)

            # Update checkpoint
            job_state.checkpoint.text_extracted = True
            job_state.checkpoint.extracted_text_path = file_name
            job_state.checkpoint.extraction_method = extraction_method
            job_state.checkpoint.page_count = page_count
            job_state.state = IndexingState.EXTRACTED

            await self.state_manager.save_state(job_state)

            logger.info(
                "Text extracted",
                job_id=job_state.job_id,
                text_length=len(extracted_text),
                method=extraction_method,
            )

            return extracted_text, extraction_method, page_count

        except Exception as e:
            logger.error("Text extraction failed", job_id=job_state.job_id, error=e)
            raise

    async def _index_document(
        self,
        job_state: IndexingJobState,
        text: str,
        file_name: str,
        extraction_method: str,
    ) -> tuple[str, int]:
        """Index document with LightRAG"""
        try:
            job_state.state = IndexingState.INDEXING
            await self.state_manager.save_state(job_state)

            logger.info("Indexing document", job_id=job_state.job_id)

            doc_id, chunks = await self._index_with_lightrag(
                text=text,
                workspace_id=job_state.workspace_id,
                file_name=file_name,
            )

            # Update checkpoint
            job_state.checkpoint.indexed = True
            job_state.checkpoint.doc_id = doc_id
            job_state.checkpoint.chunks_processed = chunks
            job_state.state = IndexingState.INDEXED

            await self.state_manager.save_state(job_state)

            logger.info(
                "Document indexed",
                job_id=job_state.job_id,
                doc_id=doc_id,
                chunks=chunks,
            )

            return doc_id, chunks

        except Exception as e:
            logger.error("Indexing failed", job_id=job_state.job_id, error=e)
            raise

    async def _update_metadata(
        self,
        job_state: IndexingJobState,
        doc_id: str,
        file_name: str,
        file_size: int,
        page_count: int,
        extraction_method: str,
    ) -> None:
        """Update database metadata"""
        try:
            job_state.state = IndexingState.UPDATING_METADATA
            await self.state_manager.save_state(job_state)

            logger.info("Updating metadata", job_id=job_state.job_id, doc_id=doc_id)

            await self._create_document_metadata(
                doc_id=doc_id,
                workspace_id=job_state.workspace_id,
                kb_id=job_state.kb_id,
                file_name=file_name,
                file_path=job_state.document_url,
                file_size=file_size,
                page_count=page_count,
                metadata={
                    "extraction_method": extraction_method,
                    "job_id": job_state.job_id,
                },
            )

            # Update checkpoint
            job_state.checkpoint.metadata_updated = True

            await self.state_manager.save_state(job_state)

            logger.info("Metadata updated", job_id=job_state.job_id, doc_id=doc_id)

        except Exception as e:
            logger.error("Metadata update failed", job_id=job_state.job_id, error=e)
            raise

    async def _cache_extracted_text(self, job_id: str, text: str) -> None:
        """Cache extracted text for resume"""
        try:
            cache_file = self.cache_dir / f"{job_id}_text.txt"
            cache_file.write_text(text, encoding="utf-8")

            logger.debug("Cached extracted text", job_id=job_id)

        except Exception as e:
            logger.warning("Failed to cache extracted text", job_id=job_id, error=e)

    async def _load_cached_text(self, job_id: str) -> Optional[str]:
        """Load cached extracted text"""
        try:
            cache_file = self.cache_dir / f"{job_id}_text.txt"

            if not cache_file.exists():
                return None

            text = cache_file.read_text(encoding="utf-8")

            logger.debug("Loaded cached text", job_id=job_id)

            return text

        except Exception as e:
            logger.warning("Failed to load cached text", job_id=job_id, error=e)
            return None

    async def _cleanup_cache(self, job_id: str) -> None:
        """Clean up cached files"""
        try:
            cache_file = self.cache_dir / f"{job_id}_text.txt"
            if cache_file.exists():
                cache_file.unlink()

            logger.debug("Cleaned up cache", job_id=job_id)

        except Exception as e:
            logger.warning("Failed to cleanup cache", job_id=job_id, error=e)

    async def _index_with_lightrag(
        self, text: str, workspace_id: int, file_name: str
    ) -> tuple[str, int]:
        """Index with LightRAG (same as before)"""
        try:
            from core.config import settings
            from lightrag import LightRAG
            from lightrag.llm.azure_openai import azure_openai_complete_if_cache
            from lightrag.llm.ollama import ollama_embed
            from lightrag.utils import EmbeddingFunc

            working_dir = f"./lightrag_data/workspace_{workspace_id}"

            embedding_func = EmbeddingFunc(
                embedding_dim=settings.llm.OLLAMA_MODEL_EMBEDDING_MODEL_DIMS,
                max_token_size=settings.llm.OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS,
                func=lambda texts: ollama_embed(
                    texts,
                    embed_model=settings.llm.OLLAMA_MODEL_EMBEDDING_MODEL,
                    host=settings.llm.OLLAMA_MODEL_BASE_URL,
                ),
            )

            async def llm_model_func(
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: Optional[list] = None,
                **kwargs,
            ) -> str:
                return await azure_openai_complete_if_cache(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    api_key=settings.llm.AZURE_OPENAI_LLM_MODEL_API_KEY,
                    api_base=settings.llm.AZURE_OPENAI_LLM_MODEL_API_BASE,
                    api_version=settings.llm.AZURE_OPENAI_LLM_MODEL_API_VERSION,
                    model=settings.llm.AZURE_OPENAI_LLM_MODEL_LLM_MODEL,
                    **kwargs,
                )

            rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                graph_storage="Neo4JStorage",
                vector_storage="PostgreSQLVectorStorage",
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
            )

            await rag.ainsert(text)

            doc_id = f"doc-{hashlib.md5((file_name + str(workspace_id)).encode()).hexdigest()[:12]}"

            chunks = self._estimate_chunk_count(text)

            return doc_id, chunks

        except Exception as e:
            logger.error("Failed to index with LightRAG", error=e)
            raise

    async def _create_document_metadata(
        self,
        doc_id: str,
        workspace_id: int,
        kb_id: Optional[int],
        file_name: str,
        file_path: str,
        file_size: int,
        page_count: Optional[int],
        metadata: Dict,
    ) -> None:
        """Create document metadata (same as before)"""
        try:
            from core.database import DocumentMetadata

            async with db_manager.get_session() as session:
                doc_metadata = DocumentMetadata(
                    doc_id=doc_id,
                    file_name=file_name,
                    workspace_id=workspace_id,
                    file_path=file_path,
                    file_size=file_size,
                    chunk_count=self._estimate_chunk_count_from_size(file_size),
                    metadata={
                        **metadata,
                        "kb_id": kb_id,
                        "page_count": page_count,
                    },
                    indexed_at=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )

                session.add(doc_metadata)
                await session.commit()

        except Exception as e:
            logger.error("Failed to create document metadata", error=e)
            raise

    def _estimate_chunk_count(self, text: str) -> int:
        """Estimate chunk count"""
        estimated_tokens = len(text) / 4
        chunk_size = 1200
        chunk_overlap = 100

        if estimated_tokens <= chunk_size:
            return 1

        effective_chunk_size = chunk_size - chunk_overlap
        chunks = int((estimated_tokens - chunk_overlap) / effective_chunk_size) + 1

        return max(1, chunks)

    def _estimate_chunk_count_from_size(self, file_size: int) -> int:
        """Estimate chunks from file size"""
        return max(1, file_size // 5000)


# Singleton instance
_processor_instance: Optional[DocumentProcessorWithRetry] = None


def get_document_processor_with_retry() -> DocumentProcessorWithRetry:
    """Get or create singleton processor"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DocumentProcessorWithRetry()
    return _processor_instance

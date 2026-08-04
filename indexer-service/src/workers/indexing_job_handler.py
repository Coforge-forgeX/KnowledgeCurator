"""
Indexing Job Handler - Processes documents from upload_and_index API

Receives jobs from Azure Storage Queue, processes documents using LightRAG,
and updates file_tasks status.
"""

import asyncio
import base64
import io
import inspect
import os
from datetime import datetime, timezone
from typing import Optional, Tuple

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.storage.blob import BlobServiceClient
from docx import Document
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


async def initialize_lightrag(domain: str, kb_name: str) -> LightRAG:
    """
    Initialize LightRAG instance for specific knowledge base.

    Args:
        domain: Domain name
        kb_name: Knowledge base name

    Returns:
        Initialized LightRAG instance
    """
    # Create workspace name (alphanumeric only)
    workspace_name = ''.join(c for c in f"{domain}{kb_name}" if c.isalpha())

    # Set Neo4j database
    os.environ['NEO4J_DATABASE'] = workspace_name

    # Set PostgreSQL environment variables for LightRAG
    os.environ['POSTGRES_HOST'] = settings.database.POSTGRESQL_DATABASE_HOST
    os.environ['POSTGRES_USER'] = settings.database.POSTGRESQL_DATABASE_USER
    os.environ['POSTGRES_PASSWORD'] = settings.database.POSTGRESQL_DATABASE_PASSWORD
    os.environ['POSTGRES_DATABASE'] = settings.database.POSTGRESQL_DATABASE_DATABASE
    os.environ['POSTGRES_PORT'] = str(settings.database.POSTGRESQL_DATABASE_PORT)

    # Set Neo4j environment variables for LightRAG
    os.environ['NEO4J_URI'] = settings.database.NEO4J_DATABASE_NEO4J_BOLT_URI
    os.environ['NEO4J_USERNAME'] = settings.database.NEO4J_DATABASE_NEO4J_USER
    os.environ['NEO4J_PASSWORD'] = settings.database.NEO4J_DATABASE_NEO4J_PASSWORD

    # Provide model_name so LightRAG can suffix vector tables for workspace isolation.
    embedding_model_name = settings.lightrag.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    # Initialize LightRAG with Neo4j + PostgreSQL
    rag = LightRAG(
        working_dir=settings.lightrag.WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=settings.lightrag.EMBEDDING_DIM,
            max_token_size=settings.lightrag.MAX_TOKEN_SIZE,
            func=embedding_func,
            model_name=embedding_model_name,
        ),
        graph_storage=settings.lightrag.GRAPH_STORAGE_TYPE,
        workspace=workspace_name,
        vector_storage=settings.lightrag.VECTOR_STORAGE_TYPE,
        chunk_token_size=settings.lightrag.CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=settings.lightrag.CHUNK_OVERLAP_TOKEN_SIZE,
    )

    await rag.initialize_storages()

    logger.info("LightRAG initialized", workspace=workspace_name)
    return rag


async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """LLM function for LightRAG (Azure OpenAI)"""
    import aiohttp

    headers = {
        "Content-Type": "application/json",
        "api-key": settings.lightrag.AZURE_OPENAI_LLM_API_KEY,
    }

    endpoint = (
        f"{settings.lightrag.AZURE_OPENAI_LLM_API_BASE}"
        f"openai/deployments/{settings.lightrag.AZURE_OPENAI_LLM_DEPLOYMENT}/chat/completions"
        f"?api-version={settings.lightrag.AZURE_OPENAI_LLM_API_VERSION}"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0),
        "top_p": kwargs.get("top_p", 1),
        "n": kwargs.get("n", 1),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(f"Request failed: {await response.text()}")
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def embedding_func(texts: list[str]):
    """Embedding function for LightRAG (Azure OpenAI)"""
    import aiohttp
    import numpy as np

    # Validate configuration
    if not settings.lightrag.AZURE_OPENAI_EMBEDDING_API_BASE:
        raise ValueError("AZURE_OPENAI_EMBEDDING_API_BASE is not configured")
    if not settings.lightrag.AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT is not configured")
    if not settings.lightrag.AZURE_OPENAI_EMBEDDING_API_KEY:
        raise ValueError("AZURE_OPENAI_EMBEDDING_API_KEY is not configured")

    headers = {
        "Content-Type": "application/json",
        "api-key": settings.lightrag.AZURE_OPENAI_EMBEDDING_API_KEY,
    }

    endpoint = (
        f"{settings.lightrag.AZURE_OPENAI_EMBEDDING_API_BASE}"
        f"openai/deployments/{settings.lightrag.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}/embeddings"
        f"?api-version={settings.lightrag.AZURE_OPENAI_EMBEDDING_API_VERSION}"
    )

    payload = {
        "input": texts,
        "dimensions": settings.lightrag.EMBEDDING_DIM
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(f"Request failed: {await response.text()}")
            result = await response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)


async def download_file_from_blob(
    storage_adapter, blob_path: str
) -> Tuple[bool, Optional[bytes], Optional[str]]:
    """
    Download file from storage using the storage adapter.

    Args:
        storage_adapter: Storage adapter instance
        blob_path: Path to blob/file

    Returns:
        (success, file_bytes, error_message)
    """
    try:
        # Support both async and sync storage adapters.
        download_method = storage_adapter.download
        if inspect.iscoroutinefunction(download_method):
            blob_data = await download_method(blob_path)
        else:
            blob_data = await asyncio.to_thread(download_method, blob_path)

        logger.info("File downloaded from storage", blob_path=blob_path)
        return True, blob_data, None

    except Exception as e:
        logger.error("File download failed", error_msg=str(e), blob_path=blob_path)
        return False, None, str(e)


async def extract_text_from_file(
    file_bytes: bytes, file_path: str
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Extract text content from file based on extension.

    Returns:
        (success, text_content, error_message)
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".txt", ".md"]:
            content = file_bytes.decode("utf-8", errors="replace")
            return True, content, None

        elif ext == ".pdf":
            # Use Azure Document Intelligence for PDF
            endpoint = settings.azure.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
            api_key = settings.azure.AZURE_DOCUMENT_INTELLIGENCE_KEY

            if not endpoint or not api_key:
                return False, None, "Document Intelligence not configured"

            doc_client = DocumentIntelligenceClient(
                endpoint, AzureKeyCredential(api_key)
            )

            poller = await asyncio.to_thread(
                doc_client.begin_analyze_document,
                "prebuilt-read",
                body=AnalyzeDocumentRequest(bytes_source=file_bytes),
                locale="en-US"
            )

            result = await asyncio.to_thread(poller.result)
            content = result.content
            return True, content, None

        elif ext == ".docx":
            doc = Document(io.BytesIO(file_bytes))
            content = "\n".join([p.text for p in doc.paragraphs])
            return True, content, None

        else:
            return False, None, f"Unsupported file type: {ext}"

    except Exception as e:
        logger.error("Text extraction failed", error_msg=str(e), file_path=file_path)
        return False, None, str(e)


async def index_document_with_lightrag(
    rag: LightRAG, text_content: str, file_path: str
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Index document content into LightRAG.

    Returns:
        (success, chunk_count, error_message)
    """
    try:
        # Chunk text (2000 chars per chunk)
        chunk_size = 2000
        chunks = [
            text_content[i:i+chunk_size]
            for i in range(0, len(text_content), chunk_size)
        ]

        # Insert chunks into LightRAG
        for idx, chunk in enumerate(chunks):
            await rag.ainsert(input=chunk, file_paths=[file_path])
            logger.debug(
                "Chunk indexed",
                chunk_num=idx + 1,
                total_chunks=len(chunks),
                file_path=file_path
            )

        logger.info(
            "Document indexed successfully",
            chunk_count=len(chunks),
            file_path=file_path
        )

        return True, len(chunks), None

    except Exception as e:
        logger.error("Indexing failed", error_msg=str(e), file_path=file_path)
        return False, None, str(e)


async def update_file_task_status(
    task_id: int, status: str, error_message: Optional[str] = None
) -> bool:
    """
    Update file_tasks status in database using SQLAlchemy ORM.

    Returns:
        True if successful, False otherwise
    """
    try:
        from src.core.database import get_async_session, FileTask
        from sqlalchemy import select

        async with get_async_session() as session:
            # Fetch the file task
            stmt = select(FileTask).where(FileTask.id == task_id)
            result = await session.execute(stmt)
            file_task = result.scalar_one_or_none()

            if not file_task:
                logger.warning("File task not found", task_id=task_id)
                return False

            # Update status
            file_task.status = status
            if error_message:
                file_task.error_message = error_message
            # file_tasks.updated_at is TIMESTAMP WITHOUT TIME ZONE.
            file_task.updated_at = datetime.utcnow()

            logger.info(
                "File task status updated",
                task_id=task_id,
                status=status
            )
            return True

    except Exception as e:
        logger.error(
            "Failed to update task status",
            error_msg=str(e),
            task_id=task_id
        )
        return False


async def file_task_exists(task_id: Optional[int]) -> Optional[bool]:
    """Check whether a file task exists.

    Returns:
        True if task exists, False if not found, None if lookup failed.
    """
    if task_id is None:
        return False

    try:
        from src.core.database import FileTask, get_async_session
        from sqlalchemy import select

        async with get_async_session() as session:
            stmt = select(FileTask.id).where(FileTask.id == task_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none() is not None
    except Exception as e:
        logger.error(
            "Failed to check file task existence",
            task_id=task_id,
            error_msg=str(e),
        )
        return None


async def get_file_task_storage_hints(task_id: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    """Fetch storage hints from file_tasks row.

    Returns:
        (container_name, file_path) if task exists, otherwise (None, None)
    """
    if task_id is None:
        return None, None

    try:
        from src.core.database import FileTask, get_async_session
        from sqlalchemy import select

        async with get_async_session() as session:
            stmt = select(FileTask.container_name, FileTask.file_path).where(FileTask.id == task_id)
            result = await session.execute(stmt)
            row = result.first()
            if not row:
                return None, None
            return row[0], row[1]
    except Exception as e:
        logger.warning(
            "Failed to load file task storage hints",
            task_id=task_id,
            error_msg=str(e),
        )
        return None, None


async def create_or_update_indexing_job(
    job_id: str,
    workspace_id: int,
    file_path: str,
    state: str,
    retry_count: int = 0,
    error: Optional[str] = None,
    checkpoint: Optional[dict] = None,
    kb_id: Optional[int] = None
) -> bool:
    """
    Create or update indexing_jobs record for detailed tracking.

    Args:
        job_id: Unique job identifier
        workspace_id: Workspace ID
        file_path: File path
        state: Current state (pending, downloading, processing, completed, failed, etc.)
        retry_count: Number of retry attempts
        error: Error message if failed
        checkpoint: Checkpoint data for resume
        kb_id: Knowledge base ID

    Returns:
        True if successful
    """
    try:
        from src.core.database import get_async_session, IndexingJob
        from sqlalchemy import select

        async with get_async_session() as session:
            # Try to fetch existing job
            stmt = select(IndexingJob).where(IndexingJob.job_id == job_id)
            result = await session.execute(stmt)
            indexing_job = result.scalar_one_or_none()

            now = datetime.now(timezone.utc)

            if indexing_job:
                # Update existing job
                indexing_job.state = state
                indexing_job.retry_count = retry_count
                if error:
                    indexing_job.last_error = error
                if checkpoint:
                    indexing_job.checkpoint_data = checkpoint
                indexing_job.updated_at = now

                # Set timestamps based on state
                if state == "processing" and not indexing_job.started_at:
                    indexing_job.started_at = now
                elif state in ["completed", "failed"]:
                    indexing_job.completed_at = now

            else:
                # Create new job
                indexing_job = IndexingJob(
                    job_id=job_id,
                    workspace_id=workspace_id,
                    document_url=file_path,
                    kb_id=kb_id,
                    state=state,
                    retry_count=retry_count,
                    last_error=error,
                    checkpoint_data=checkpoint or {},
                    created_at=now,
                    started_at=now if state == "processing" else None,
                    completed_at=now if state in ["completed", "failed"] else None,
                    updated_at=now
                )
                session.add(indexing_job)

            logger.info(
                "Indexing job tracked",
                job_id=job_id,
                state=state,
                retry_count=retry_count
            )
            return True

    except Exception as e:
        logger.error(
            "Failed to track indexing job",
            error_msg=str(e),
            job_id=job_id
        )
        return False


async def process_indexing_job(job_data: dict, retry_count: int = 0) -> dict:
    """
    Process a single indexing job with full state tracking.

    Args:
        job_data: Job data from queue message
        retry_count: Number of times this job has been retried (from dequeue_count)

    Returns:
        Result dict with success/error information
    """
    from shared.adapters.storage import get_storage_adapter as _get_storage_adapter

    def get_storage_adapter(container_override: Optional[str] = None):
        """Get storage adapter configured with indexer-service settings"""
        from src.core.config import settings
        connection_string = (
            settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
            or settings.azure.AZURE_STORAGE_CONNECTION_STRING
        )

        return _get_storage_adapter(
            provider=settings.storage.STORAGE_PROVIDER or "azure",
            connection_string=connection_string,
            container_name=container_override or settings.storage.STORAGE_CONTAINER_NAME,
        )

    def resolve_container_name(preferred: Optional[str], default: str) -> str:
        """Resolve target blob container for this job.

        Priority:
        1. Message-supplied container_name
        2. Workspace container env (AZURE_BLOB_STORAGE_WORKSPACE_CONTAINER_NAME)
           when file path indicates workspace uploads
        3. Default configured container
        """
        workspace_container = os.getenv(
            "AZURE_BLOB_STORAGE_WORKSPACE_CONTAINER_NAME",
            "workspace",
        )
        if file_path and file_path.lower().startswith("other/"):
            # For workspace uploads, prefer workspace container unless caller provided
            # a non-default explicit container.
            if preferred and preferred != default:
                return preferred
            return workspace_container

        if preferred:
            return preferred
        return default

    task_id = job_data.get("task_id")
    job_id = job_data.get("job_id", str(task_id))  # Use job_id or fall back to task_id
    workspace_id = job_data.get("workspace_id")
    file_path = job_data.get("file_path")
    domain = job_data.get("domain")
    kb_name = job_data.get("kb_name")
    kb_id = job_data.get("kb_id")
    container_name = job_data.get("container_name")

    logger.info(
        "Processing indexing job",
        job_id=job_id,
        task_id=task_id,
        workspace_id=workspace_id,
        file_path=file_path,
        retry_count=retry_count
    )

    if task_id is None:
        error_msg = "Missing task_id in queue message"
        logger.warning(error_msg, job_id=job_id, file_path=file_path)
        await create_or_update_indexing_job(
            job_id=job_id,
            workspace_id=workspace_id,
            file_path=file_path,
            state="failed",
            retry_count=retry_count,
            error=error_msg,
            kb_id=kb_id,
        )
        return {"success": False, "error": error_msg, "non_retryable": True}

    task_exists = await file_task_exists(task_id)
    if task_exists is False:
        error_msg = f"File task {task_id} not found"
        logger.warning(
            "Skipping stale indexing message",
            task_id=task_id,
            job_id=job_id,
            retry_count=retry_count,
        )
        await create_or_update_indexing_job(
            job_id=job_id,
            workspace_id=workspace_id,
            file_path=file_path,
            state="failed",
            retry_count=retry_count,
            error=error_msg,
            kb_id=kb_id,
        )
        return {"success": False, "error": error_msg, "non_retryable": True}

    try:
        # Enrich missing queue metadata from source file_tasks row.
        file_task_container, file_task_path = await get_file_task_storage_hints(task_id)
        if not container_name and file_task_container:
            container_name = file_task_container
        if not file_path and file_task_path:
            file_path = file_task_path

        # Resolve target container before creating adapter to avoid misleading initialization.
        from src.core.config import settings
        default_container = settings.storage.STORAGE_CONTAINER_NAME
        target_container = resolve_container_name(container_name, default_container)

        # Get storage adapter
        storage = get_storage_adapter(container_override=target_container)

        # Track job start
        await create_or_update_indexing_job(
            job_id=job_id,
            workspace_id=workspace_id,
            file_path=file_path,
            state="processing",
            retry_count=retry_count,
            kb_id=kb_id
        )

        # 1. Update status to processing
        await update_file_task_status(task_id, "processing")

        # 2. Download file from storage
        await create_or_update_indexing_job(job_id, workspace_id, file_path, "downloading", retry_count, kb_id=kb_id)

        # Defensive fallback in case adapter ignored override.
        if target_container != getattr(storage, "container_name", target_container):
            from shared.adapters.storage.adapters.azure_blob import AzureBlobStorageAdapter
            storage = AzureBlobStorageAdapter(
                connection_string=(
                    settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
                    or settings.azure.AZURE_STORAGE_CONNECTION_STRING
                ),
                container_name=target_container,
            )

        success, file_bytes, error = await download_file_from_blob(
            storage, file_path
        )
        if not success:
            await create_or_update_indexing_job(job_id, workspace_id, file_path, "failed", retry_count, error, kb_id=kb_id)
            await update_file_task_status(task_id, "failed", error)
            return {"success": False, "error": error}

        await create_or_update_indexing_job(job_id, workspace_id, file_path, "downloaded", retry_count, kb_id=kb_id)

        # 3. Extract text from file
        await create_or_update_indexing_job(job_id, workspace_id, file_path, "extracting", retry_count, kb_id=kb_id)

        success, text_content, error = await extract_text_from_file(
            file_bytes, file_path
        )
        if not success:
            await create_or_update_indexing_job(job_id, workspace_id, file_path, "failed", retry_count, error, kb_id=kb_id)
            await update_file_task_status(task_id, "failed", error)
            return {"success": False, "error": error}

        await create_or_update_indexing_job(
            job_id, workspace_id, file_path, "extracted", retry_count,
            checkpoint={"text_length": len(text_content), "file_size": len(file_bytes)},
            kb_id=kb_id
        )

        # 4. Initialize LightRAG
        await create_or_update_indexing_job(job_id, workspace_id, file_path, "initializing", retry_count, kb_id=kb_id)
        rag = await initialize_lightrag(domain, kb_name)

        # 5. Index document
        await create_or_update_indexing_job(job_id, workspace_id, file_path, "indexing", retry_count, kb_id=kb_id)

        success, chunk_count, error = await index_document_with_lightrag(
            rag, text_content, file_path
        )
        if not success:
            await create_or_update_indexing_job(job_id, workspace_id, file_path, "failed", retry_count, error, kb_id=kb_id)
            await update_file_task_status(task_id, "failed", error)
            return {"success": False, "error": error}

        await create_or_update_indexing_job(
            job_id, workspace_id, file_path, "indexed", retry_count,
            checkpoint={"chunk_count": chunk_count},
            kb_id=kb_id
        )

        # Reflect indexing completion on file task before final completion state.
        await update_file_task_status(task_id, "indexed")

        # 6. Update status to completed
        await create_or_update_indexing_job(job_id, workspace_id, file_path, "completed", retry_count, kb_id=kb_id)
        await update_file_task_status(task_id, "completed")

        logger.info(
            "Indexing job completed",
            task_id=task_id,
            chunk_count=chunk_count
        )

        return {
            "success": True,
            "task_id": task_id,
            "chunk_count": chunk_count
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(
            "Indexing job failed",
            error_msg=error_msg,
            job_id=job_id,
            task_id=task_id,
            retry_count=retry_count,
            exc_info=True
        )
        await create_or_update_indexing_job(job_id, workspace_id, file_path, "failed", retry_count, error_msg, kb_id=kb_id)
        await update_file_task_status(task_id, "failed", error_msg)
        return {"success": False, "error": error_msg}

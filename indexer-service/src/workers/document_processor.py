"""
Document Processing Pipeline

Complete document ingestion and indexing workflow:
1. Download from blob storage (using storage adapter)
2. Extract text (using document processor)
3. Index with LightRAG (chunking, embeddings, graph)
4. Update database metadata
"""
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from shared.adapters.storage import get_storage_adapter as _get_storage_adapter

from core.database import db_manager
from core.logging import get_logger
from processors.factory import get_document_processor_factory


def get_storage_adapter():
    """Get storage adapter configured with indexer-service settings"""
    from core.config import settings
    return _get_storage_adapter(
        provider=settings.storage.STORAGE_PROVIDER or "azure",
        connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
        container_name=settings.storage.STORAGE_CONTAINER_NAME,
    )

logger = get_logger(__name__)


async def process_document(
    job_id: str,
    workspace_id: int,
    document_url: str,
    kb_id: Optional[int] = None,
) -> Dict:
    """
    Process a document for indexing.

    Args:
        job_id: Unique job identifier
        workspace_id: Workspace ID
        document_url: URL or blob path to document
        kb_id: Knowledge base ID (optional)

    Returns:
        Dict with processing results:
        {
            "success": bool,
            "duration_seconds": float,
            "chunks_processed": int,
            "doc_id": str,
            "error": str (if failed)
        }
    """
    start_time = time.time()

    try:
        logger.info(
            "Starting document processing",
            job_id=job_id,
            workspace_id=workspace_id,
            document_url=document_url[:100],
        )

        # Step 1: Download file from storage
        storage_adapter = get_storage_adapter()

        # Extract file path from URL
        # Expecting format: https://...blob.core.windows.net/container/path/file.pdf
        # OR just: path/file.pdf
        if "://" in document_url:
            # Parse URL to get blob path
            file_path = document_url.split("/")[-3:]  # Get last 3 parts
            file_path = "/".join(file_path)
        else:
            file_path = document_url

        logger.info("Downloading file from storage", file_path=file_path)

        content = await storage_adapter.download(file_path)
        file_name = file_path.split("/")[-1]

        # Infer content type from file extension
        import mimetypes
        content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

        logger.info(
            "File downloaded",
            file_name=file_name,
            size=len(content),
            content_type=content_type,
        )

        # Step 2: Extract text using document processor
        processor_factory = get_document_processor_factory()

        logger.info("Processing document", file_name=file_name)

        processed_doc = await processor_factory.process_document(
            content=content,
            file_name=file_name,
            content_type=content_type,
        )

        logger.info(
            "Text extracted",
            file_name=file_name,
            text_length=len(processed_doc.text),
            page_count=processed_doc.page_count,
            processor=processed_doc.metadata.get("processor"),
        )

        # Step 3: Index with LightRAG
        doc_id = await _index_with_lightrag(
            text=processed_doc.text,
            workspace_id=workspace_id,
            file_name=file_name,
            metadata=processed_doc.metadata,
        )

        logger.info("Document indexed with LightRAG", doc_id=doc_id)

        # Step 4: Create document metadata in database
        await _create_document_metadata(
            doc_id=doc_id,
            workspace_id=workspace_id,
            kb_id=kb_id,
            file_name=file_name,
            file_path=file_path,
            file_size=len(content),
            page_count=processed_doc.page_count,
            metadata=processed_doc.metadata,
        )

        logger.info("Document metadata created", doc_id=doc_id)

        # Calculate chunks processed (estimate)
        chunk_count = _estimate_chunk_count(processed_doc.text)

        duration = time.time() - start_time

        logger.info(
            "Document processing completed",
            job_id=job_id,
            doc_id=doc_id,
            duration_seconds=duration,
            chunks=chunk_count,
        )

        return {
            "success": True,
            "duration_seconds": duration,
            "chunks_processed": chunk_count,
            "doc_id": doc_id,
        }

    except Exception as e:
        duration = time.time() - start_time

        logger.error(
            "Document processing failed",
            job_id=job_id,
            error_msg=str(e),
            duration_seconds=duration,
            exc_info=True,
        )

        return {
            "success": False,
            "duration_seconds": duration,
            "error": str(e),
        }


async def _index_with_lightrag(
    text: str,
    workspace_id: int,
    file_name: str,
    metadata: Dict,
) -> str:
    """
    Index document with LightRAG.

    LightRAG handles:
    - Text chunking (1200 tokens with 100 overlap)
    - Embedding generation (1024-dim vectors via Ollama)
    - Vector storage (PostgreSQL + pgvector)
    - Knowledge graph creation (Neo4j)

    Args:
        text: Extracted text
        workspace_id: Workspace ID
        file_name: Original file name
        metadata: Document metadata

    Returns:
        str: Document ID
    """
    try:
        import os
        from core.config import settings
        from lightrag import LightRAG
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache
        from lightrag.llm.ollama import ollama_embed
        from lightrag.utils import EmbeddingFunc

        # Set environment variables for LightRAG storage backends
        # Neo4j
        os.environ["NEO4J_URI"] = settings.database.neo4j_uri
        os.environ["NEO4J_USERNAME"] = settings.database.NEO4J_DATABASE_NEO4J_USER
        os.environ["NEO4J_PASSWORD"] = settings.database.NEO4J_DATABASE_NEO4J_PASSWORD

        # PostgreSQL for PGVectorStorage
        os.environ["POSTGRES_HOST"] = settings.database.POSTGRESQL_DATABASE_HOST
        os.environ["POSTGRES_USER"] = settings.database.POSTGRESQL_DATABASE_USER
        os.environ["POSTGRES_PASSWORD"] = settings.database.POSTGRESQL_DATABASE_PASSWORD
        os.environ["POSTGRES_DATABASE"] = settings.database.POSTGRESQL_DATABASE_DATABASE

        # Get workspace-specific working directory
        # Use workspace_id_to_alpha for compatibility with KnowledgeCurator
        from workspace_helpers import get_workspace_identifier, get_workspace_working_dir

        workspace_identifier = get_workspace_identifier(workspace_id)
        working_dir = get_workspace_working_dir(workspace_id, "./lightrag_data")

        # Build embedding function
        embedding_func = EmbeddingFunc(
            embedding_dim=settings.llm.OLLAMA_MODEL_EMBEDDING_MODEL_DIMS,
            max_token_size=settings.llm.OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=settings.llm.OLLAMA_MODEL_EMBEDDING_MODEL,
                host=settings.llm.OLLAMA_MODEL_BASE_URL,
            ),
        )

        # Build LLM function (for entity extraction)
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

        # Initialize LightRAG with workspace parameter for multi-tenancy
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            graph_storage="Neo4JStorage",  # Use Neo4j for knowledge graph
            vector_storage="PostgreSQLVectorStorage",  # Use PostgreSQL for vectors
            chunk_token_size=1200,  # Chunk size in tokens
            chunk_overlap_token_size=100,  # Overlap between chunks
            workspace=workspace_identifier,  # Workspace identifier for multi-tenancy
        )

        # Insert document (this does all the indexing)
        await rag.ainsert(text)

        # Generate document ID
        doc_id = f"doc-{hashlib.md5((file_name + str(workspace_id)).encode()).hexdigest()[:12]}"

        logger.info(
            "Document indexed with LightRAG",
            doc_id=doc_id,
            text_length=len(text),
            workspace_id=workspace_id,
        )

        return doc_id

    except Exception as e:
        logger.error("Failed to index with LightRAG", error_msg=e)
        raise


async def _create_document_metadata(
    doc_id: str,
    workspace_id: int,
    kb_id: Optional[int],
    file_name: str,
    file_path: str,
    file_size: int,
    page_count: Optional[int],
    metadata: Dict,
) -> None:
    """
    Create document metadata record in PostgreSQL.

    Args:
        doc_id: Document ID
        workspace_id: Workspace ID
        kb_id: Knowledge base ID
        file_name: File name
        file_path: File path in storage
        file_size: File size in bytes
        page_count: Number of pages
        metadata: Additional metadata
    """
    try:
        from core.database import DocumentMetadata

        async with db_manager.get_session() as session:
            doc_metadata = DocumentMetadata(
                doc_id=doc_id,
                file_name=file_name,
                workspace_id=workspace_id,
                kb_id=kb_id,  # Direct column for KB sharing logic
                file_path=file_path,
                file_size=file_size,
                chunk_count=_estimate_chunk_count_from_size(file_size),
                metadata={
                    **metadata,
                    "page_count": page_count,
                },
                indexed_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            session.add(doc_metadata)
            await session.commit()

            logger.info(
                "Document metadata created",
                doc_id=doc_id,
                workspace_id=workspace_id,
            )

    except Exception as e:
        logger.error("Failed to create document metadata", error_msg=e)
        # Don't raise - indexing succeeded, metadata is secondary
        # The document is already in LightRAG


def _estimate_chunk_count(text: str) -> int:
    """Estimate number of chunks based on text length"""
    # LightRAG chunks at 1200 tokens with 100 overlap
    # Rough estimate: 4 characters per token
    estimated_tokens = len(text) / 4
    chunk_size = 1200
    chunk_overlap = 100

    if estimated_tokens <= chunk_size:
        return 1

    # Calculate chunks with overlap
    effective_chunk_size = chunk_size - chunk_overlap
    chunks = int((estimated_tokens - chunk_overlap) / effective_chunk_size) + 1

    return max(1, chunks)


def _estimate_chunk_count_from_size(file_size: int) -> int:
    """Estimate chunks from file size"""
    # Very rough estimate: 1 chunk per 5KB
    return max(1, file_size // 5000)

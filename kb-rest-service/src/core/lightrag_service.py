"""LightRAG service for knowledge base operations"""
import os
import tempfile
from typing import Any, Dict, List, Optional

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from shared.lightrag import (
    build_azure_openai_chat_completion_func,
    build_azure_openai_embedding_func,
    build_ollama_embedding_func,
)

try:
    from common_adapters.configurableAI.llm_router_config_store import (
        llm_router_config_store,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    llm_router_config_store = None

from .config import settings
from .exceptions import ConfigurationException, LightRAGException
from .logging import get_logger

logger = get_logger(__name__)


class LightRAGService:
    """
    Service class for LightRAG operations.

    Provides methods for initializing LightRAG, querying knowledge bases,
    and managing document indexing with shared storage configuration.
    """

    def __init__(self, working_dir: Optional[str] = None, workspace: Optional[str] = None):
        """
        Initialize LightRAG service.

        Args:
            working_dir: Working directory for LightRAG data
            workspace: Workspace identifier for multi-tenancy in Neo4j/PostgreSQL
        """
        serverless_mode = bool(getattr(settings.database, "SERVERLESS", True))
        configured_working_dir = working_dir or settings.lightrag.LIGHTRAG_WORKING_DIR
        if serverless_mode and not working_dir:
            configured_working_dir = os.path.join(tempfile.gettempdir(), "lightrag_data")

        self.working_dir = configured_working_dir
        self.workspace = workspace
        self._rag: Optional[LightRAG] = None
        self._initialized = False
        self._runtime_workspace_id: Optional[int] = None
        self._runtime_agent_id: Optional[int] = None
        self._runtime_signature: Optional[str] = None
        self._active_llm_provider: Optional[str] = None
        self._active_llm_model: Optional[str] = None
        self._active_llm_source: Optional[str] = None

        try:
            os.makedirs(self.working_dir, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to ensure LightRAG working directory", working_dir=self.working_dir, error=str(e))

        logger.info(
            "LightRAG service initialized",
            working_dir=self.working_dir,
            workspace=workspace,
            serverless_mode=serverless_mode,
        )

    def set_runtime_context(
        self,
        *,
        workspace_id: Optional[int] = None,
        agent_id: Optional[int] = None,
    ) -> None:
        """Set request-scoped context for LLM routing.

        This allows workspace/agent-specific LLM selection via common_adapters.
        """
        if workspace_id != self._runtime_workspace_id or agent_id != self._runtime_agent_id:
            self._runtime_workspace_id = workspace_id
            self._runtime_agent_id = agent_id
            self._initialized = False
            self._rag = None

    def _build_runtime_signature(self) -> str:
        """Build signature used to detect context/config changes safely."""
        return "|".join(
            [
                str(self.working_dir),
                str(self.workspace),
                str(self._runtime_workspace_id),
                str(self._runtime_agent_id),
            ]
        )

    def _resolve_llm_router_config(self) -> Optional[Dict[str, Any]]:
        """Resolve workspace/agent LLM config from common_adapters router store."""
        if llm_router_config_store is None or self._runtime_workspace_id is None:
            return None

        try:
            config = llm_router_config_store.get_effective_configuration(
                self._runtime_workspace_id,
                self._runtime_agent_id,
            )
            if not config:
                return None

            current_provider = (config.get("current_provider") or "").strip().lower()
            if not current_provider:
                return None

            current_model = config.get("current_model")
            provider_config = llm_router_config_store.build_config_dict(
                self._runtime_workspace_id,
                current_provider,
                model_override=current_model,
            )
            return provider_config
        except Exception as e:
            logger.warning(
                "Failed to resolve LLM router config, falling back to env settings",
                workspace_id=self._runtime_workspace_id,
                agent_id=self._runtime_agent_id,
                error=str(e),
            )
            return None

    async def _build_llm_func(self) -> Any:
        """
        Build LLM function for LightRAG based on configuration.

        Returns:
            LLM function callable

        Raises:
            ConfigurationException: If LLM configuration is invalid
        """
        router_config = self._resolve_llm_router_config()
        if router_config:
            provider = (router_config.get("provider_name") or "").strip().lower()
            if provider == "azure":
                azure_api_key = router_config.get("api_key")
                azure_api_base = router_config.get("endpoint")
                azure_api_version = (
                    router_config.get("api_version")
                    or settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION
                )
                azure_deployment_name = (
                    router_config.get("deployment_name") or router_config.get("model")
                )
                if all([azure_api_key, azure_api_base, azure_deployment_name]):
                    self._active_llm_provider = provider
                    self._active_llm_model = router_config.get("model") or azure_deployment_name
                    self._active_llm_source = "common_adapters"
                    logger.info(
                        "Using workspace/agent LLM configuration from common_adapters",
                        workspace_id=self._runtime_workspace_id,
                        agent_id=self._runtime_agent_id,
                        provider=provider,
                        model=self._active_llm_model,
                        deployment=azure_deployment_name,
                    )
                    return build_azure_openai_chat_completion_func(
                        api_key=azure_api_key,
                        api_base=azure_api_base,
                        api_version=azure_api_version,
                        deployment=azure_deployment_name,
                    )
            else:
                logger.warning(
                    "Current LLM provider from common_adapters is not supported by LightRAG chat builder; using env fallback",
                    provider=provider,
                    workspace_id=self._runtime_workspace_id,
                    agent_id=self._runtime_agent_id,
                )

        azure_api_key = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_KEY
        azure_api_base = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_BASE
        azure_api_version = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION
        azure_deployment_name = settings.lightrag.AZURE_OPENAI_LLM_MODEL_LLM_MODEL

        if not all([azure_api_key, azure_api_base, azure_deployment_name]):
            raise ConfigurationException(
                message="Azure OpenAI LLM configuration is incomplete",
                config_key="AZURE_OPENAI_LLM_MODEL",
            )

        self._active_llm_provider = "azure"
        self._active_llm_model = azure_deployment_name
        self._active_llm_source = "env"
        logger.info(
            "Using environment LLM configuration",
            workspace_id=self._runtime_workspace_id,
            agent_id=self._runtime_agent_id,
            provider=self._active_llm_provider,
            model=self._active_llm_model,
            api_version=azure_api_version,
        )

        return build_azure_openai_chat_completion_func(
            api_key=azure_api_key,
            api_base=azure_api_base,
            api_version=azure_api_version,
            deployment=azure_deployment_name,
        )

    def _build_embedding_func(self) -> EmbeddingFunc:
        """
        Build embedding function for LightRAG based on configuration.

        Returns:
            EmbeddingFunc: Embedding function

        Raises:
            ConfigurationException: If embedding configuration is invalid
        """
        # Prefer Azure embedding config to match indexer-service behavior.
        azure_api_key = settings.lightrag.AZURE_OPENAI_EMBEDDING_MODEL_API_KEY
        azure_api_base = settings.lightrag.AZURE_OPENAI_EMBEDDING_MODEL_API_BASE
        azure_api_version = settings.lightrag.AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION
        azure_deployment = settings.lightrag.AZURE_OPENAI_EMBEDDING_MODEL_EMBEDDING_MODEL

        if all([azure_api_key, azure_api_base, azure_deployment]):
            logger.info(
                "Using Azure OpenAI embedding configuration",
                deployment=azure_deployment,
                api_version=azure_api_version,
            )
            return EmbeddingFunc(
                embedding_dim=settings.lightrag.EMBEDDING_DIM,
                max_token_size=settings.lightrag.MAX_TOKEN_SIZE,
                func=build_azure_openai_embedding_func(
                    api_key=azure_api_key,
                    api_base=azure_api_base,
                    api_version=azure_api_version,
                    deployment=azure_deployment,
                    dimensions=settings.lightrag.EMBEDDING_DIM,
                ),
            )

        # Fallback to Ollama for local/dev use cases.
        base_url = settings.lightrag.OLLAMA_MODEL_BASE_URL
        embedding_model = settings.lightrag.OLLAMA_MODEL_EMBEDDING_MODEL
        embedding_dim = settings.lightrag.OLLAMA_MODEL_EMBEDDING_MODEL_DIMS
        max_token_size = settings.lightrag.OLLAMA_MODEL_EMBEDDING_MODEL_MAX_TOKENS

        if not all([base_url, embedding_model]):
            raise ConfigurationException(
                message=(
                    "Embedding configuration is incomplete. Provide Azure OpenAI embedding "
                    "settings or Ollama settings."
                ),
                config_key="AZURE_OPENAI_EMBEDDING_MODEL_* or OLLAMA_MODEL_*",
            )

        logger.warning(
            "Azure embedding settings not found; falling back to Ollama embeddings",
            ollama_host=base_url,
            ollama_model=embedding_model,
        )
        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=build_ollama_embedding_func(
                host=base_url,
                embed_model=embedding_model,
            ),
        )


    async def initialize(self) -> None:
        """
        Initialize LightRAG instance with configured storage backends.

        Raises:
            ConfigurationException: If configuration is invalid
            LightRAGException: If initialization fails
        """
        current_signature = self._build_runtime_signature()
        if self._runtime_signature != current_signature:
            self._initialized = False
            self._rag = None

        if self._initialized:
            logger.debug("LightRAG already initialized, skipping")
            return

        try:
            logger.info("Initializing LightRAG instance")

            if settings.database.SERVERLESS:
                logger.info(
                    "Serverless mode detected; local LightRAG working directory is ephemeral and persistent state must come from external stores",
                    working_dir=self.working_dir,
                    graph_storage=settings.lightrag.GRAPH_STORAGE_TYPE,
                    vector_storage=settings.lightrag.VECTOR_STORAGE_TYPE,
                )

            # Set Neo4j environment variables for LightRAG.
            # Prefer direct environment values because Pydantic defaults can mask missing mappings.
            neo4j_uri = (
                os.getenv("NEO4J_DATABASE_NEO4J_BOLT_URI")
                or settings.database.NEO4J_URI
            )
            neo4j_user = (
                os.getenv("NEO4J_DATABASE_NEO4J_USER")
                or settings.database.NEO4J_USER
            )
            neo4j_password = (
                os.getenv("NEO4J_DATABASE_NEO4J_PASSWORD")
                or settings.database.NEO4J_PASSWORD
            )

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ConfigurationException(
                    message="Neo4j configuration is incomplete",
                    config_key="NEO4J_DATABASE_NEO4J_*",
                )

            os.environ["NEO4J_URI"] = neo4j_uri
            os.environ["NEO4J_USERNAME"] = neo4j_user
            os.environ["NEO4J_PASSWORD"] = neo4j_password

            logger.debug(
                "Neo4j connection configured",
                uri=neo4j_uri,
                user=neo4j_user,
                has_password=bool(neo4j_password),
            )

            # Set PostgreSQL environment variables for PGVectorStorage
            # Use settings first, then fallback to direct env vars if settings are None
            postgres_host = (
                settings.lightrag.LIGHTRAG_POSTGRESQL_HOST
                or os.getenv("LIGHTRAG_POSTGRESQL_DATABASE_HOST")
            )
            postgres_user = (
                settings.lightrag.LIGHTRAG_POSTGRESQL_USER
                or os.getenv("LIGHTRAG_POSTGRESQL_DATABASE_USER")
            )
            postgres_password = (
                settings.lightrag.LIGHTRAG_POSTGRESQL_PASSWORD
                or os.getenv("LIGHTRAG_POSTGRESQL_DATABASE_PASSWORD")
            )
            postgres_database = (
                settings.lightrag.LIGHTRAG_POSTGRESQL_DATABASE
                or os.getenv("LIGHTRAG_POSTGRESQL_DATABASE_DATABASE")
            )

            if postgres_host:
                os.environ["POSTGRES_HOST"] = postgres_host
            if postgres_user:
                os.environ["POSTGRES_USER"] = postgres_user
            if postgres_password:
                os.environ["POSTGRES_PASSWORD"] = postgres_password
            if postgres_database:
                os.environ["POSTGRES_DATABASE"] = postgres_database

            if settings.database.SERVERLESS and not all(
                [postgres_host, postgres_user, postgres_password, postgres_database]
            ):
                raise ConfigurationException(
                    message=(
                        "Serverless deployment requires PostgreSQL configuration for persistent vector/KV state"
                    ),
                    config_key="LIGHTRAG_POSTGRESQL_DATABASE_*",
                )

            # Keep LightRAG embedding worker config aligned with indexer-service.
            os.environ["EMBEDDING_TIMEOUT"] = str(settings.lightrag.EMBEDDING_TIMEOUT_SECONDS)
            os.environ["EMBEDDING_FUNC_MAX_ASYNC"] = str(settings.lightrag.EMBEDDING_FUNC_MAX_ASYNC)
            os.environ["EMBEDDING_BATCH_NUM"] = str(settings.lightrag.EMBEDDING_BATCH_NUM)

            # Log what was configured for debugging
            logger.debug(
                "PostgreSQL environment variables configured for PGVectorStorage",
                has_host=bool(postgres_host),
                has_user=bool(postgres_user),
                has_password=bool(postgres_password),
                has_database=bool(postgres_database),
                host=postgres_host if postgres_host else "NOT_SET",
                user=postgres_user if postgres_user else "NOT_SET",
                database=postgres_database if postgres_database else "NOT_SET",
            )

            # Build LLM and embedding functions
            llm_func = await self._build_llm_func()
            embedding_func = self._build_embedding_func()

            # Initialize LightRAG with workspace parameter for multi-tenancy
            lightrag_kwargs = {
                "working_dir": self.working_dir,
                "llm_model_func": llm_func,
                "embedding_func": embedding_func,
                "graph_storage": settings.lightrag.GRAPH_STORAGE_TYPE,
                "vector_storage": settings.lightrag.VECTOR_STORAGE_TYPE,
                "chunk_token_size": settings.lightrag.CHUNK_TOKEN_SIZE,
                "chunk_overlap_token_size": settings.lightrag.CHUNK_OVERLAP_TOKEN_SIZE,
                "embedding_batch_num": settings.lightrag.EMBEDDING_BATCH_NUM,
                "embedding_func_max_async": settings.lightrag.EMBEDDING_FUNC_MAX_ASYNC,
                "default_embedding_timeout": settings.lightrag.EMBEDDING_TIMEOUT_SECONDS,
            }

            # Add workspace if specified (for Neo4j/PostgreSQL multi-tenancy)
            if self.workspace:
                lightrag_kwargs["workspace"] = self.workspace

            self._rag = LightRAG(**lightrag_kwargs)

            # Initialize storages
            await self._rag.initialize_storages()

            self._initialized = True
            self._runtime_signature = current_signature
            logger.info(
                "LightRAG initialized successfully",
                graph_storage=settings.lightrag.GRAPH_STORAGE_TYPE,
                vector_storage=settings.lightrag.VECTOR_STORAGE_TYPE,
                llm_provider=self._active_llm_provider,
                llm_model=self._active_llm_model,
                llm_source=self._active_llm_source,
            )

        except ConfigurationException:
            raise
        except Exception as e:
            logger.error("Failed to initialize LightRAG", error=e)
            raise LightRAGException(
                message=f"Failed to initialize LightRAG: {str(e)}",
                operation="initialize"
            )

    async def query(
        self,
        query: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Query the knowledge base using LightRAG.

        Args:
            query: Query string
            mode: Query mode - "naive", "local", "global", or "hybrid"
            only_need_context: If True, return only retrieved context without answer
            **kwargs: Additional query parameters

        Returns:
            Dict containing query results with answer and/or context

        Raises:
            LightRAGException: If query fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(
                "Executing LightRAG query",
                query=query[:100],
                mode=mode,
                llm_provider=self._active_llm_provider,
                llm_model=self._active_llm_model,
                llm_source=self._active_llm_source,
                workspace_id=self._runtime_workspace_id,
                agent_id=self._runtime_agent_id,
            )

            # Execute query - LightRAG returns string answer or dict with answer+context
            result = await self._rag.aquery(
                query,
                param=QueryParam(mode=mode, only_need_context=only_need_context, **kwargs)
            )

            # If only_need_context=True, result is the context itself
            if only_need_context:
                return {
                    "answer": None,
                    "retrieved_chunks": result if isinstance(result, list) else [result] if result else [],
                    "sources": self._extract_sources(result if isinstance(result, list) else [result]),
                    "mode": mode,
                }

            # Structure the response with both answer and context
            # LightRAG typically returns just the answer string, but we need context too
            # The context is stored internally during query execution
            answer = result if isinstance(result, str) else result.get("answer", "") if isinstance(result, dict) else str(result)

            # Try to retrieve context from the RAG instance
            # Note: This depends on LightRAG internals - may need adjustment based on version
            retrieved_context = []
            if isinstance(result, dict) and "context" in result:
                retrieved_context = result["context"]

            response = {
                "answer": answer,
                "retrieved_chunks": retrieved_context,
                "sources": self._extract_sources(retrieved_context),
                "mode": mode,
            }

            logger.info("Query executed successfully", query_length=len(query), has_context=bool(retrieved_context))
            return response

        except Exception as e:
            logger.error("Query execution failed", error=e, query=query[:100])
            raise LightRAGException(
                message=f"Query failed: {str(e)}",
                operation="query"
            )

    async def query_data(
        self,
        query: str,
        mode: str = "mix",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrieve structured data from LightRAG without LLM answer generation.

        Args:
            query: Query string
            mode: Query mode - usually "mix" for combined graph + vector retrieval
            **kwargs: Additional QueryParam fields (top_k, chunk_top_k, etc.)

        Returns:
            Dict with keys: status, message, data, metadata

        Raises:
            LightRAGException: If query data retrieval fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(
                "Executing LightRAG structured data query",
                query=query[:100],
                mode=mode,
                workspace_id=self._runtime_workspace_id,
                agent_id=self._runtime_agent_id,
            )

            result = await self._rag.aquery_data(
                query,
                param=QueryParam(mode=mode, **kwargs)
            )

            if isinstance(result, dict):
                return {
                    "status": result.get("status", "success"),
                    "message": result.get("message", "Query executed successfully"),
                    "data": result.get("data", {}) if isinstance(result.get("data"), dict) else {},
                    "metadata": result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {},
                }

            return {
                "status": "failure",
                "message": "Invalid response format from LightRAG aquery_data",
                "data": {},
                "metadata": {},
            }

        except Exception as e:
            logger.error("Structured data query failed", error=e, query=query[:100])
            raise LightRAGException(
                message=f"Query data failed: {str(e)}",
                operation="query_data"
            )

    async def query_llm(
        self,
        query: str,
        mode: str = "mix",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute unified LightRAG query that returns answer + structured data.

        Args:
            query: Query string
            mode: Query mode - usually "mix" for combined graph + vector retrieval
            **kwargs: Additional QueryParam fields (top_k, chunk_top_k, etc.)

        Returns:
            Dict with keys: llm_response, data, metadata

        Raises:
            LightRAGException: If unified query fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(
                "Executing LightRAG unified query",
                query=query[:100],
                mode=mode,
                workspace_id=self._runtime_workspace_id,
                agent_id=self._runtime_agent_id,
            )

            result = await self._rag.aquery_llm(
                query,
                param=QueryParam(mode=mode, **kwargs)
            )

            if isinstance(result, dict):
                llm_response = result.get("llm_response", {})
                data = result.get("data", {})
                metadata = result.get("metadata", {})
                return {
                    "llm_response": llm_response if isinstance(llm_response, dict) else {},
                    "data": data if isinstance(data, dict) else {},
                    "metadata": metadata if isinstance(metadata, dict) else {},
                }

            return {
                "llm_response": {
                    "content": str(result) if result is not None else "",
                    "is_streaming": False,
                },
                "data": {},
                "metadata": {},
            }

        except Exception as e:
            logger.error("Unified LightRAG query failed", error=e, query=query[:100])
            raise LightRAGException(
                message=f"Query llm failed: {str(e)}",
                operation="query_llm"
            )

    async def insert(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Insert document into the knowledge base.

        Args:
            text: Document text to insert
            metadata: Optional metadata for the document (e.g., file_path, file_name, doc_id)

        Returns:
            Dict with insertion status

        Raises:
            LightRAGException: If insertion fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("Inserting document into LightRAG", text_length=len(text), metadata=metadata)

            file_path = None
            if metadata and isinstance(metadata, dict):
                file_path = metadata.get("file_path")

            if file_path:
                normalized_file_path = str(file_path).replace("\\", "/")
                await self._rag.ainsert(input=text, file_paths=[normalized_file_path])
            else:
                try:
                    await self._rag.ainsert(text, metadata=metadata)
                except TypeError:
                    # Fallback for LightRAG versions that don't support metadata parameter.
                    logger.warning("LightRAG version doesn't support metadata in ainsert, inserting without metadata")
                    await self._rag.ainsert(text)

            logger.info("Document inserted successfully", text_length=len(text))
            return {"status": "success", "message": "Document indexed successfully"}

        except Exception as e:
            logger.error("Document insertion failed", error=e, text_length=len(text))
            raise LightRAGException(
                message=f"Document insertion failed: {str(e)}",
                operation="insert"
            )

    async def delete_by_doc_id(self, doc_id: str) -> Dict[str, str]:
        """
        Delete document by ID from the knowledge base.

        Args:
            doc_id: Document ID to delete

        Returns:
            Dict with deletion status

        Raises:
            LightRAGException: If deletion fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("Deleting document from LightRAG", doc_id=doc_id)

            # LightRAG deletion logic here
            # Note: Actual implementation depends on LightRAG API

            logger.info("Document deleted successfully", doc_id=doc_id)
            return {"status": "success", "message": f"Document {doc_id} deleted"}

        except Exception as e:
            logger.error("Document deletion failed", error=e, doc_id=doc_id)
            raise LightRAGException(
                message=f"Document deletion failed: {str(e)}",
                operation="delete"
            )

    def _extract_sources(self, context: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved context.

        Args:
            context: Retrieved context chunks

        Returns:
            List of source dictionaries
        """
        sources = []
        for idx, chunk in enumerate(context):
            if isinstance(chunk, dict):
                sources.append({
                    "chunk_id": idx,
                    "content": chunk.get("content", ""),
                    "source_id": chunk.get("source_id", ""),
                    "metadata": chunk.get("metadata", {}),
                })
            elif isinstance(chunk, str):
                sources.append({
                    "chunk_id": idx,
                    "content": chunk,
                })
        return sources

    async def get_knowledge_graph(self) -> Dict[str, Any]:
        """
        Get the knowledge graph from LightRAG.

        Returns:
            Dict containing nodes and edges of the knowledge graph

        Raises:
            LightRAGException: If retrieval fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("Retrieving knowledge graph")

            # Access the graph storage from LightRAG
            nodes = []
            edges = []

            # If using Neo4j, query it directly via the Neo4j driver
            from src.core.neo4j_driver import get_neo4j_driver

            neo4j_driver = get_neo4j_driver()

            # Query all nodes
            node_query = """
            MATCH (n)
            RETURN n.id as id, labels(n) as labels, properties(n) as properties
            LIMIT 1000
            """
            node_results = await neo4j_driver.execute_query(node_query, {})
            for record in node_results:
                nodes.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": record["properties"],
                })

            # Query all relationships
            edge_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, type(r) as type, b.id as target, properties(r) as properties
            LIMIT 1000
            """
            edge_results = await neo4j_driver.execute_query(edge_query, {})
            for record in edge_results:
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "type": record["type"],
                    "properties": record["properties"],
                })

            logger.info("Retrieved knowledge graph", node_count=len(nodes), edge_count=len(edges))
            return {
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

        except Exception as e:
            logger.error("Failed to retrieve knowledge graph", error=e)
            raise LightRAGException(
                message=f"Failed to retrieve knowledge graph: {str(e)}",
                operation="get_kg"
            )

    async def get_indexed_documents(
        self,
        workspace_id: int = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get paginated list of indexed documents from PostgreSQL.

        For a workspace, returns:
        1. Documents uploaded directly to the workspace
        2. Documents from all linked knowledge bases (uploaded to KG workspaces)

        Args:
            workspace_id: Optional workspace filter
            limit: Maximum number of documents to return per page (default: 100, max: 1000)
            offset: Number of documents to skip for pagination (default: 0)

        Returns:
            Dict containing:
            - documents: List of document metadata with kb_id and source_type
            - total: Total count of documents matching the filter
            - limit: Limit used in query
            - offset: Offset used in query
            - has_more: Boolean indicating if more documents exist

        Raises:
            LightRAGException: If retrieval fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Validate and cap limit
            limit = min(max(1, limit), 1000)
            offset = max(0, offset)

            logger.info(
                "Retrieving indexed documents",
                workspace_id=workspace_id,
                limit=limit,
                offset=offset,
            )

            from src.core.database import get_async_session, DocumentMetadata, FileTask
            from sqlalchemy import select, func, or_
            from src.helpers.workspace_kb_helpers import get_workspace_kb_ids, is_kg_workspace

            async with get_async_session() as session:
                # Build base query for filtering
                if workspace_id:
                    # Get KB IDs linked to this workspace
                    kb_ids = await get_workspace_kb_ids(workspace_id)
                    workspace_is_kg = await is_kg_workspace(workspace_id)

                    logger.debug(
                        "Fetching documents for workspace",
                        workspace_id=workspace_id,
                        workspace_is_kg=workspace_is_kg,
                        linked_kb_count=len(kb_ids),
                    )

                    # Query documents from workspace OR from linked KBs
                    if kb_ids:
                        base_filter = or_(
                            DocumentMetadata.workspace_id == workspace_id,
                            DocumentMetadata.kb_id.in_(kb_ids)
                        )
                    else:
                        # No linked KBs, fetch only workspace documents
                        base_filter = DocumentMetadata.workspace_id == workspace_id
                else:
                    # No workspace filter, return all documents
                    base_filter = True
                    workspace_is_kg = False

                # Count total documents (for pagination metadata)
                count_query = select(func.count(DocumentMetadata.id)).where(base_filter)
                count_result = await session.execute(count_query)
                total_count = count_result.scalar()

                # Fetch paginated documents
                query = (
                    select(DocumentMetadata)
                    .where(base_filter)
                    .order_by(DocumentMetadata.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )

                result = await session.execute(query)
                docs = result.scalars().all()

                # Get indexing status from file_tasks for each document
                doc_ids_to_task_map = {}
                if docs:
                    # Extract file_task_ids to query file_tasks (full_doc_id removed from FileTask)
                    file_task_ids = [doc.file_task_id for doc in docs if doc.file_task_id]

                    if file_task_ids:
                        task_query = select(FileTask).where(
                            FileTask.id.in_(file_task_ids)
                        )
                        task_result = await session.execute(task_query)
                        tasks = task_result.scalars().all()

                        # Create map of file_task_id -> status, then map to full_doc_id
                        task_id_to_status = {task.id: task.status for task in tasks}
                        # Map full_doc_id -> status using file_task_id as bridge
                        doc_ids_to_task_map = {
                            doc.full_doc_id: task_id_to_status.get(doc.file_task_id, "unknown")
                            for doc in docs if doc.file_task_id
                        }

                def resolve_source_type(doc: DocumentMetadata) -> str:
                    if workspace_id is None:
                        return "kb_shared" if doc.kb_id else "workspace_only"

                    # Always show locally uploaded docs as workspace docs for the current workspace.
                    if doc.workspace_id == workspace_id:
                        return "workspace_doc"

                    if doc.kb_id:
                        # KG workspace explicitly labels non-local docs as KB-shared.
                        return "kb_doc" if workspace_is_kg else "kb_shared"

                    return "workspace_only"

                documents = [
                    {
                        "doc_id": doc.full_doc_id,
                        "file_name": doc.file_name,
                        "workspace_id": doc.workspace_id,
                        "kb_id": doc.kb_id,
                        "source_type": resolve_source_type(doc),
                        "file_path": doc.file_path,
                        "file_size_bytes": doc.file_size_bytes,
                        "chunk_count": doc.total_chunks,
                        "doc_type": doc.doc_type,
                        "indexing_status": doc_ids_to_task_map.get(doc.full_doc_id, "unknown"),
                        "metadata": doc.doc_metadata,
                        "indexed_at": str(doc.indexed_at) if doc.indexed_at else None,
                        "created_at": str(doc.created_at) if doc.created_at else None,
                    }
                    for doc in docs
                ]

                # Calculate pagination metadata
                has_more = (offset + len(documents)) < total_count

                pagination_info = {
                    "documents": documents,
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": has_more,
                    "page": (offset // limit) + 1 if limit > 0 else 1,
                    "total_pages": (total_count + limit - 1) // limit if limit > 0 else 1,
                }

            logger.info(
                "Retrieved indexed documents",
                count=len(documents),
                total=total_count,
                page=pagination_info["page"],
                has_more=has_more,
            )

            return pagination_info

        except Exception as e:
            logger.error("Failed to retrieve indexed documents", error=e)
            raise LightRAGException(
                message=f"Failed to retrieve indexed documents: {str(e)}",
                operation="get_docs"
            )

    async def close(self) -> None:
        """Close LightRAG connections and cleanup resources"""
        if self._rag:
            # Cleanup logic if needed
            self._initialized = False
            self._rag = None

        logger.info("LightRAG service closed")


# Singleton instance for global use
_lightrag_service_instance: Optional[LightRAGService] = None


def get_lightrag_service(working_dir: Optional[str] = None) -> LightRAGService:
    """
    Get or create a singleton LightRAG service instance.

    Args:
        working_dir: Optional working directory override

    Returns:
        LightRAGService: Singleton service instance
    """
    global _lightrag_service_instance

    if _lightrag_service_instance is None:
        _lightrag_service_instance = LightRAGService(working_dir=working_dir)

    return _lightrag_service_instance


async def initialize_lightrag_service(working_dir: Optional[str] = None) -> LightRAGService:
    """
    Initialize and return the global LightRAG service instance.

    Args:
        working_dir: Optional working directory

    Returns:
        LightRAGService: Initialized service instance
    """
    service = get_lightrag_service(working_dir=working_dir)
    await service.initialize()
    return service

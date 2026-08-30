"""LightRAG service for knowledge base operations"""
import os
import tempfile
from typing import Any, Dict, List, Optional, cast

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from shared.lightrag import (
    build_azure_openai_chat_completion_func,
    build_azure_openai_embedding_func,
)

try:
    from common_adapters.configurableAI.llm_router_config_store import (
        llm_router_config_store,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    llm_router_config_store = None

from .config import settings
from .exceptions import ConfigurationException, LightRAGException, ValidationException
from .logging import get_logger

logger = get_logger(__name__)


class LightRAGService:
    """
    Service class for LightRAG operations.

    Provides methods for initializing LightRAG, querying knowledge bases,
    and managing document indexing with shared storage configuration.
    """

    def __init__(self, workspace: str, working_dir: Optional[str] = None):
        """
        Initialize LightRAG service.

        Args:
            workspace: Workspace identifier for multi-tenancy in Neo4j/PostgreSQL (REQUIRED)
            working_dir: Working directory for LightRAG data
        """
        if not workspace or not workspace.strip():
            raise ValidationException(
                message="workspace parameter is required for LightRAGService"
            )

        serverless_mode = bool(getattr(settings.database, "SERVERLESS", True))
        configured_working_dir = working_dir or settings.lightrag.LIGHTRAG_WORKING_DIR
        if serverless_mode and not working_dir:
            configured_working_dir = os.path.join(tempfile.gettempdir(), "lightrag_data")

        self.working_dir = configured_working_dir
        self.workspace = workspace.strip()
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
                self._runtime_agent_id or 1,
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
                        api_key=str(azure_api_key or ""),
                        api_base=str(azure_api_base or ""),
                        api_version=str(azure_api_version or ""),
                        deployment=str(azure_deployment_name or ""),
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
            api_key=str(azure_api_key or ""),
            api_base=str(azure_api_base or ""),
            api_version=str(azure_api_version or ""),
            deployment=str(azure_deployment_name or ""),
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
                    api_key=str(azure_api_key or ""),
                    api_base=str(azure_api_base or ""),
                    api_version=str(azure_api_version or ""),
                    deployment=str(azure_deployment or ""),
                    dimensions=settings.lightrag.EMBEDDING_DIM,
                ),
            )

        raise ConfigurationException(
            message="Azure OpenAI Embedding configuration is incomplete",
            config_key="AZURE_OPENAI_EMBEDDING_MODEL",
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

            os.environ["NEO4J_URI"] = str(neo4j_uri)
            os.environ["NEO4J_USERNAME"] = str(neo4j_user)
            os.environ["NEO4J_PASSWORD"] = str(neo4j_password)

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
            if self._rag is not None:
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

            if self._rag is None:
                raise LightRAGException(message="LightRAG instance is not initialized", operation="query")

            # Execute query - LightRAG returns string answer or dict with answer+context
            result = await self._rag.aquery(
                query,
                param=QueryParam(mode=cast(Any, mode), only_need_context=only_need_context, **kwargs)
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

            if self._rag is None:
                raise LightRAGException(message="LightRAG instance is not initialized", operation="query_data")

            result = await self._rag.aquery_data(
                query,
                param=QueryParam(mode=cast(Any, mode), **kwargs)
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

            if self._rag is None:
                raise LightRAGException(message="LightRAG instance is not initialized", operation="query_llm")

            result = await self._rag.aquery_llm(
                query,
                param=QueryParam(mode=cast(Any, mode), **kwargs)
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

            if self._rag is None:
                raise LightRAGException(message="LightRAG instance is not initialized", operation="insert")

            if file_path:
                normalized_file_path = str(file_path).replace("\\", "/")
                await self._rag.ainsert(input=text, file_paths=[normalized_file_path])
            else:
                try:
                    insert_kwargs: Dict[str, Any] = {"metadata": metadata} if metadata else {}
                    await self._rag.ainsert(text, **insert_kwargs)
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

            if self._rag is None:
                raise LightRAGException(message="LightRAG instance is not initialized", operation="delete")

            deleted = False
            rag_inst = self._rag
            # Primary path for current LightRAG versions.
            if hasattr(rag_inst, "adelete_by_doc_id"):
                try:
                    await getattr(rag_inst, "adelete_by_doc_id")(doc_id)
                    deleted = True
                except TypeError:
                    await getattr(rag_inst, "adelete_by_doc_id")(doc_id=doc_id)
                    deleted = True
            # Compatibility path for older variants that expose `adelete_by_doc_ids`.
            elif hasattr(rag_inst, "adelete_by_doc_ids"):
                try:
                    await getattr(rag_inst, "adelete_by_doc_ids")([doc_id])
                    deleted = True
                except TypeError:
                    await getattr(rag_inst, "adelete_by_doc_ids")(doc_ids=[doc_id])
                    deleted = True
            else:
                raise LightRAGException(
                    message="LightRAG delete API not available on current runtime",
                    operation="delete",
                )

            # Best-effort cache cleanup after successful delete invocation.
            if deleted and hasattr(self._rag, "aclear_cache"):
                try:
                    await self._rag.aclear_cache()
                except Exception as cache_error:
                    logger.warning(
                        "LightRAG cache clear failed after delete",
                        doc_id=doc_id,
                        error=str(cache_error),
                    )

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

    async def get_knowledge_graph(
        self,
        node_label: Optional[str] = None,
        max_depth: int = 2,
        max_nodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get the knowledge graph from LightRAG or direct Neo4j query.

        Args:
            node_label: Optional label/entity name to center the subgraph query around. If None or "*", gets global graph.
            max_depth: Depth of graph traversal (default: 2)
            max_nodes: Maximum number of nodes to return

        Returns:
            Dict containing nodes and edges of the knowledge graph

        Raises:
            LightRAGException: If retrieval fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info("Retrieving knowledge graph", node_label=node_label, workspace=self.workspace)

            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []

            # Option 1: Primary path for Neo4j database - query Neo4j directly using workspace label for true element_id and full properties
            try:
                from src.core.neo4j_driver import get_neo4j_driver
                neo4j_driver = get_neo4j_driver()
                if not neo4j_driver._driver:
                    await neo4j_driver.connect()

                workspace_label = (self.workspace or "base").strip().replace("`", "``")
                limit_val = max_nodes or 1000
                target_label = (node_label or "*").strip()

                if target_label != "*":
                    cypher_query = f"""
                    MATCH (n:`{workspace_label}`)
                    WHERE n.entity_id = $node_label OR n.entity_name = $node_label
                    OPTIONAL MATCH (n)-[r]-(m:`{workspace_label}`)
                    RETURN elementId(n) as node_eid, labels(n) as node_labels, properties(n) as node_props,
                           elementId(r) as rel_eid, type(r) as rel_type, properties(r) as rel_props,
                           elementId(m) as target_eid, labels(m) as target_labels, properties(m) as target_props
                    LIMIT $limit
                    """
                    results = await neo4j_driver.execute_query(cypher_query, {"node_label": target_label, "limit": limit_val})
                else:
                    cypher_query = f"""
                    MATCH (n:`{workspace_label}`)
                    OPTIONAL MATCH (n)-[r]->(m:`{workspace_label}`)
                    RETURN elementId(n) as node_eid, labels(n) as node_labels, properties(n) as node_props,
                           elementId(r) as rel_eid, type(r) as rel_type, properties(r) as rel_props,
                           elementId(m) as target_eid, labels(m) as target_labels, properties(m) as target_props
                    LIMIT $limit
                    """
                    results = await neo4j_driver.execute_query(cypher_query, {"limit": limit_val})

                seen_nodes: set[str] = set()
                seen_edges: set[str] = set()

                for rec in results:
                    n_eid = rec.get("node_eid")
                    if n_eid and str(n_eid) not in seen_nodes:
                        seen_nodes.add(str(n_eid))
                        props = rec.get("node_props") or {}
                        n_labels = rec.get("node_labels") or []
                        nodes.append({
                            "id": str(n_eid),
                            "labels": n_labels,
                            "entity_name": props.get("entity_id") or props.get("entity_name") or (n_labels[0] if n_labels else str(n_eid)),
                            "entity_type": props.get("entity_type") or "UNKNOWN",
                            "created_at": props.get("created_at") or props.get("create_time"),
                            "description": props.get("description"),
                            "file_path": props.get("file_path") or props.get("source_file"),
                            "source_id": props.get("source_id") or props.get("chunk_id"),
                            "properties": props,
                        })

                    m_eid = rec.get("target_eid")
                    if m_eid and str(m_eid) not in seen_nodes:
                        seen_nodes.add(str(m_eid))
                        m_props = rec.get("target_props") or {}
                        m_labels = rec.get("target_labels") or []
                        nodes.append({
                            "id": str(m_eid),
                            "labels": m_labels,
                            "entity_name": m_props.get("entity_id") or m_props.get("entity_name") or (m_labels[0] if m_labels else str(m_eid)),
                            "entity_type": m_props.get("entity_type") or "UNKNOWN",
                            "created_at": m_props.get("created_at") or m_props.get("create_time"),
                            "description": m_props.get("description"),
                            "file_path": m_props.get("file_path") or m_props.get("source_file"),
                            "source_id": m_props.get("source_id") or m_props.get("chunk_id"),
                            "properties": m_props,
                        })

                    r_eid = rec.get("rel_eid")
                    if r_eid and str(r_eid) not in seen_edges and n_eid and m_eid:
                        seen_edges.add(str(r_eid))
                        r_props = rec.get("rel_props") or {}
                        n_props = rec.get("node_props") or {}
                        m_props = rec.get("target_props") or {}
                        n_name = n_props.get("entity_id") or n_props.get("entity_name") or str(n_eid)
                        m_name = m_props.get("entity_id") or m_props.get("entity_name") or str(m_eid)
                        edges.append({
                            "id": str(r_eid),
                            "source": str(n_name),
                            "target": str(m_name),
                            "relation": r_props.get("relation") or rec.get("rel_type") or "related_to",
                            "type": rec.get("rel_type") or r_props.get("relation") or "related_to",
                            "created_at": r_props.get("created_at") or r_props.get("create_time"),
                            "description": r_props.get("description"),
                            "file_path": r_props.get("file_path") or r_props.get("source_file"),
                            "keywords": r_props.get("keywords"),
                            "source_id": r_props.get("source_id") or r_props.get("chunk_id"),
                            "weight": r_props.get("weight"),
                            "properties": r_props,
                        })

                if nodes or edges:
                    logger.info("Retrieved knowledge graph via Neo4j driver", node_count=len(nodes), edge_count=len(edges))
                    return {
                        "nodes": nodes,
                        "edges": edges,
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                    }
            except Exception as neo_err:
                logger.warning("Direct Neo4j query failed or unavailable, falling back to LightRAG graph", error=str(neo_err))

            # Option 2: Fallback path - call LightRAG in-memory graph if direct Neo4j query yielded nothing
            if self._rag is not None:
                try:
                    target_label = (node_label or "*").strip()
                    kg_kwargs: Dict[str, Any] = {"node_label": target_label, "max_depth": max_depth}
                    if max_nodes is not None:
                        kg_kwargs["max_nodes"] = max_nodes
                    kg = await self._rag.get_knowledge_graph(**kg_kwargs)
                    raw_nodes = getattr(kg, "nodes", []) or []
                    raw_edges = getattr(kg, "edges", []) or []

                    for n in raw_nodes:
                        n_id = str(getattr(n, "id", "") or "")
                        n_props = getattr(n, "properties", {}) or {}
                        n_labels = getattr(n, "labels", []) or []
                        nodes.append({
                            "id": n_id,
                            "labels": n_labels,
                            "entity_name": n_props.get("entity_id") or (n_labels[0] if n_labels else n_id),
                            "entity_type": n_props.get("entity_type") or "UNKNOWN",
                            "created_at": n_props.get("created_at") or n_props.get("create_time"),
                            "description": n_props.get("description"),
                            "file_path": n_props.get("file_path"),
                            "source_id": n_props.get("source_id"),
                            "properties": n_props,
                        })

                    for e in raw_edges:
                        e_id = str(getattr(e, "id", "") or "")
                        e_props = getattr(e, "properties", {}) or {}
                        edges.append({
                            "id": e_id,
                            "source": str(getattr(e, "source", "") or ""),
                            "target": str(getattr(e, "target", "") or ""),
                            "relation": e_props.get("relation") or str(getattr(e, "type", "") or "related_to"),
                            "type": str(getattr(e, "type", "") or "related_to"),
                            "created_at": e_props.get("created_at") or e_props.get("create_time"),
                            "description": e_props.get("description"),
                            "file_path": e_props.get("file_path"),
                            "keywords": e_props.get("keywords"),
                            "source_id": e_props.get("source_id"),
                            "weight": e_props.get("weight"),
                            "properties": e_props,
                        })

                    logger.info("Retrieved knowledge graph via LightRAG fallback", node_count=len(nodes), edge_count=len(edges))
                    return {
                        "nodes": nodes,
                        "edges": edges,
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                        "is_truncated": getattr(kg, "is_truncated", False),
                    }
                except Exception as rag_err:
                    logger.warning("LightRAG fallback get_knowledge_graph failed", error=str(rag_err))

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

    async def close(self) -> None:
        """Close LightRAG connections and cleanup resources"""
        if self._rag:
            # Cleanup logic if needed
            self._initialized = False
            self._rag = None

        logger.info("LightRAG service closed")


# Singleton instance for global use
_lightrag_service_instance: Optional[LightRAGService] = None


def get_lightrag_service(workspace: str, working_dir: Optional[str] = None) -> LightRAGService:
    """
    Get or create a LightRAG service instance for a required workspace label.

    Args:
        workspace: Workspace identifier for multi-tenancy in Neo4j/PostgreSQL (REQUIRED)
        working_dir: Optional working directory override

    Returns:
        LightRAGService: Service instance
    """
    if not workspace or not workspace.strip():
        raise ValidationException(
            message="workspace parameter is required for get_lightrag_service"
        )
    return LightRAGService(workspace=workspace.strip(), working_dir=working_dir)

"""
Per-mode message handlers.

Each handler implements the same `handle(...)` contract (Liskov substitution)
so the orchestrator can route to either without knowing the concrete type.
Access/permission checks are NOT repeated here — the orchestrator validates
once up front and passes the resulting `AccessContext` in.

LLM provider / intent detector / context pipeline are resolved per-call from
`(workspace_id, agent_id)` rather than injected once at construction, because
model routing in this platform is workspace/agent-specific (see
`llm_provider.get_llm_provider`); the underlying manager is still cached
there, so this costs nothing beyond a dict lookup on the common path.
"""
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.models.chat_models import ChatRequest
from src.services.intent_detection.models import Intent
from src.services.query_rag_executor import execute_query_rag
from src.services.rag_service import get_rag_service

from .context_middleware import build_default_context_pipeline
from .file_context_extractor import FileContextExtractor
from .intent_service import get_chat_intent_detector
from .llm_provider import get_llm_provider
from .models import AccessContext, HandlerResult

logger = get_logger(__name__)

# Greeting variations for more natural interaction
GREETING_VARIATIONS = [
    "Hello! I'm your knowledge base assistant. Ask me anything about your workspace's documents.",
    "Hi there! I can help you search and explore your knowledge base. What would you like to know?",
    "Hey! Ready to help you find information in your workspace. What can I assist you with?",
    "Greetings! I'm here to answer questions about your documents and knowledge base.",
    "Hello! I'm your AI assistant for this workspace. Feel free to ask me anything!",
]

# Help variations with different levels of detail
HELP_VARIATIONS = [
    (
        "I can help you search your knowledge base. In SEARCH mode, ask a question or attach a "
        "document for me to use as context. In UPDATE mode (curator access required), upload files "
        "to index them into the knowledge base."
    ),
    (
        "Here's what I can do:\n"
        "• Search your knowledge base for answers\n"
        "• Answer questions based on attached documents\n"
        "• Index new files (UPDATE mode, requires curator access)\n"
        "• Index URLs for web content (UPDATE mode)\n\n"
        "Just ask me a question or tell me what you need!"
    ),
    (
        "I'm here to help you work with your knowledge base. You can:\n"
        "- Ask questions and I'll search for relevant information\n"
        "- Attach files and ask questions about them directly\n"
        "- Upload files to index them (if you have curator access)\n"
        "- Index web pages by URL (UPDATE mode)\n\n"
        "What would you like to do?"
    ),
    (
        "Need assistance? I can:\n"
        "✓ Search through your workspace documents\n"
        "✓ Answer questions using your knowledge base\n"
        "✓ Analyze attached files without indexing them\n"
        "✓ Index new content (with proper permissions)\n\n"
        "How can I help you today?"
    ),
]

# Confidence thresholds
CONFIDENCE_THRESHOLD_LOW = 0.6  # Below this, ask for clarification
CONFIDENCE_THRESHOLD_MEDIUM = 0.8  # Log for monitoring

# Low confidence clarification messages
LOW_CONFIDENCE_CLARIFICATIONS = [
    "I'm not entirely sure what you're asking for. Could you rephrase or provide more details?",
    "I want to make sure I understand correctly. Are you trying to search the knowledge base, or do you need help with something else?",
    "Could you clarify what you're looking for? I can search your documents or help with general questions.",
]


class ModeHandler(ABC):
    """Contract every mode handler fulfills."""

    @abstractmethod
    async def handle(
        self,
        payload: ChatRequest,
        access: AccessContext,
        history: List[Dict[str, Any]],
    ) -> HandlerResult:
        raise NotImplementedError


class SearchModeHandler(ModeHandler):
    """
    Handles SEARCH mode (the default of the two supported modes).

    - greeting/help resolve to canned responses (no LLM/RAG round-trip).
    - a message with attached files answers from that file's content as
      context, WITHOUT indexing it (requirement: context-only for search).
    - anything else is treated as search_kb and answered from the KB via the
      SAME `execute_query_rag` path the `/query-rag` endpoint uses, mirroring
      KnowledgeCurator's SEARCH-mode collapse of restricted intents to search_kb.
    """

    ANSWER_FROM_CONTEXT_SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions using ONLY the provided "
        "document context and conversation history. If the answer isn't contained "
        "in the context, say you don't have enough information."
    )

    def __init__(self, file_extractor: FileContextExtractor) -> None:
        self._file_extractor = file_extractor

    async def handle(
        self,
        payload: ChatRequest,
        access: AccessContext,
        history: List[Dict[str, Any]],
    ) -> HandlerResult:
        if payload.files:
            return await self._answer_with_file_context(payload, access, history)

        intent_detector = get_chat_intent_detector(access.workspace_id, payload.agent_id)
        intent_result = await intent_detector.detect(
            payload.user_message,
            context={"workspace_id": access.workspace_id},
        )

        # Log intent detection results with confidence monitoring
        logger.info(
            "Intent detected",
            intent=intent_result.intent.value,
            confidence=intent_result.confidence,
            method=intent_result.method,
            metadata=intent_result.metadata,
        )

        # Monitor low-confidence detections for quality tracking
        if intent_result.confidence < CONFIDENCE_THRESHOLD_MEDIUM:
            logger.warning(
                "Low confidence intent detection",
                intent=intent_result.intent.value,
                confidence=intent_result.confidence,
                message_preview=payload.user_message[:100],
                workspace_id=access.workspace_id,
            )

        # Ask for clarification on very low confidence
        if intent_result.confidence < CONFIDENCE_THRESHOLD_LOW:
            clarification = random.choice(LOW_CONFIDENCE_CLARIFICATIONS)
            logger.info("Requesting clarification due to low confidence")
            return HandlerResult(
                response=clarification,
                intent="clarification_needed",
                metadata={"original_intent": intent_result.intent.value, "confidence": intent_result.confidence},
            )

        # Handle specific intents
        if intent_result.intent == Intent.GREETING:
            greeting = random.choice(GREETING_VARIATIONS)
            return HandlerResult(response=greeting, intent=Intent.GREETING.value)

        if intent_result.intent == Intent.HELP:
            help_msg = random.choice(HELP_VARIATIONS)
            return HandlerResult(response=help_msg, intent=Intent.HELP.value)

        # SEARCH mode never executes curation intents even if detected;
        # everything else collapses to a knowledge-base search.
        return await self._search_kb(payload, access, history)

    async def _search_kb(
        self,
        payload: ChatRequest,
        access: AccessContext,
        history: List[Dict[str, Any]],
    ) -> HandlerResult:
        """
        Answer from the knowledge base via the shared `execute_query_rag` path.

        Chat deliberately does NOT call `RAGQueryService` directly: going
        through the executor means chat gets the Redis result cache, the signed
        `file_id` minting and the cached evidence/source mappings for free, and
        keeps chat answers identical to `/query-rag` answers for the same
        question. Access was already validated by the orchestrator, so the
        pre-resolved `AccessContext` is passed straight through instead of
        re-reading workspace metadata.
        """
        response_dict, cache_hit = await execute_query_rag(
            query=payload.user_message,
            workspace_id=access.workspace_id,
            role_id=access.role_id,
            domain=access.domain,
            kb_name=access.kb_name,
            mode="hybrid",
            history=history,
            additional_kbs=access.additional_kbs,
            agent_id=payload.agent_id,
            is_kg=access.is_kg,
            container_name=access.container_name,
        )

        response_text = (
            str(response_dict.get("final_answer") or "").strip()
            or "I don't have enough information to answer that question."
        )
        sources = self._build_sources(response_dict)

        logger.info(
            "Chat search completed",
            cache_hit=cache_hit,
            source_count=len(sources),
            answer_length=len(response_text),
        )
        return HandlerResult(response=response_text, sources=sources, intent=Intent.SEARCH_KB.value)

    @staticmethod
    def _build_sources(response_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Reduce the executor's `source` entries to what chat persists and returns.

        Each source already carries a signed `file_id` (same contract as
        query_rag), which the client exchanges for a short-lived URL via
        `GET /api/v2/files/{file_id}/download`. `container_name`/`blob_path`/
        `provider` are internal-only fields excluded from the serialized model,
        so nothing storage-specific leaks into the chat history documents.
        """
        raw_sources = response_dict.get("source")
        if not isinstance(raw_sources, list):
            return []

        return [
            {
                "file_id": str(source.get("file_id") or ""),
                "file_name": str(source.get("file_name") or ""),
                "citation": source.get("citation"),
            }
            for source in raw_sources
            if isinstance(source, dict) and source.get("file_id")
        ]

    async def _answer_with_file_context(
        self,
        payload: ChatRequest,
        access: AccessContext,
        history: List[Dict[str, Any]],
    ) -> HandlerResult:
        llm_provider = get_llm_provider(workspace_id=access.workspace_id, agent_id=payload.agent_id)

        # Extract file names and contents from payload files
        file_names: List[str] = []
        file_contents: List[str] = []
        if payload.files:
            file_names = [f.file_name for f in payload.files]
            file_contents = [f.file_content for f in payload.files]

        raw_context = await self._file_extractor.extract(file_names, file_contents)
        if not raw_context.strip():
            return HandlerResult(
                response="I couldn't extract any readable text from the attached file(s).",
                intent=Intent.UPLOAD_FILE.value,
            )

        context_pipeline = build_default_context_pipeline(llm_provider)
        messages = [*history, {"role": "user", "content": raw_context}]
        compacted = await context_pipeline.process(messages)
        context_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in compacted)

        prompt = f"Context:\n{context_text}\n\nUser question: {payload.user_message}"
        answer = await llm_provider.complete(
            prompt=prompt,
            system_prompt=self.ANSWER_FROM_CONTEXT_SYSTEM_PROMPT,
            max_tokens=1000,
            temperature=0.2,
        )
        return HandlerResult(response=answer.strip(), intent=Intent.UPLOAD_FILE.value)


class UpdateModeHandler(ModeHandler):
    """
    Handles UPDATE mode: indexes uploaded files into the knowledge base.

    Requires `can_curate_kb=True` on the already-resolved `AccessContext` —
    no additional DB round-trip is made here.
    """

    def __init__(self) -> None:
        self._rag_service = get_rag_service()

    async def handle(
        self,
        payload: ChatRequest,
        access: AccessContext,
        history: List[Dict[str, Any]],
    ) -> HandlerResult:
        if not access.can_curate_kb:
            raise AuthorizationException(
                message="You have search-only access in this workspace. "
                "Indexing or editing the knowledge base is not allowed for your account."
            )

        if not (payload.files):
            raise ValidationException(message="UPDATE mode requires file_names and file_contents to index")

        upload_result = await self._rag_service.upload_and_index_tool(
            # file_names=payload.file_names,
            # file_contents=payload.file_contents,
            files=payload.files,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
            domain=access.domain,
            kb_name=access.kb_name,
            upload_path=access.upload_path,
            kb_ids=access.all_kb_ids,
            container_name=access.container_name,
            is_kg=access.is_kg,
        )

        return HandlerResult(
            response=upload_result.get("response", "Files uploaded successfully"),
            task_ids=upload_result.get("task_ids", []),
            intent=Intent.UPLOAD_FILE.value,
        )

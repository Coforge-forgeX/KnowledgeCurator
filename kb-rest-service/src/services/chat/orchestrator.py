"""
Top-level entry point for message_gpt.

Responsibilities (and nothing else — each is delegated to its own class):
1. Validate user + workspace access exactly once (`ChatAccessValidator`).
2. Load recent history for context (`MongoDBService` — the SAME store used
   by start_conversation/get_conversation_history/load_conversation/
   rename_conversation/delete_conversation, so all five endpoints stay
   interconnected over one chatbot database).
3. Route to the mode-appropriate handler, running it as a cancellable task
   (`common_adapters.cancel_convesation`) so a client can abort an in-flight
   request via `cancel_chat_message`.
4. Persist the turn and return a uniform result.

Downstream handlers receive the already-validated `AccessContext` and never
re-query the database for permissions, which is what keeps this endpoint
from paying the multi-step validation cost that `query_rag` /
`upload_and_index` each pay independently today.
"""
import asyncio
from typing import Optional

from common_adapters.cancel_convesation import (
    CancelledError,
    clear_cancellation,
    register_task,
    unregister_task,
)

from src.core.config import settings
from src.core.exceptions import ValidationException
from src.core.logging import get_logger
from src.core.redis import (
    append_cached_message,
    cache_conversation_history,
    get_cached_conversation_history,
)
from src.models.chat_models import ChatRequest, ChatResponse
from src.services.mongodb_service import MongoDBService, get_mongodb_service

from .access_validator import ChatAccessValidator, get_chat_access_validator
from .file_context_extractor import FileContextExtractor, get_file_context_extractor
from .handlers import ModeHandler, SearchModeHandler, UpdateModeHandler
from .models import HandlerResult

logger = get_logger(__name__)

_UPDATE_MODES = {"UPDATE"}


class ChatOrchestrator:
    """Coordinates access validation, history, mode routing, and persistence."""

    def __init__(
        self,
        access_validator: ChatAccessValidator,
        mongo_service: MongoDBService,
        file_extractor: FileContextExtractor,
    ) -> None:
        self._access_validator = access_validator
        self._mongo = mongo_service
        self._search_handler: ModeHandler = SearchModeHandler(file_extractor=file_extractor)
        self._update_handler: ModeHandler = UpdateModeHandler()

    async def handle_message(self, payload: ChatRequest, user_id: int) -> ChatResponse:
        """`user_id` is the authenticated caller (from the Bearer token); it is
        not part of the payload so there is no client-supplied identity to
        reconcile."""
        await self._mongo.initialize()

        access = await self._access_validator.validate(
            user_id=user_id,
            workspace_id=payload.workspace_id,
        )

        # message_gpt is frequently called with a client-generated session_id and
        # no prior start_conversation. Create the session document up front so the
        # turn is visible to list_sessions / load_conversation and so the title
        # update below has a document to write to.
        await self._mongo.ensure_session(
            session_id=payload.session_id,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
        )

        # Read history BEFORE persisting the current turn: the message being
        # handled is passed to the LLM as the query, so including it in the
        # history too would duplicate it in the prompt.
        #
        # Try Redis cache first, fall back to MongoDB on cache miss
        cached_messages = get_cached_conversation_history(
            session_id=payload.session_id,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
        )

        if cached_messages is not None:
            logger.debug(
                "Using cached conversation history",
                session_id=payload.session_id,
                message_count=len(cached_messages),
            )
            messages = cached_messages
        else:
            logger.debug(
                "Cache miss - fetching conversation history from MongoDB",
                session_id=payload.session_id,
            )
            messages = await self._mongo.get_conversation_history(
                session_id=payload.session_id,
                workspace_id=access.workspace_id,
                user_id=access.user_id,
                limit=settings.CHAT_HISTORY_TURNS_FOR_CONTEXT,
            )
            # Cache the result for subsequent messages
            cache_conversation_history(
                session_id=payload.session_id,
                workspace_id=access.workspace_id,
                user_id=access.user_id,
                history=messages,
                            )

        history = self._as_llm_history(messages)

        # Persist user message to MongoDB
        await self._mongo.append_message(
            session_id=payload.session_id,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
            role="user",
            content=payload.user_message,
        )

        # Update cache with the new user message
        append_cached_message(
            session_id=payload.session_id,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
            message={"role": "user", "content": payload.user_message},
                    )

        await self._track_title(payload, access.workspace_id, access.user_id)

        handler = self._update_handler if payload.mode.upper() in _UPDATE_MODES else self._search_handler

        try:
            result = await self._run_cancellable(handler, payload, access, history)
        except CancelledError:
            response_text = "Your request was cancelled."
            await self._mongo.append_message(
                session_id=payload.session_id,
                workspace_id=access.workspace_id,
                user_id=access.user_id,
                role="assistant",
                content=response_text,
            )
            # Update cache with cancelled response
            append_cached_message(
                session_id=payload.session_id,
                workspace_id=access.workspace_id,
                user_id=access.user_id,
                message={"role": "assistant", "content": response_text},
                            )
            return ChatResponse(response=response_text, session_id=payload.session_id)
        except ValidationException:
            raise
        except Exception as e:
            logger.error("message_gpt handler failed", error=e, session_id=payload.session_id)
            response_text = f"Sorry, something went wrong while processing your request: {str(e)}"
            await self._mongo.append_message(
                session_id=payload.session_id,
                workspace_id=access.workspace_id,
                user_id=access.user_id,
                role="assistant",
                content=response_text,
            )
            # Update cache with error response
            append_cached_message(
                session_id=payload.session_id,
                workspace_id=access.workspace_id,
                user_id=access.user_id,
                message={"role": "assistant", "content": response_text},
                            )
            return ChatResponse(response=response_text, session_id=payload.session_id)

        # Persist assistant message to MongoDB
        await self._mongo.append_message(
            session_id=payload.session_id,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
            role="assistant",
            content=result.response,
            sources=result.sources,
            task_ids=result.task_ids,
        )

        # Update cache with the assistant response
        append_cached_message(
            session_id=payload.session_id,
            workspace_id=access.workspace_id,
            user_id=access.user_id,
            message={
                "role": "assistant",
                "content": result.response,
                "sources": result.sources,
                "task_ids": result.task_ids,
            },
                    )

        return ChatResponse(
            response=result.response,
            sources=result.sources,
            task_ids=result.task_ids,
            session_id=payload.session_id,
        )

    @staticmethod
    def _as_llm_history(messages) -> list:
        """
        Reduce stored messages to the `{"role", "content"}` shape LightRAG's
        `conversation_history` (and the context-compaction pipeline) expect.
        Stored documents also carry timestamps/sources/metadata, which are not
        JSON-serialisable prompt material.
        """
        history = []
        for message in messages or []:
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            role = str(message.get("role") or "user")
            history.append({"role": role, "content": content})
        return history

    async def _run_cancellable(self, handler: ModeHandler, payload, access, history) -> HandlerResult:
        """
        Runs the handler as an asyncio Task registered for cooperative
        cancellation, so `cancel_chat_message` can interrupt an in-flight
        request (mirrors KnowledgeCurator's register_task/is_cancelled flow).
        """
        task = asyncio.ensure_future(handler.handle(payload, access, history))
        register_task(conversation_id=payload.session_id, task=task)
        try:
            return await task
        except asyncio.CancelledError:
            raise CancelledError("Request was cancelled")
        finally:
            unregister_task(conversation_id=payload.session_id)
            clear_cancellation(conversation_id=payload.session_id)

    async def _track_title(self, payload: ChatRequest, workspace_id: int, user_id: int) -> None:
        """
        Keep the session title in sync with the latest user message.

        This is a no-op once the user has manually renamed the session
        (`update_session_title`'s `is_manual=False` guard on
        `title_set_by_user`), so a rename always wins over this auto-tracking.
        """
        title = payload.user_message.strip()
        if len(title) > 50:
            title = title[:50] + "..."

        await self._mongo.update_session_title(
            session_id=payload.session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            title=title,
            is_manual=False,
        )


_orchestrator_instance: Optional[ChatOrchestrator] = None


def get_chat_orchestrator() -> ChatOrchestrator:
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ChatOrchestrator(
            access_validator=get_chat_access_validator(),
            mongo_service=get_mongodb_service(),
            file_extractor=get_file_context_extractor(),
        )
    return _orchestrator_instance

"""
Pluggable context middleware.

`ContextMiddleware` is a minimal interface (`process(messages) -> messages`)
operating on chat-message lists (`[{"role": ..., "content": ...}, ...]`).
`ContextMiddlewarePipeline` runs an ordered list of them. New behaviors
(PII redaction, translation, ...) plug in by adding another
`ContextMiddleware` implementation to the pipeline — no caller changes.

`SummarizationMiddleware` wraps `common_adapters.context_compaction.ContextCompactor`
(the platform's shared compaction utility) rather than a bespoke
implementation: once the message list exceeds the configured token budget,
older turns are summarized while the most recent ones are kept verbatim.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from common_adapters.context_compaction import ContextCompactor

from src.core.config import settings
from src.core.logging import get_logger
from src.services.intent_detection.protocols import LLMProvider

logger = get_logger(__name__)

Message = Dict[str, Any]


class ContextMiddleware(ABC):
    """A single context post-processing step."""

    @abstractmethod
    async def process(self, messages: List[Message]) -> List[Message]:
        """Transform `messages`, returning the (possibly unchanged) result."""
        raise NotImplementedError


class SummarizationMiddleware(ContextMiddleware):
    """Summarizes older messages once the conversation exceeds the token budget."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider],
        max_tokens: Optional[int] = None,
        keep_last: Optional[int] = None,
    ) -> None:
        self._llm = llm_provider
        self._compactor = ContextCompactor(
            max_tokens=max_tokens or settings.CHAT_CONTEXT_TOKEN_THRESHOLD,
            strategy="summarize",
            keep_last=keep_last or settings.CHAT_HISTORY_TURNS_FOR_CONTEXT,
            llm=llm_provider,
        )

    async def process(self, messages: List[Message]) -> List[Message]:
        if not messages:
            return messages
        if self._llm is None:
            logger.debug("No LLM provider configured; skipping summarization middleware")
            return messages
        try:
            return await self._compactor.compact(messages)
        except Exception as e:
            logger.error("Context summarization failed; passing messages through unchanged", error=e)
            return messages


class ContextMiddlewarePipeline:
    """Runs a list of `ContextMiddleware` in order."""

    def __init__(self, middlewares: Optional[List[ContextMiddleware]] = None) -> None:
        self._middlewares = list(middlewares or [])

    def add(self, middleware: ContextMiddleware) -> "ContextMiddlewarePipeline":
        self._middlewares.append(middleware)
        return self

    async def process(self, messages: List[Message]) -> List[Message]:
        for middleware in self._middlewares:
            messages = await middleware.process(messages)
        return messages


def build_default_context_pipeline(llm_provider: Optional[LLMProvider]) -> ContextMiddlewarePipeline:
    """Default pipeline: summarization only. Plug in more middlewares here as needed."""
    return ContextMiddlewarePipeline([SummarizationMiddleware(llm_provider=llm_provider)])

"""
LLM-based intent detector for complex cases.

Provider-agnostic implementation - works with any LLM provider.
"""
import json
import logging
from typing import Optional

from ..models import Intent, IntentResult
from ..protocols import LLMProvider
from ..utils.cache import cached_detection
from .base import BaseIntentDetector

logger = logging.getLogger(__name__)


class LLMIntentDetector(BaseIntentDetector):
    """
    LLM-powered intent detector for complex or ambiguous queries.

    Advantages:
    - Handles nuanced and complex intents
    - Better for conversational inputs
    - Can understand context and user intent

    Disadvantages:
    - Higher latency (~100-500ms)
    - Higher cost per request
    - Slower cold starts
    - Requires external API

    Use cases:
    - Fallback for rule-based detector
    - Complex multi-intent queries
    - Conversational interfaces
    """

    SYSTEM_PROMPT = """You are an intent classifier for a knowledge base chatbot.

Classify the user message into one of these intents:
- search_kb: Searching or querying the knowledge base
- upload_file: Uploading or indexing files/documents
- add_entity: Adding new entities to knowledge base
- delete_entity: Deleting entities from knowledge base
- update_entity: Updating or modifying entities
- delete_file: Deleting files from knowledge base
- index_url: Indexing content from a URL
- greeting: Greetings and pleasantries
- help: Requests for help or capabilities
- unknown: Cannot determine intent

Respond with ONLY the intent name, nothing else."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        enable_cache: bool = True,
        fallback_intent: Intent = Intent.SEARCH_KB,
    ):
        """
        Initialize LLM-based detector.

        Args:
            llm_provider: LLM provider implementation
            enable_cache: Whether to enable caching
            fallback_intent: Fallback intent
        """
        super().__init__(fallback_intent=fallback_intent)
        self.llm = llm_provider
        self.enable_cache = enable_cache

    def supports_caching(self) -> bool:
        """LLM detection benefits from caching due to cost"""
        return self.enable_cache

    @cached_detection(enabled=True, ttl_seconds=1800)  # 30 minute cache
    async def _detect_intent(
        self,
        message: str,
        context: dict,
    ) -> IntentResult:
        """
        Detect intent using LLM.

        Args:
            message: User message
            context: Context dictionary

        Returns:
            IntentResult from LLM classification
        """
        # Build prompt
        prompt = f'User message: "{message}"\nIntent:'

        try:
            # Call LLM with low temperature for deterministic results
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=50,
                temperature=0.0,  # Deterministic
            )

            # Parse response
            intent_str = response.strip().split()[0].lower()

            # Map to Intent enum
            try:
                detected_intent = Intent(intent_str)
            except ValueError:
                logger.warning(f"Unknown intent from LLM: {intent_str}")
                detected_intent = self.fallback_intent

            logger.debug(f"LLM detected intent: {detected_intent.value}")

            return IntentResult(
                intent=detected_intent,
                confidence=0.85,  # LLM confidence (could be improved with logprobs)
                method="llm",
                metadata={
                    "llm_response": response,
                    "message_length": len(message),
                },
            )

        except Exception as e:
            logger.error(f"LLM intent detection failed: {e}")
            return self._create_fallback_result()


class SimpleLLMProvider(LLMProvider):
    """
    Simple LLM provider adapter.

    This is a placeholder - integrate with your actual LLM service.
    Can be replaced with OpenAI, Anthropic, Azure OpenAI, or custom provider.
    """

    def __init__(self, llm_function):
        """
        Initialize with existing LLM function.

        Args:
            llm_function: Async function that takes (prompt, system_prompt) and returns response
        """
        self.llm_function = llm_function

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Call LLM function"""
        try:
            response = await self.llm_function(
                user_input=prompt,
                sys_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response
        except Exception as e:
            logger.error(f"LLM provider error: {e}")
            raise

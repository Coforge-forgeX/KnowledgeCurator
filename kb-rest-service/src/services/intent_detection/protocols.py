"""Abstract interfaces for intent detection (SOLID - Interface Segregation Principle)"""
from abc import ABC, abstractmethod
from typing import Optional

from .models import IntentResult


class IntentDetector(ABC):
    """
    Abstract base class for intent detectors.

    Follows SOLID principles:
    - Single Responsibility: Only handles intent detection
    - Open/Closed: Open for extension (new detectors), closed for modification
    - Liskov Substitution: All implementations are interchangeable
    - Interface Segregation: Minimal, focused interface
    - Dependency Inversion: Depends on abstraction, not concrete implementations
    """

    @abstractmethod
    async def detect(
        self,
        message: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent from user message.

        Args:
            message: User input text
            context: Optional context (e.g., workspace_id, conversation history)

        Returns:
            IntentResult with detected intent and metadata

        Raises:
            ValueError: If message is empty or invalid
        """
        pass

    def supports_caching(self) -> bool:
        """Whether this detector benefits from caching"""
        return False


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers (Strategy Pattern).

    Enables provider-agnostic implementation - works with OpenAI, Anthropic,
    Azure OpenAI, or any custom LLM provider.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """
        Get completion from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            LLM response text
        """
        pass

"""Base detector implementation with common functionality"""
import logging
from typing import Optional

from ..models import Intent, IntentResult
from ..protocols import IntentDetector

logger = logging.getLogger(__name__)


class BaseIntentDetector(IntentDetector):
    """
    Base implementation of IntentDetector with input validation.

    Follows Template Method pattern - subclasses implement _detect_intent().
    """

    def __init__(self, fallback_intent: Intent = Intent.SEARCH_KB):
        """
        Initialize base detector.

        Args:
            fallback_intent: Default intent when detection fails
        """
        self.fallback_intent = fallback_intent

    async def detect(
        self,
        message: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent with validation and error handling.

        Args:
            message: User input text
            context: Optional context

        Returns:
            IntentResult with detected intent

        Raises:
            ValueError: If message is empty or invalid
        """
        # Input validation (fail fast)
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        # Normalize input
        normalized_message = message.strip()
        context = context or {}

        try:
            # Delegate to subclass implementation
            result = await self._detect_intent(normalized_message, context)

            # Validate result
            if not isinstance(result, IntentResult):
                logger.error(f"Invalid result type: {type(result)}")
                return self._create_fallback_result()

            return result

        except Exception as e:
            logger.error(f"Intent detection failed: {e}", exc_info=True)
            return self._create_fallback_result()

    async def _detect_intent(
        self,
        message: str,
        context: dict,
    ) -> IntentResult:
        """
        Actual detection logic - implemented by subclasses.

        Args:
            message: Normalized user message
            context: Context dictionary

        Returns:
            IntentResult
        """
        raise NotImplementedError("Subclasses must implement _detect_intent()")

    def _create_fallback_result(self) -> IntentResult:
        """Create fallback result for error cases"""
        return IntentResult(
            intent=self.fallback_intent,
            confidence=0.5,
            method="fallback",
            metadata={"reason": "detection_failed"},
        )

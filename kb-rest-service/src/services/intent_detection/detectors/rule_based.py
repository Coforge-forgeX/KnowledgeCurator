"""
Rule-based intent detector using pattern matching.

Fast, deterministic, and serverless-friendly - no LLM calls required.
Ideal for 80-90% of common use cases.
"""
import logging
from typing import Optional

from ..models import Intent, IntentResult
from ..utils.cache import cached_detection
from ..utils.patterns import IntentPatternMatcher
from .base import BaseIntentDetector

logger = logging.getLogger(__name__)


class RuleBasedIntentDetector(BaseIntentDetector):
    """
    Rule-based intent detector using regex pattern matching.

    Advantages:
    - Zero latency (no API calls)
    - Zero cost (no LLM usage)
    - Deterministic and predictable
    - Perfect for serverless cold starts
    - Highly cacheable

    Use cases:
    - Production default for cost optimization
    - Fallback when LLM is unavailable
    - Low-latency requirements
    """

    def __init__(
        self,
        matcher: Optional[IntentPatternMatcher] = None,
        enable_cache: bool = True,
        fallback_intent: Intent = Intent.SEARCH_KB,
    ):
        """
        Initialize rule-based detector.

        Args:
            matcher: Pattern matcher (creates default if None)
            enable_cache: Whether to enable result caching
            fallback_intent: Fallback intent for unmatched messages
        """
        super().__init__(fallback_intent=fallback_intent)
        self.matcher = matcher or IntentPatternMatcher()
        self.enable_cache = enable_cache

    def supports_caching(self) -> bool:
        """Rule-based detection is deterministic and highly cacheable"""
        return self.enable_cache

    @cached_detection(enabled=True, ttl_seconds=600)  # 10 minute cache
    async def _detect_intent(
        self,
        message: str,
        context: dict,
    ) -> IntentResult:
        """
        Detect intent using pattern matching.

        Args:
            message: User message
            context: Context (mode, workspace_id, etc.)

        Returns:
            IntentResult with matched intent
        """
        # Special case: uploaded files should be indexed
        if context.get("file_names"):
            logger.debug("Files detected in context, intent=UPLOAD_FILE")
            return IntentResult(
                intent=Intent.UPLOAD_FILE,
                confidence=1.0,
                method="rule",
                metadata={"reason": "files_present", "file_count": len(context["file_names"])},
            )

        # Pattern matching
        matched_intent = self.matcher.match(message)

        if matched_intent:
            confidence = self.matcher.get_confidence(message, matched_intent)
            logger.debug(f"Pattern match: {matched_intent.value} (confidence={confidence:.2f})")

            return IntentResult(
                intent=matched_intent,
                confidence=confidence,
                method="rule",
                metadata={"message_length": len(message)},
            )

        # No pattern matched - use fallback
        logger.debug(f"No pattern matched, using fallback: {self.fallback_intent.value}")
        return IntentResult(
            intent=self.fallback_intent,
            confidence=0.5,
            method="rule",
            metadata={"reason": "no_pattern_match"},
        )

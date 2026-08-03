"""
Hybrid intent detector combining rule-based and LLM approaches.

Optimizes for both speed and accuracy.
"""
import logging
from typing import Optional

from ..models import Intent, IntentResult
from ..protocols import IntentDetector, LLMProvider
from .llm_based import LLMIntentDetector
from .rule_based import RuleBasedIntentDetector

logger = logging.getLogger(__name__)


class HybridIntentDetector(IntentDetector):
    """
    Hybrid detector using cascading strategy.

    Strategy:
    1. Try rule-based detection first (fast, free)
    2. If confidence is high (>= threshold), return immediately
    3. If confidence is low, fall back to LLM (slower, accurate)

    This gives best of both worlds:
    - Fast response for 80-90% of queries (rule-based)
    - Accurate fallback for complex queries (LLM)
    - Cost-optimized (only uses LLM when needed)

    Perfect for production serverless deployments.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        confidence_threshold: float = 0.8,
        enable_llm_fallback: bool = True,
    ):
        """
        Initialize hybrid detector.

        Args:
            llm_provider: LLM provider for fallback (optional)
            confidence_threshold: Minimum confidence to skip LLM (0.0-1.0)
            enable_llm_fallback: Whether to use LLM fallback
        """
        self.rule_detector = RuleBasedIntentDetector()
        self.llm_detector = (
            LLMIntentDetector(llm_provider) if llm_provider and enable_llm_fallback else None
        )
        self.confidence_threshold = confidence_threshold
        self.enable_llm_fallback = enable_llm_fallback and llm_provider is not None

    async def detect(
        self,
        message: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent using hybrid approach.

        Args:
            message: User message
            context: Optional context

        Returns:
            IntentResult from rule-based or LLM detection
        """
        # Step 1: Try rule-based detection
        rule_result = await self.rule_detector.detect(message, context)

        # Step 2: Check if we're confident enough
        if rule_result.confidence >= self.confidence_threshold:
            logger.debug(
                f"High confidence rule match ({rule_result.confidence:.2f}), "
                f"skipping LLM: {rule_result.intent.value}"
            )
            return rule_result

        # Step 3: For certain intents, always trust rule-based
        high_confidence_intents = {
            Intent.GREETING,
            Intent.HELP,
            Intent.UPLOAD_FILE,
        }
        if rule_result.intent in high_confidence_intents:
            logger.debug(
                f"High-confidence intent type {rule_result.intent.value}, "
                "skipping LLM"
            )
            return rule_result

        # Step 4: Fall back to LLM if enabled
        if self.enable_llm_fallback and self.llm_detector:
            logger.debug(
                f"Low confidence ({rule_result.confidence:.2f}), "
                "falling back to LLM"
            )

            try:
                llm_result = await self.llm_detector.detect(message, context)

                # Update metadata to indicate hybrid detection
                llm_result.metadata.update({
                    "hybrid": True,
                    "rule_intent": rule_result.intent.value,
                    "rule_confidence": rule_result.confidence,
                })

                return llm_result

            except Exception as e:
                logger.error(f"LLM fallback failed: {e}")
                # Return rule result even with low confidence
                return rule_result

        # No LLM fallback - return rule result
        logger.debug(
            f"LLM fallback disabled, using rule result: {rule_result.intent.value}"
        )
        return rule_result

    def supports_caching(self) -> bool:
        """Hybrid detector benefits from caching"""
        return True

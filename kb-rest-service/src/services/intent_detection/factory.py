"""
Factory for creating intent detectors (Factory Pattern).

Provides simple interface for creating appropriate detector based on configuration.
"""
import logging
from enum import Enum
from typing import Optional

from core.config import settings

from .detectors.hybrid import HybridIntentDetector
from .detectors.llm_based import LLMIntentDetector, SimpleLLMProvider
from .detectors.rule_based import RuleBasedIntentDetector
from .protocols import IntentDetector, LLMProvider

logger = logging.getLogger(__name__)


class DetectorType(str, Enum):
    """Supported detector types"""

    RULE_BASED = "rule"
    LLM_BASED = "llm"
    HYBRID = "hybrid"


class IntentDetectorFactory:
    """
    Factory for creating intent detectors.

    Usage:
        # Rule-based (default, fastest)
        detector = IntentDetectorFactory.create()

        # LLM-based (most accurate)
        detector = IntentDetectorFactory.create(
            detector_type="llm",
            llm_provider=my_provider
        )

        # Hybrid (best of both)
        detector = IntentDetectorFactory.create(
            detector_type="hybrid",
            llm_provider=my_provider
        )
    """

    @staticmethod
    def create(
        detector_type: Optional[str] = None,
        llm_provider: Optional[LLMProvider] = None,
        llm_function: Optional[callable] = None,
        **kwargs,
    ) -> IntentDetector:
        """
        Create intent detector.

        Args:
            detector_type: Type of detector ("rule", "llm", "hybrid")
            llm_provider: LLM provider instance (for llm/hybrid)
            llm_function: Simple LLM function (alternative to provider)
            **kwargs: Additional configuration for detector

        Returns:
            IntentDetector instance

        Examples:
            # Rule-based
            detector = IntentDetectorFactory.create()

            # LLM-based with custom provider
            detector = IntentDetectorFactory.create(
                detector_type="llm",
                llm_provider=MyLLMProvider()
            )

            # LLM-based with simple function
            detector = IntentDetectorFactory.create(
                detector_type="llm",
                llm_function=my_llm_function
            )

            # Hybrid with confidence threshold
            detector = IntentDetectorFactory.create(
                detector_type="hybrid",
                llm_provider=provider,
                confidence_threshold=0.85
            )
        """
        # Default to rule-based for serverless optimization
        detector_type = detector_type or settings.INTENT_DETECTOR_TYPE

        # Convert string to enum
        try:
            detector_enum = DetectorType(detector_type.lower())
        except ValueError:
            logger.warning(
                f"Unknown detector type '{detector_type}', defaulting to rule-based"
            )
            detector_enum = DetectorType.RULE_BASED

        # Create LLM provider if function provided
        if llm_function and not llm_provider:
            llm_provider = SimpleLLMProvider(llm_function)

        # Create detector based on type
        if detector_enum == DetectorType.RULE_BASED:
            return RuleBasedIntentDetector(**kwargs)

        elif detector_enum == DetectorType.LLM_BASED:
            if not llm_provider:
                logger.warning(
                    "LLM detector requested but no provider given, "
                    "falling back to rule-based"
                )
                return RuleBasedIntentDetector(**kwargs)

            return LLMIntentDetector(llm_provider=llm_provider, **kwargs)

        elif detector_enum == DetectorType.HYBRID:
            # Hybrid detector gracefully handles missing LLM provider
            return HybridIntentDetector(
                llm_provider=llm_provider,
                enable_llm_fallback=llm_provider is not None,
                **kwargs,
            )

        else:
            # Should never reach here, but fail gracefully
            logger.error(f"Unhandled detector type: {detector_enum}")
            return RuleBasedIntentDetector(**kwargs)

    @staticmethod
    def create_from_env() -> IntentDetector:
        """
        Create detector from settings.

        Settings used:
            INTENT_DETECTOR_TYPE: "rule", "llm", or "hybrid" (default: "rule")
            INTENT_CONFIDENCE_THRESHOLD: For hybrid detector (default: 0.8)
            INTENT_CACHE_ENABLED: Enable caching (default: True)
            INTENT_CACHE_TTL: Cache TTL in seconds (default: 600)

        Returns:
            Configured IntentDetector
        """
        detector_type = settings.INTENT_DETECTOR_TYPE
        confidence_threshold = settings.INTENT_CONFIDENCE_THRESHOLD
        enable_cache = settings.INTENT_CACHE_ENABLED
        cache_ttl = settings.INTENT_CACHE_TTL

        kwargs = {
            "enable_cache": enable_cache,
        }

        if detector_type == "hybrid":
            kwargs["confidence_threshold"] = confidence_threshold

        logger.info(
            f"Creating intent detector from settings: "
            f"type={detector_type}, cache={enable_cache}"
        )

        return IntentDetectorFactory.create(
            detector_type=detector_type,
            **kwargs,
        )

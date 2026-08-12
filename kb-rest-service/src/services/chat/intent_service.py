"""Wires the existing `intent_detection` module for the chat subsystem."""
from typing import Optional

from src.core.config import settings
from src.core.logging import get_logger
from src.services.intent_detection.detectors.rule_based import RuleBasedIntentDetector
from src.services.intent_detection.factory import IntentDetectorFactory
from src.services.intent_detection.protocols import IntentDetector

from .llm_provider import get_llm_provider

logger = get_logger(__name__)

_rule_based_detector: Optional[IntentDetector] = None


def get_chat_intent_detector(workspace_id: int, agent_id: Optional[int]) -> IntentDetector:
    """
    Build the configured intent detector.

    Rule-based (the default) needs no LLM and is cached as a process-wide
    singleton. "llm"/"hybrid" detectors need a workspace/agent-specific LLM
    provider (model routing is per-workspace), so those are built per call —
    cheap, since the underlying provider/manager is itself cached in
    `llm_provider.get_llm_provider`.
    """
    detector_type = settings.INTENT_DETECTOR_TYPE

    if detector_type == "rule":
        global _rule_based_detector
        if _rule_based_detector is None:
            _rule_based_detector = RuleBasedIntentDetector()
        return _rule_based_detector

    kwargs = {"enable_cache": settings.INTENT_CACHE_ENABLED}
    if detector_type == "hybrid":
        kwargs["confidence_threshold"] = settings.INTENT_CONFIDENCE_THRESHOLD

    return IntentDetectorFactory.create(
        detector_type=detector_type,
        llm_provider=get_llm_provider(workspace_id=workspace_id, agent_id=agent_id),
        **kwargs,
    )

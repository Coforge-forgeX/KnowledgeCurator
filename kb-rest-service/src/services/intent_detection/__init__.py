"""
Intent Detection Service

A production-grade intent detection system following SOLID principles,
optimized for serverless deployments and provider-agnostic LLM integration.
"""
from .factory import IntentDetectorFactory
from .models import Intent, IntentResult
from .protocols import IntentDetector

__all__ = [
    "IntentDetectorFactory",
    "Intent",
    "IntentResult",
    "IntentDetector",
]

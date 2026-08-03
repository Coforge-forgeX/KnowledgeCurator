"""Domain models for intent detection"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class Intent(str, Enum):
    """Supported user intents"""

    SEARCH_KB = "search_kb"
    UPLOAD_FILE = "upload_file"
    ADD_ENTITY = "add_entity"
    DELETE_ENTITY = "delete_entity"
    INDEX_URL = "index_url"
    UPDATE_ENTITY = "update_entity"
    DELETE_FILE = "delete_file"
    GREETING = "greeting"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentResult:
    """
    Immutable result of intent detection.

    Attributes:
        intent: Detected intent
        confidence: Detection confidence (0.0 to 1.0)
        method: Detection method used ('rule' or 'llm')
        metadata: Additional context (e.g., matched patterns, entities)
        timestamp: When detection occurred
    """

    intent: Intent
    confidence: float = field(default=1.0)
    method: str = field(default="rule")
    metadata: Dict[str, any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate confidence range"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass(frozen=True)
class IntentPattern:
    """
    Pattern definition for rule-based intent detection.

    Attributes:
        intent: Target intent for this pattern
        keywords: List of keywords to match
        phrases: List of exact phrases to match
        priority: Pattern priority (higher = checked first)
        requires_all: Whether all keywords must match (default: any)
    """

    intent: Intent
    keywords: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    priority: int = field(default=0)
    requires_all: bool = field(default=False)

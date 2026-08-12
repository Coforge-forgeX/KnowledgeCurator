"""
Intent patterns and pattern matching utilities.

Optimized for fast, deterministic intent detection without LLM calls.
"""
import re
from typing import List, Optional, Tuple

from ..models import Intent, IntentPattern


# Predefined patterns for each intent (KISS principle - simple and clear)
INTENT_PATTERNS: List[IntentPattern] = [
    # Greeting patterns (highest priority for common interactions)
    IntentPattern(
        intent=Intent.GREETING,
        keywords=["hello", "hi", "hey", "greetings"],
        phrases=["good morning", "good afternoon", "good evening", "what's up", "how are you"],
        priority=100,
    ),

    # Help patterns
    IntentPattern(
        intent=Intent.HELP,
        keywords=["help", "assist", "support", "guide"],
        phrases=[
            "what can you do",
            "how do i",
            "i need help",
            "can you help",
            "what are your capabilities",
        ],
        priority=90,
    ),

    # File upload patterns
    IntentPattern(
        intent=Intent.UPLOAD_FILE,
        keywords=["upload", "import", "attach", "index file", "index document"],
        phrases=[
            "add file",
            "add document",
            "attach file",
            "attach document",
            "import file",
            "import document",
        ],
        priority=80,
    ),

    # URL indexing patterns
    IntentPattern(
        intent=Intent.INDEX_URL,
        keywords=["index url", "index this url", "index the url"],
        phrases=[
            "index data from url",
            "index website",
            "add url",
            "crawl url",
        ],
        priority=75,
    ),

    # Search patterns (lowest priority - default fallback)
    IntentPattern(
        intent=Intent.SEARCH_KB,
        keywords=[
            "search",
            "find",
            "lookup",
            "look for",
            "what is",
            "tell me about",
            "what are",
            "information on",
            "information about",
            "describe",
            "explain",
            "show me",
        ],
        priority=10,
    ),
]


class IntentPatternMatcher:
    """
    Efficient pattern matcher for intent detection.

    Uses compiled regex patterns for optimal performance in serverless environments.
    Thread-safe and stateless (can be shared across requests).
    """

    def __init__(self, patterns: Optional[List[IntentPattern]] = None):
        """
        Initialize matcher with patterns.

        Args:
            patterns: List of intent patterns (defaults to INTENT_PATTERNS)
        """
        self.patterns = sorted(
            patterns or INTENT_PATTERNS,
            key=lambda p: p.priority,
            reverse=True,  # Check high-priority patterns first
        )

        # Compile regex patterns for efficiency (DRY principle)
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[Tuple[IntentPattern, List[re.Pattern]]]:
        """Compile regex patterns for each intent (optimization)"""
        compiled = []
        for pattern in self.patterns:
            regex_patterns = []

            # Compile keyword patterns (word boundaries for exact word matching)
            for keyword in pattern.keywords:
                # Escape special regex characters in keyword
                escaped = re.escape(keyword)
                regex_patterns.append(
                    re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                )

            # Compile phrase patterns
            for phrase in pattern.phrases:
                escaped = re.escape(phrase)
                regex_patterns.append(
                    re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                )

            compiled.append((pattern, regex_patterns))

        return compiled

    def match(self, message: str) -> Optional[Intent]:
        """
        Match message against patterns to find intent.

        Args:
            message: User message to analyze

        Returns:
            Matched Intent or None if no match
        """
        if not message or not message.strip():
            return None

        message_lower = message.lower().strip()

        # Check patterns in priority order (KISS - simple linear search)
        for pattern, regex_list in self._compiled_patterns:
            if self._pattern_matches(message_lower, pattern, regex_list):
                return pattern.intent

        return None

    def _pattern_matches(
        self,
        message: str,
        pattern: IntentPattern,
        regex_list: List[re.Pattern],
    ) -> bool:
        """
        Check if message matches a specific pattern.

        Args:
            message: Lowercase user message
            pattern: Pattern to check
            regex_list: Compiled regex patterns

        Returns:
            True if pattern matches
        """
        if not regex_list:
            return False

        matches = [regex.search(message) for regex in regex_list]
        match_count = sum(1 for m in matches if m is not None)

        if pattern.requires_all:
            # All patterns must match
            return match_count == len(regex_list)
        else:
            # At least one pattern must match
            return match_count > 0

    def get_confidence(self, message: str, intent: Intent) -> float:
        """
        Calculate confidence score for detected intent.

        Simple heuristic:
        - More pattern matches = higher confidence
        - Exact phrase match = highest confidence

        Args:
            message: User message
            intent: Detected intent

        Returns:
            Confidence score between 0.0 and 1.0
        """
        message_lower = message.lower().strip()

        # Find the matching pattern
        for pattern, regex_list in self._compiled_patterns:
            if pattern.intent != intent:
                continue

            # Check phrase matches (exact phrases = high confidence)
            for phrase in pattern.phrases:
                if phrase.lower() in message_lower:
                    return 1.0  # Exact phrase match

            # Count keyword matches
            keyword_matches = sum(
                1 for regex in regex_list if regex.search(message_lower)
            )

            if keyword_matches > 0:
                # Base confidence 0.7, +0.1 for each additional match (max 1.0)
                return min(0.7 + (keyword_matches - 1) * 0.1, 1.0)

        return 0.5  # Default medium confidence

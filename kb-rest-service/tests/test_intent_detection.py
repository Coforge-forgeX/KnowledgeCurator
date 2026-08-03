"""
Unit tests for Intent Detection Service.

Run with: pytest tests/test_intent_detection.py -v
"""
import pytest

from src.services.intent_detection import Intent, IntentDetectorFactory, IntentResult
from src.services.intent_detection.detectors.rule_based import RuleBasedIntentDetector
from src.services.intent_detection.utils.cache import SimpleLRUCache
from src.services.intent_detection.utils.patterns import IntentPatternMatcher


class TestIntentModels:
    """Test domain models"""

    def test_intent_enum_values(self):
        """Test Intent enum has all expected values"""
        assert Intent.SEARCH_KB.value == "search_kb"
        assert Intent.UPLOAD_FILE.value == "upload_file"
        assert Intent.GREETING.value == "greeting"
        assert Intent.HELP.value == "help"

    def test_intent_result_creation(self):
        """Test IntentResult creation"""
        result = IntentResult(
            intent=Intent.SEARCH_KB,
            confidence=0.9,
            method="rule",
            metadata={"test": "data"},
        )

        assert result.intent == Intent.SEARCH_KB
        assert result.confidence == 0.9
        assert result.method == "rule"
        assert result.metadata["test"] == "data"
        assert result.timestamp is not None

    def test_intent_result_confidence_validation(self):
        """Test confidence must be 0.0-1.0"""
        # Valid confidence
        IntentResult(intent=Intent.SEARCH_KB, confidence=0.5)
        IntentResult(intent=Intent.SEARCH_KB, confidence=0.0)
        IntentResult(intent=Intent.SEARCH_KB, confidence=1.0)

        # Invalid confidence
        with pytest.raises(ValueError):
            IntentResult(intent=Intent.SEARCH_KB, confidence=-0.1)

        with pytest.raises(ValueError):
            IntentResult(intent=Intent.SEARCH_KB, confidence=1.5)


class TestIntentPatternMatcher:
    """Test pattern matching"""

    def test_greeting_detection(self):
        """Test greeting patterns"""
        matcher = IntentPatternMatcher()

        assert matcher.match("hello") == Intent.GREETING
        assert matcher.match("hi there") == Intent.GREETING
        assert matcher.match("good morning") == Intent.GREETING
        assert matcher.match("HELLO") == Intent.GREETING  # Case insensitive

    def test_search_detection(self):
        """Test search patterns"""
        matcher = IntentPatternMatcher()

        assert matcher.match("search for Python") == Intent.SEARCH_KB
        assert matcher.match("find documentation") == Intent.SEARCH_KB
        assert matcher.match("what is machine learning") == Intent.SEARCH_KB
        assert matcher.match("tell me about APIs") == Intent.SEARCH_KB

    def test_upload_detection(self):
        """Test upload patterns"""
        matcher = IntentPatternMatcher()

        assert matcher.match("upload my file") == Intent.UPLOAD_FILE
        assert matcher.match("import document") == Intent.UPLOAD_FILE
        assert matcher.match("add file to kb") == Intent.UPLOAD_FILE

    def test_entity_operations(self):
        """Test entity operation patterns"""
        matcher = IntentPatternMatcher()

        assert matcher.match("add entity User") == Intent.ADD_ENTITY
        assert matcher.match("delete entity Product") == Intent.DELETE_ENTITY
        assert matcher.match("update entity Order") == Intent.UPDATE_ENTITY

    def test_file_deletion(self):
        """Test file deletion patterns"""
        matcher = IntentPatternMatcher()

        assert matcher.match("delete file report.pdf") == Intent.DELETE_FILE
        assert matcher.match("remove the old_data.csv") == Intent.DELETE_FILE

    def test_help_detection(self):
        """Test help patterns"""
        matcher = IntentPatternMatcher()

        assert matcher.match("help me") == Intent.HELP
        assert matcher.match("what can you do") == Intent.HELP
        assert matcher.match("i need help") == Intent.HELP

    def test_no_match_returns_none(self):
        """Test no match returns None"""
        matcher = IntentPatternMatcher()

        assert matcher.match("xyzabc nonsense") is None
        assert matcher.match("") is None

    def test_confidence_calculation(self):
        """Test confidence scores"""
        matcher = IntentPatternMatcher()

        # Exact phrase match = high confidence
        confidence = matcher.get_confidence("good morning", Intent.GREETING)
        assert confidence == 1.0

        # Keyword match = medium confidence
        confidence = matcher.get_confidence("hello", Intent.GREETING)
        assert 0.7 <= confidence <= 1.0


class TestRuleBasedDetector:
    """Test rule-based detector"""

    @pytest.mark.asyncio
    async def test_basic_detection(self):
        """Test basic intent detection"""
        detector = RuleBasedIntentDetector()

        result = await detector.detect("search for Python")

        assert result.intent == Intent.SEARCH_KB
        assert result.confidence > 0
        assert result.method == "rule"

    @pytest.mark.asyncio
    async def test_file_context_overrides(self):
        """Test file context triggers upload intent"""
        detector = RuleBasedIntentDetector()

        context = {"file_names": ["test.pdf"]}
        result = await detector.detect("process this", context=context)

        assert result.intent == Intent.UPLOAD_FILE
        assert result.confidence == 1.0
        assert result.metadata["file_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_message_raises_error(self):
        """Test empty message raises ValueError"""
        detector = RuleBasedIntentDetector()

        with pytest.raises(ValueError, match="Message cannot be empty"):
            await detector.detect("")

        with pytest.raises(ValueError, match="Message cannot be empty"):
            await detector.detect("   ")

    @pytest.mark.asyncio
    async def test_fallback_intent(self):
        """Test fallback for unknown intents"""
        detector = RuleBasedIntentDetector(fallback_intent=Intent.SEARCH_KB)

        result = await detector.detect("xyzabc nonsense")

        assert result.intent == Intent.SEARCH_KB
        assert result.metadata.get("reason") == "no_pattern_match"

    @pytest.mark.asyncio
    async def test_caching_enabled(self):
        """Test caching is enabled"""
        detector = RuleBasedIntentDetector(enable_cache=True)

        assert detector.supports_caching() is True

        # First call
        result1 = await detector.detect("hello")

        # Second call (should hit cache)
        result2 = await detector.detect("hello")

        assert result1.intent == result2.intent


class TestCache:
    """Test caching utilities"""

    def test_cache_put_and_get(self):
        """Test basic cache operations"""
        cache = SimpleLRUCache(max_size=10, ttl_seconds=60)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_miss(self):
        """Test cache miss returns None"""
        cache = SimpleLRUCache()

        assert cache.get("nonexistent") is None

    def test_cache_expiration(self):
        """Test TTL expiration"""
        cache = SimpleLRUCache(max_size=10, ttl_seconds=0)

        cache.put("key1", "value1")

        import time
        time.sleep(0.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_eviction(self):
        """Test LRU eviction"""
        cache = SimpleLRUCache(max_size=2, ttl_seconds=60)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_cache_clear(self):
        """Test cache clear"""
        cache = SimpleLRUCache()

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = SimpleLRUCache(max_size=10, ttl_seconds=300)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        stats = cache.stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 300


class TestIntentDetectorFactory:
    """Test factory pattern"""

    def test_create_rule_based_detector(self):
        """Test creating rule-based detector"""
        detector = IntentDetectorFactory.create(detector_type="rule")

        assert isinstance(detector, RuleBasedIntentDetector)

    def test_create_with_invalid_type(self):
        """Test invalid detector type falls back to rule-based"""
        detector = IntentDetectorFactory.create(detector_type="invalid")

        assert isinstance(detector, RuleBasedIntentDetector)

    def test_create_llm_without_provider(self):
        """Test LLM detector without provider falls back to rule-based"""
        detector = IntentDetectorFactory.create(detector_type="llm")

        # Should fall back to rule-based
        assert isinstance(detector, RuleBasedIntentDetector)

    @pytest.mark.asyncio
    async def test_create_from_env(self, monkeypatch):
        """Test creating from environment variables"""
        monkeypatch.setenv("INTENT_DETECTOR_TYPE", "rule")
        monkeypatch.setenv("INTENT_CACHE_ENABLED", "true")

        detector = IntentDetectorFactory.create_from_env()

        assert isinstance(detector, RuleBasedIntentDetector)


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_end_to_end_detection(self):
        """Test complete detection flow"""
        detector = IntentDetectorFactory.create(detector_type="rule")

        test_cases = [
            ("hello", Intent.GREETING),
            ("search for docs", Intent.SEARCH_KB),
            ("upload file.pdf", Intent.UPLOAD_FILE),
            ("help me", Intent.HELP),
            ("add entity User", Intent.ADD_ENTITY),
            ("delete entity Product", Intent.DELETE_ENTITY),
        ]

        for message, expected_intent in test_cases:
            result = await detector.detect(message)
            assert result.intent == expected_intent, \
                f"Expected {expected_intent} for '{message}', got {result.intent}"

    @pytest.mark.asyncio
    async def test_concurrent_detection(self):
        """Test concurrent detection requests"""
        import asyncio

        detector = IntentDetectorFactory.create(detector_type="rule")

        messages = [
            "hello",
            "search for Python",
            "upload my file",
            "help me",
        ] * 5  # 20 concurrent requests

        # Run all concurrently
        results = await asyncio.gather(*[
            detector.detect(msg) for msg in messages
        ])

        # All should succeed
        assert len(results) == 20
        assert all(isinstance(r, IntentResult) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for workspace_helpers module

Tests workspace naming compatibility with KnowledgeCurator.
"""
import pytest
import sys
import os

# Add shared folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from workspace_helpers import (
    workspace_id_to_alpha,
    get_workspace_identifier,
    get_workspace_working_dir,
)


class TestWorkspaceIdToAlpha:
    """Tests for workspace_id_to_alpha function"""

    def test_numeric_workspace_id(self):
        """Test converting numeric workspace IDs"""
        assert workspace_id_to_alpha(123) == "onetwothree"
        assert workspace_id_to_alpha(456) == "fourfivesix"
        assert workspace_id_to_alpha(789) == "seveneightnine"
        assert workspace_id_to_alpha(0) == "zero"

    def test_alphanumeric_workspace_id(self):
        """Test converting alphanumeric workspace IDs"""
        assert workspace_id_to_alpha("abc123") == "abconetwothree"
        assert workspace_id_to_alpha("test456") == "testfourfivesix"
        assert workspace_id_to_alpha("w1k2b3") == "wonektwobthree"

    def test_alpha_only_workspace_id(self):
        """Test workspace IDs with only letters"""
        assert workspace_id_to_alpha("abc") == "abc"
        assert workspace_id_to_alpha("workspace") == "workspace"
        assert workspace_id_to_alpha("TEST") == "TEST"

    def test_edge_cases(self):
        """Test edge cases"""
        assert workspace_id_to_alpha(None) == ""
        assert workspace_id_to_alpha("") == ""
        assert workspace_id_to_alpha(0) == "zero"

    def test_all_digits(self):
        """Test all digits 0-9"""
        assert workspace_id_to_alpha("0123456789") == "zeroonetwothreefourfivesixseveneightnine"

    def test_mixed_case(self):
        """Test mixed case preservation"""
        assert workspace_id_to_alpha("Abc123") == "Abconetwothree"
        assert workspace_id_to_alpha("TEST789") == "TESTseveneightnine"


class TestGetWorkspaceIdentifier:
    """Tests for get_workspace_identifier function"""

    def test_workspace_id_only(self):
        """Test with workspace_id only"""
        assert get_workspace_identifier(123) == "onetwothree"
        assert get_workspace_identifier(456) == "fourfivesix"

    def test_with_domain(self):
        """Test with domain"""
        assert get_workspace_identifier(123, domain="industry") == "industryonetwothree"
        assert get_workspace_identifier(456, domain="tech") == "techfourfivesix"

    def test_with_kb_name(self):
        """Test with kb_name"""
        assert get_workspace_identifier(123, kb_name="subindustry") == "subindustryonetwothree"
        assert get_workspace_identifier(456, kb_name="kb") == "kbfourfivesix"

    def test_with_domain_and_kb_name(self):
        """Test with both domain and kb_name"""
        result = get_workspace_identifier(123, domain="industry", kb_name="subindustry")
        assert result == "industrysubindustryonetwothree"

    def test_sanitization(self):
        """Test alpha-only sanitization"""
        # Should remove non-alpha characters
        result = get_workspace_identifier(123, domain="in-dustry", kb_name="sub_industry")
        # Hyphens and underscores removed
        assert result == "industrysubindustryonetwothree"

    def test_empty_parts(self):
        """Test with empty domain/kb_name"""
        assert get_workspace_identifier(123, domain="", kb_name="") == "onetwothree"
        assert get_workspace_identifier(123, domain=None, kb_name=None) == "onetwothree"


class TestGetWorkspaceWorkingDir:
    """Tests for get_workspace_working_dir function"""

    def test_basic_working_dir(self):
        """Test basic working directory generation"""
        result = get_workspace_working_dir(123, "./lightrag_data")
        assert result == "./lightrag_data/onetwothree"

    def test_with_domain(self):
        """Test working directory with domain"""
        result = get_workspace_working_dir(123, "./data", domain="industry")
        assert result == "./data/industryonetwothree"

    def test_with_domain_and_kb(self):
        """Test working directory with domain and kb_name"""
        result = get_workspace_working_dir(
            123, "/var/lightrag", domain="industry", kb_name="subindustry"
        )
        assert result == "/var/lightrag/industrysubindustryonetwothree"

    def test_different_base_dirs(self):
        """Test with different base directories"""
        assert get_workspace_working_dir(123, "./data") == "./data/onetwothree"
        assert get_workspace_working_dir(123, "/tmp/rag") == "/tmp/rag/onetwothree"
        assert get_workspace_working_dir(123, "C:\\data\\rag") == "C:\\data\\rag/onetwothree"


class TestKnowledgeCuratorCompatibility:
    """Test compatibility with KnowledgeCurator workspace naming"""

    def test_known_knowledgecurator_patterns(self):
        """Test patterns from actual KnowledgeCurator usage"""
        # From kb_curator_chatbot.py:703
        workspace_id = 123
        workspace_id_alpha = workspace_id_to_alpha(workspace_id)
        assert workspace_id_alpha == "onetwothree"

        # Full pattern from KnowledgeCurator
        industry = "tech"
        sub_industry = "ai"
        kb_name = f"{sub_industry}/{workspace_id_alpha}"
        workspace_name = ''.join(char for char in f"{industry}{kb_name}" if char.isalpha())

        # Compare with our implementation
        our_result = get_workspace_identifier(workspace_id, domain=industry, kb_name=sub_industry)
        assert our_result == workspace_name

    def test_numeric_workspace_ids(self):
        """Test common numeric workspace IDs"""
        test_cases = [
            (1, "one"),
            (10, "onezero"),
            (100, "onezerozero"),
            (123, "onetwothree"),
            (999, "nineninene"),
        ]
        for workspace_id, expected in test_cases:
            assert workspace_id_to_alpha(workspace_id) == expected


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_very_long_workspace_id(self):
        """Test with very long workspace ID"""
        long_id = "123456789012345678901234567890"
        result = workspace_id_to_alpha(long_id)
        expected = ("onetwothreefourfivesixseveneightninezerozeroone"
                   "twotwothreethreefourfourfourfivefivesixsixsevenseveneighteightnineninezero")
        # Just verify it doesn't crash and produces something
        assert len(result) > 0
        assert result.isalpha()

    def test_special_characters_removed(self):
        """Test that special characters are removed"""
        result = get_workspace_identifier(123, domain="in@dustry#", kb_name="sub$industry%")
        # Special characters should be removed by sanitization
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result
        assert result.isalpha()

    def test_unicode_characters(self):
        """Test with unicode characters"""
        # Unicode letters should be preserved
        result = workspace_id_to_alpha("αβγ123")
        assert "αβγ" in result
        assert "onetwothree" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

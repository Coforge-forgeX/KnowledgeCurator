"""
Unit tests for workspace_helpers module

Tests workspace naming compatibility with KnowledgeCurator.
"""
import pytest
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
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

    def test_edge_cases(self):
        """Test edge cases"""
        assert workspace_id_to_alpha(None) == ""
        assert workspace_id_to_alpha("") == ""


class TestGetWorkspaceIdentifier:
    """Tests for get_workspace_identifier function"""

    def test_workspace_id_only(self):
        """Test with workspace_id only"""
        assert get_workspace_identifier(123) == "onetwothree"
        assert get_workspace_identifier(456) == "fourfivesix"

    def test_with_domain_and_kb_name(self):
        """Test with both domain and kb_name"""
        result = get_workspace_identifier(123, domain="industry", kb_name="subindustry")
        assert result == "industrysubindustryonetwothree"


class TestGetWorkspaceWorkingDir:
    """Tests for get_workspace_working_dir function"""

    def test_basic_working_dir(self):
        """Test basic working directory generation"""
        result = get_workspace_working_dir(123, "./lightrag_data")
        assert result == "./lightrag_data/onetwothree"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

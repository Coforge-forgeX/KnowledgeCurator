"""
Workspace Resolver - Helper Utilities

Resolves workspace identifiers and knowledge base names.
Follows Single Responsibility Principle.
"""
from typing import Optional

from src.core.logging import get_logger

logger = get_logger(__name__)


class WorkspaceResolver:
    """
    Resolves workspace identifiers to alphanumeric representations.

    Design:
    - Pure functions: No side effects
    - Immutable: Returns new values, doesn't modify input
    - Clear naming: Methods describe what they do
    """

    # Digit to word mapping for workspace ID conversion
    DIGIT_MAP = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }

    @classmethod
    def workspace_id_to_alpha(cls, workspace_id: Optional[int]) -> str:
        """
        Convert workspace ID to alphanumeric representation.

        Args:
            workspace_id: Numeric workspace ID

        Returns:
            Alphanumeric representation (e.g., 123 -> "onetwothree")

        Examples:
            >>> WorkspaceResolver.workspace_id_to_alpha(123)
            'onetwothree'
            >>> WorkspaceResolver.workspace_id_to_alpha(None)
            ''
        """
        if workspace_id is None:
            return ""

        result = []
        for char in str(workspace_id):
            if char.isalpha():
                result.append(char)
            elif char.isdigit():
                result.append(cls.DIGIT_MAP[char])
            # Ignore other characters

        alpha_id = ''.join(result)
        logger.debug(f"Converted workspace_id {workspace_id} to '{alpha_id}'")
        return alpha_id

    @classmethod
    def build_kb_name(
        cls,
        base_kb_name: str,
        workspace_id: Optional[int] = None,
        include_workspace: bool = True
    ) -> str:
        """
        Build knowledge base name with optional workspace scoping.

        Args:
            base_kb_name: Base KB name (e.g., "AssetManagement")
            workspace_id: Optional workspace ID for scoping
            include_workspace: Whether to include workspace in name

        Returns:
            Scoped or unscoped KB name

        Examples:
            >>> WorkspaceResolver.build_kb_name("AssetManagement", 123)
            'AssetManagement/onetwothree'
            >>> WorkspaceResolver.build_kb_name("AssetManagement", None)
            'AssetManagement'
        """
        if not include_workspace or not workspace_id:
            return base_kb_name

        workspace_alpha = cls.workspace_id_to_alpha(workspace_id)
        if not workspace_alpha:
            return base_kb_name

        kb_name = f"{base_kb_name}/{workspace_alpha}"
        logger.debug(f"Built scoped KB name: {kb_name}")
        return kb_name

    @classmethod
    def build_workspace_name(
        cls,
        domain: str,
        kb_name: str,
        workspace_suffix: str = ""
    ) -> str:
        """
        Build workspace name for LightRAG (alphanumeric only).

        Args:
            domain: Domain name
            kb_name: Knowledge base name
            workspace_suffix: Optional workspace suffix

        Returns:
            Alphanumeric workspace name

        Examples:
            >>> WorkspaceResolver.build_workspace_name("Banking", "AssetMgmt", "123")
            'BankingAssetMgmtonethree'
        """
        combined = f"{domain}{kb_name}{workspace_suffix}"
        # Keep only alphabetic characters
        workspace_name = ''.join(char for char in combined if char.isalpha())

        logger.debug(
            f"Built workspace name: domain={domain}, "
            f"kb={kb_name}, suffix={workspace_suffix} -> {workspace_name}"
        )
        return workspace_name

    @classmethod
    def resolve_kb_list(
        cls,
        workspace_id: Optional[int],
        knowledge_bases: Optional[list] = None
    ) -> list:
        """
        Resolve final list of knowledge bases to query.

        Adds workspace-scoped KB if workspace_id provided.

        Args:
            workspace_id: Workspace identifier
            knowledge_bases: Optional list of KB names

        Returns:
            Complete list of KBs to query

        Examples:
            >>> WorkspaceResolver.resolve_kb_list(123, ["kb1", "kb2"])
            ['kb1', 'kb2', 'onetwothree']
        """
        kb_list = list(knowledge_bases) if knowledge_bases else []

        if workspace_id:
            workspace_alpha = cls.workspace_id_to_alpha(workspace_id)
            if workspace_alpha and workspace_alpha not in kb_list:
                kb_list.append(workspace_alpha)
                logger.debug(f"Added workspace KB '{workspace_alpha}' to query list")

        return kb_list


class BlobPathBuilder:
    """
    Builds blob storage paths for documents.

    Design: Single Responsibility - only builds paths
    """

    @staticmethod
    def build_upload_path(
        domain: str,
        kb_name: str,
        workspace_id: Optional[int] = None,
        role_id: Optional[int] = None
    ) -> str:
        """
        Build blob storage upload path.

        Args:
            domain: Domain name
            kb_name: Knowledge base name
            workspace_id: Optional workspace ID
            role_id: Optional role ID

        Returns:
            Blob storage path

        Examples:
            >>> BlobPathBuilder.build_upload_path("Banking", "AssetMgmt", 123)
            'Banking/AssetMgmt/123'
        """
        parts = [domain, kb_name]

        # Add workspace ID if not SME role (role_id != 34)
        if workspace_id and role_id != 34:
            parts.append(str(workspace_id))

        path = '/'.join(parts)
        logger.debug(f"Built upload path: {path}")
        return path

    @staticmethod
    def build_blob_path(
        domain: str,
        kb_name: str,
        file_name: str,
        workspace_id: Optional[int] = None
    ) -> str:
        """
        Build complete blob path for a file.

        Args:
            domain: Domain name
            kb_name: Knowledge base name
            file_name: File name
            workspace_id: Optional workspace ID

        Returns:
            Complete blob path

        Examples:
            >>> BlobPathBuilder.build_blob_path("Banking", "AssetMgmt", "doc.pdf", 123)
            'Banking/AssetMgmt/123/doc.pdf'
        """
        parts = [domain, kb_name]
        if workspace_id:
            parts.append(str(workspace_id))
        parts.append(file_name)

        path = '/'.join(parts)
        logger.debug(f"Built blob path: {path}")
        return path


def workspace_id_to_alpha(workspace_id: Optional[int]) -> str:
    """Convenience function for workspace ID conversion"""
    return WorkspaceResolver.workspace_id_to_alpha(workspace_id)


def build_kb_name(
    base_kb_name: str,
    workspace_id: Optional[int] = None
) -> str:
    """Convenience function for building KB names"""
    return WorkspaceResolver.build_kb_name(base_kb_name, workspace_id)

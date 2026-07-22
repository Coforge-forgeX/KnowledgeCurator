"""
Workspace Helper Functions

Utilities for workspace identification and naming conventions.
Compatible with KnowledgeCurator workspace naming.
"""


def workspace_id_to_alpha(workspace_id) -> str:
    """
    Convert workspace_id to alpha representation by converting digits to words.

    This function maintains compatibility with KnowledgeCurator's workspace naming.

    Examples:
        >>> workspace_id_to_alpha(123)
        'onetwothree'
        >>> workspace_id_to_alpha("abc456")
        'abcfourfivesix'
        >>> workspace_id_to_alpha(None)
        ''

    Args:
        workspace_id: Workspace identifier (int or str)

    Returns:
        str: Alpha representation where digits are converted to words
    """
    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    result = []
    for c in str(workspace_id or ""):
        if c.isalpha():
            result.append(c)
        elif c.isdigit():
            result.append(digit_map[c])
    return ''.join(result)


def get_workspace_identifier(
    workspace_id: int,
    domain: str = None,
    kb_name: str = None,
) -> str:
    """
    Get LightRAG workspace identifier for a workspace.

    Compatible with KnowledgeCurator's workspace naming convention:
    - Converts workspace_id to alpha (123 → "onetwothree")
    - Constructs full workspace path if domain/kb_name provided
    - Sanitizes to alpha characters only

    Examples:
        >>> get_workspace_identifier(123)
        'onetwothree'
        >>> get_workspace_identifier(123, domain="industry", kb_name="subindustry")
        'industrysubindustryonetwothree'

    Args:
        workspace_id: Numeric workspace identifier
        domain: Optional domain/industry name
        kb_name: Optional knowledge base/sub-industry name

    Returns:
        str: Sanitized alpha-only workspace identifier
    """
    # Convert workspace ID to alpha
    workspace_id_alpha = workspace_id_to_alpha(workspace_id)

    # Build full workspace name
    parts = []
    if domain:
        parts.append(domain)
    if kb_name:
        parts.append(kb_name)
    parts.append(workspace_id_alpha)

    full_name = ''.join(parts)

    # Sanitize to alpha characters only (same as KnowledgeCurator)
    workspace_name = ''.join(char for char in full_name if char.isalpha())

    return workspace_name


def get_workspace_working_dir(
    workspace_id: int,
    base_dir: str = "./lightrag_data",
    domain: str = None,
    kb_name: str = None,
) -> str:
    """
    Get LightRAG working directory for a workspace.

    Compatible with KnowledgeCurator's directory structure.

    Args:
        workspace_id: Numeric workspace identifier
        base_dir: Base directory for LightRAG data
        domain: Optional domain name
        kb_name: Optional knowledge base name

    Returns:
        str: Full path to workspace working directory
    """
    workspace_identifier = get_workspace_identifier(workspace_id, domain, kb_name)
    return f"{base_dir}/{workspace_identifier}"

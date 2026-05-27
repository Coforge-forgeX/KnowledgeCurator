from ..server.main import os
import sys
from ..server import server
from ..server.server import mcp
from common_adapters.sharepoint import (
    test_sharepoint_connection as test_sp_connection,
    toggle_sharepoint_connection as toggle_sp_connection,
    SharePointService,
    SharePointClient)
import asyncio
from typing import Any, Optional
import logging
from dateutil.parser import parse as parse_date
from dateutil.tz import UTC

logger = logging.getLogger("sharepoint_agent")

        
@mcp.tool()
async def test_sharepoint_connection(
    workspace_id: str,
    user_id: str,
    data: dict[str, Any]
) -> dict[str, Any]:
    """
    This tool tests the SharePoint connection for the given workspace and user.
    Args:
        workspace_id: unique identifier for the workspace.
        user_id: unique identifier for the user.
        data: dictionary containing tenant_id, client_id, client_secret, site_hostname, site_path.
    """
    return await test_sp_connection(
        workspace_id=workspace_id,
        user_id=user_id,
        data=data,
        sharepoint_client_class=SharePointClient,
        user_config_manager=server.user_config_manager
    )

@mcp.tool()
async def toggle_sharepoint_connection(
    workspace_id: str,
    user_id: str,
    enable: bool
) -> dict[str, Any]:
    """
    This tool enables or disables the SharePoint connection for the given workspace and user.
    Args:
        workspace_id: unique identifier for the workspace.
        user_id: unique identifier for the user.
        enable: flag to enable or disable the connection.
    """
    return await toggle_sp_connection(
        workspace_id=workspace_id,
        user_id=user_id,
        enable=enable,
        sharepoint_client_manager=server.sharepoint_client_manager,
        user_config_manager=server.user_config_manager
    )

@mcp.tool()
async def extract_sharepoint_data(
    workspace_id: str,
    user_id: str,
    conversation_id: str,
    folder_path: str = "",
    file_types: Optional[list[str]] = None,
    name_contains: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    modified_after: Optional[str] = None,
    modified_before: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Extract text and metadata from SharePoint documents using Azure Document Intelligence.
    Args:
        workspace_id: unique identifier for the workspace.
        user_id: unique identifier for the user.
        conversation_id: current conversation/session identifier.
        folder_path: SharePoint folder path to scan (e.g. "Shared Documents/Invoices"). Defaults to root.
        file_types: file extensions to include, with or without leading dot (e.g. ["pdf", ".docx"]).
        name_contains: only include files whose name contains this substring (case-insensitive).
        min_size: minimum file size in bytes.
        max_size: maximum file size in bytes.
        created_after: ISO 8601 datetime string; only include files created after this time.
        created_before: ISO 8601 datetime string; only include files created before this time.
        modified_after: ISO 8601 datetime string; only include files modified after this time.
        modified_before: ISO 8601 datetime string; only include files modified before this time.
        tags: Optional dict mapping SharePoint internal column names to required values. Value can be a string (exact match, case-insensitive) or a list (any-of match).
            Example:
                tags={
                    "Status": "Approved",          # exact match (case-insensitive)
                    "Category": ["Win", "Success"], # any-of match
                }
    Returns:
        dict with "documents" list (each item has id, name, text, metadata) and "count".
        Each document's metadata includes a "tags" dict with all SharePoint column values.
    """
    try:
        client = await server.sharepoint_client_manager.get_client(
            workspace_id=workspace_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not client:
            return {"error": "Failed to obtain SharePoint client.", "documents": [], "count": 0}

        metadata_map: dict[str, Any] = {}
        if file_types:
            metadata_map["file_types"] = file_types
        if name_contains:
            metadata_map["name_contains"] = name_contains
        if min_size is not None:
            metadata_map["min_size"] = min_size
        if max_size is not None:
            metadata_map["max_size"] = max_size
        if created_after:
            metadata_map["created_after"] = created_after
        if created_before:
            metadata_map["created_before"] = created_before
        if modified_after:
            metadata_map["modified_after"] = modified_after
        if modified_before:
            metadata_map["modified_before"] = modified_before
        if tags:
            metadata_map["tags"] = tags

        loop = asyncio.get_event_loop()
        service = SharePointService(client)
        documents = await loop.run_in_executor(
            None,
            lambda: service.extract_data(
                folder_path=folder_path,
                metadata_map=metadata_map or None,
            ),
        )
        # Sort documents by modification date descending (latest first)
        def get_mod_date(doc):
            # Prefer 'modified_at' at top level, then check metadata and other common fields
            if 'modified_at' in doc and doc['modified_at']:
                return doc['modified_at']
            meta = doc.get('metadata', {})
            return meta.get('modified_at') or meta.get('modified') or meta.get('modified_date') or meta.get('lastModified')

        def parse_mod_date(doc):
            date_str = get_mod_date(doc)
            logger.debug(f"Document: {doc.get('source', doc.get('name', 'unknown'))}, modified_at: {date_str}")
            if date_str:
                try:
                    dt = parse_date(date_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    return dt
                except Exception as ex:
                    logger.warning(f"Failed to parse date '{date_str}' for document {doc.get('source', doc.get('name', 'unknown'))}: {ex}")
                    return None
            return None

        documents_sorted = sorted(
            documents,
            key=lambda doc: parse_mod_date(doc) or '',
            reverse=True
        )
        return {"documents": documents_sorted, "count": len(documents_sorted)}
    except Exception as e:
        logger.error(f"extract_sharepoint_data failed: {e}")
        return {"error": str(e), "documents": [], "count": 0}

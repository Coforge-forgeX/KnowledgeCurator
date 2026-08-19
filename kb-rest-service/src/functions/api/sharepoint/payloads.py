"""
Pydantic Request & Response Payloads for SharePoint Operations in V2 REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------- Test Connection ----------------------
class SharePointCredentials(BaseModel):
    tenant_id: str = Field(..., description="Azure AD Tenant ID")
    client_id: str = Field(..., description="Azure AD Client ID")
    client_secret: str = Field(..., description="Azure AD Client Secret")
    site_hostname: str = Field(..., description="SharePoint site hostname (e.g. contoso.sharepoint.com)")
    site_path: str = Field(..., description="SharePoint site path (e.g. /sites/Marketing)")


class TestSharePointConnectionRequest(BaseModel):
    workspace_id: str | int = Field(..., description="Workspace ID")
    user_id: str | int = Field(..., description="User ID")
    data: SharePointCredentials = Field(..., description="SharePoint credentials")


class TestSharePointConnectionResponse(BaseModel):
    status: str = Field(..., description="Status string ('success' or 'error')")
    message: str = Field(..., description="Human readable result message")
    exists: Optional[bool] = Field(None, description="Whether config matches stored config")


# ---------------------- Toggle Connection ----------------------
class ToggleSharePointConnectionRequest(BaseModel):
    workspace_id: str | int = Field(..., description="Workspace ID")
    user_id: str | int = Field(..., description="User ID")
    enable: bool = Field(..., description="Flag to enable or disable SharePoint integration")


class ToggleSharePointConnectionResponse(BaseModel):
    status: str = Field(..., description="Status string ('success' or 'error')")
    message: str = Field(..., description="Status message")
    sharepoint_active: bool = Field(..., description="Current active state")


# ---------------------- Extract Data ----------------------
class ExtractSharePointDataRequest(BaseModel):
    workspace_id: str | int = Field(..., description="Workspace ID")
    user_id: str | int = Field(..., description="User ID")
    conversation_id: Optional[str] = Field(None, description="Active conversation session ID")
    folder_path: str = Field("", description="SharePoint folder path to scan (defaults to root)")
    file_types: Optional[List[str]] = Field(None, description="Allowed file extensions (e.g. ['pdf', 'docx'])")
    name_contains: Optional[str] = Field(None, description="Case-insensitive filename substring filter")
    min_size: Optional[int] = Field(None, description="Minimum file size in bytes")
    max_size: Optional[int] = Field(None, description="Maximum file size in bytes")
    created_after: Optional[str] = Field(None, description="ISO datetime string for creation lower bound")
    created_before: Optional[str] = Field(None, description="ISO datetime string for creation upper bound")
    modified_after: Optional[str] = Field(None, description="ISO datetime string for modification lower bound")
    modified_before: Optional[str] = Field(None, description="ISO datetime string for modification upper bound")
    tags: Optional[Dict[str, Any]] = Field(None, description="Dict of custom list column names to target values")
    credentials: Optional[SharePointCredentials] = Field(None, description="SharePoint credentials")


class ExtractSharePointDataResponse(BaseModel):
    success: bool = Field(..., description="Boolean indicating success status")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted document list")
    count: int = Field(0, description="Total documents returned")
    error: Optional[str] = Field(None, description="Error message if extraction failed")

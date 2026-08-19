"""Pydantic Request & Response Payloads for Workspace Download ZIP API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class WorkspaceDownloadZipRequest(BaseModel):
    workspace_id: int = Field(..., description="Target Workspace ID")
    limit: Optional[int] = Field(1000, description="Max files to include in zip archive (default 1000)")
    expiry_minutes: Optional[int] = Field(30, description="Download URL validity duration in minutes (default 30 = 30 min)")
    user_id_filter: Optional[int] = Field(None, description="Optional filter to only include files uploaded by a specific user ID")
    include_kb_files: Optional[bool] = Field(False, description="Whether to include files linked to workspace Knowledge Base / Knowledge Graph (default False)")

class WorkspaceDownloadZipResponse(BaseModel):
    success: bool = Field(..., description="Success flag")
    message: str = Field(..., description="Human readable summary message")
    workspace_id: int = Field(..., description="Workspace ID")
    total_files_zipped: int = Field(..., description="Number of files included in ZIP archive")
    total_size_bytes: int = Field(..., description="Total size of the generated ZIP file in bytes")
    zip_file_name: str = Field(..., description="Filename of generated ZIP")
    download_url: Optional[str] = Field(None, description="Signed URL to download the ZIP file")
    expires_in_seconds: int = Field(..., description="Validity duration of download URL in seconds")
    failed_files: List[str] = Field(default_factory=list, description="Files that could not be downloaded/zipped")
    skipped_files: List[str] = Field(default_factory=list, description="Files skipped (e.g. invalid or missing path)")

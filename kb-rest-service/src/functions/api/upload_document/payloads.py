"""Request payloads for upload_document endpoint"""
from typing import Optional, Dict
from pydantic import Field

from shared.payloads import BasePayload, NonEmptyStr


class UploadDocumentRequest(BasePayload):
    """Upload document request payload"""

    workspace_id: int = Field(..., description="Workspace ID", gt=0)
    document_text: NonEmptyStr = Field(..., description="Document content")
    file_name: NonEmptyStr = Field(..., description="Original file name")
    metadata: Optional[Dict] = Field(default=None, description="Document metadata")

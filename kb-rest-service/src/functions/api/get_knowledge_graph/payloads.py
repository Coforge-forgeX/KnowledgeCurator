"""Payload models for Get Knowledge Graph API"""
from pydantic import Field

from src.common.payloads import BasePayload


class GetKnowledgeGraphRequest(BasePayload):
    """Request payload for getting knowledge graph"""

    workspace_id: int = Field(..., gt=0, description="Workspace ID")

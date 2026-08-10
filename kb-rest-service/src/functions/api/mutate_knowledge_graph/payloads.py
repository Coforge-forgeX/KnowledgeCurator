"""Payload models for graph mutation endpoint."""
from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator, model_validator

from src.shared.payloads import BasePayload


class GraphMutationScope(BasePayload):
    """Scope that anchors mutation to indexed content in one workspace."""

    file_path: str = Field(..., min_length=1, description="Indexed file path in the target workspace")
    source_id: Optional[str] = Field(default=None, description="Optional source/chunk identifier")
    full_doc_id: Optional[str] = Field(default=None, description="Optional full document identifier")

    @field_validator("file_path")
    @classmethod
    def normalize_file_path(cls, value: str) -> str:
        cleaned = (value or "").strip().replace("\\", "/")
        if not cleaned:
            raise ValueError("file_path cannot be empty")
        return cleaned

    @field_validator("source_id", "full_doc_id")
    @classmethod
    def normalize_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class NodeMutationPayload(BasePayload):
    """Node-level mutation data."""

    entity_name: str = Field(..., min_length=1, description="Current entity name (match key for update/delete)")
    new_entity_name: Optional[str] = Field(default=None, min_length=1, description="New entity name for rename")
    entity_type: Optional[str] = Field(default=None, description="Entity type")
    description: Optional[str] = Field(default=None, description="Entity description")
    source_id: Optional[str] = Field(default=None, description="Optional source/chunk ID on the node")
    additional_properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("entity_name", "new_entity_name", "entity_type", "description", "source_id")
    @classmethod
    def normalize_text_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class RelationshipMutationPayload(BasePayload):
    """Relationship-level mutation data."""

    source: str = Field(..., min_length=1, description="Source entity name")
    target: str = Field(..., min_length=1, description="Target entity name")
    relation: str = Field(..., min_length=1, description="Current relation value (match key for update/delete)")
    new_relation: Optional[str] = Field(default=None, min_length=1, description="New relation value for update")
    description: Optional[str] = Field(default=None, description="Relationship description")
    source_id: Optional[str] = Field(default=None, description="Optional source/chunk ID")
    additional_properties: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("source", "target", "relation", "new_relation", "description", "source_id")
    @classmethod
    def normalize_text_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class MutateKnowledgeGraphRequest(BasePayload):
    """Request model for graph mutation endpoint."""

    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    action: Literal["create", "update", "delete"] = Field(..., description="Mutation action")
    target: Literal["node", "relationship"] = Field(..., description="Mutation target")
    scope: GraphMutationScope
    node: Optional[NodeMutationPayload] = Field(default=None)
    relationship: Optional[RelationshipMutationPayload] = Field(default=None)

    @model_validator(mode="after")
    def validate_target_payload(self):
        if self.target == "node" and self.node is None:
            raise ValueError("node payload is required when target='node'")
        if self.target == "relationship" and self.relationship is None:
            raise ValueError("relationship payload is required when target='relationship'")
        if self.target == "node" and self.relationship is not None:
            raise ValueError("relationship payload must be omitted when target='node'")
        if self.target == "relationship" and self.node is not None:
            raise ValueError("node payload must be omitted when target='relationship'")
        return self

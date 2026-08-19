"""Payload models for graph mutation endpoint."""
from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator, model_validator

from src.common.payloads import BasePayload


class GraphMutationScope(BasePayload):
    """Optional scope anchoring mutation to indexed content."""

    file_path: Optional[str] = Field(default=None, description="Indexed file path in the target workspace")
    source_id: Optional[str] = Field(default=None, description="Optional source/chunk identifier")
    full_doc_id: Optional[str] = Field(default=None, description="Optional full document identifier")

    @field_validator("file_path", "source_id", "full_doc_id")
    @classmethod
    def normalize_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().replace("\\", "/")
        return cleaned or None


class NodeMutationPayload(BasePayload):
    """Simplified node-level mutation payload."""

    element_id: Optional[str] = Field(default=None, description="Neo4j node element ID (elementId)")
    elementId: Optional[str] = Field(default=None, description="Alias for element_id")
    entity_id: Optional[str] = Field(default=None, description="Entity ID/name property in Neo4j")
    entity_name: Optional[str] = Field(default=None, description="Alias for entity_id")
    description: Optional[str] = Field(default=None, description="Entity description")
    entity_type: Optional[str] = Field(default=None, description="Entity type (optional for create)")
    is_custom: Optional[bool] = Field(default=True, description="Flag indicating externally added entity")

    @model_validator(mode="before")
    @classmethod
    def populate_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "elementId" in data and not data.get("element_id"):
                data["element_id"] = data["elementId"]
            elif "element_id" in data and not data.get("elementId"):
                data["elementId"] = data["element_id"]
            if "entity_name" in data and not data.get("entity_id"):
                data["entity_id"] = data["entity_name"]
            elif "entity_id" in data and not data.get("entity_name"):
                data["entity_name"] = data["entity_id"]
        return data

    @field_validator("element_id", "elementId", "entity_id", "entity_name", "description", "entity_type")
    @classmethod
    def normalize_text_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class RelationshipMutationPayload(BasePayload):
    """Simplified relationship-level mutation payload."""

    element_id: Optional[str] = Field(default=None, description="Neo4j relationship element ID (elementId)")
    elementId: Optional[str] = Field(default=None, description="Alias for element_id")
    description: Optional[str] = Field(default=None, description="Relationship description (only field editable for update)")
    relation: Optional[str] = Field(default=None, description="Relationship type (for create/delete)")
    source: Optional[str] = Field(default=None, description="Source entity name (for create/delete)")
    target: Optional[str] = Field(default=None, description="Target entity name (for create/delete)")
    is_custom: Optional[bool] = Field(default=True, description="Flag indicating externally added relationship")

    @model_validator(mode="before")
    @classmethod
    def populate_element_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "elementId" in data and not data.get("element_id"):
                data["element_id"] = data["elementId"]
            elif "element_id" in data and not data.get("elementId"):
                data["elementId"] = data["element_id"]
        return data

    @field_validator("element_id", "elementId", "relation", "description", "source", "target")
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
    scope: Optional[GraphMutationScope] = Field(default=None, description="Optional scope")
    node: Optional[NodeMutationPayload] = Field(default=None)
    relationship: Optional[RelationshipMutationPayload] = Field(default=None)

    @model_validator(mode="after")
    def validate_target_payload(self):
        if self.target == "node":
            if self.node is None:
                raise ValueError("node payload is required when target='node'")
            if self.action == "update":
                eid = self.node.element_id or self.node.elementId
                if not eid or not str(eid).strip():
                    raise ValueError("element_id (or elementId) is required for updating a node")
            if self.relationship is not None:
                raise ValueError("relationship payload must be omitted when target='node'")
        if self.target == "relationship":
            if self.relationship is None:
                raise ValueError("relationship payload is required when target='relationship'")
            if self.action == "update":
                eid = self.relationship.element_id or self.relationship.elementId
                if not eid or not str(eid).strip():
                    raise ValueError("element_id (or elementId) is required for updating a relationship")
            if self.node is not None:
                raise ValueError("node payload must be omitted when target='relationship'")
        return self

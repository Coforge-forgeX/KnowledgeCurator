"""
Fetch Graph API Payloads

Request and response models for the fetch_graph endpoint.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class FetchGraphRequest(BaseModel):
    """
    Request model for fetch_graph endpoint.

    This endpoint fetches graph data relevant to a specific query and answer.
    It uses LLM to filter only the nodes/relationships related to the answer.
    """

    # ============================================================================
    # Required Fields
    # ============================================================================

    query: str = Field(
        ...,
        description="User's original question",
        min_length=1,
        max_length=5000
    )
    answer: str = Field(
        ...,
        description="Generated answer for which to fetch relevant graph data",
        min_length=1,
        max_length=50000
    )
    workspace_id: int = Field(
        ...,
        description="Workspace ID for fetching graph data",
        ge=1
    )

    # ============================================================================
    # Optional Fields
    # ============================================================================

    mode: str = Field(
        default="hybrid",
        description="Query strategy - hybrid (recommended), local, global, naive, or mix"
    )
    graph_only: bool = Field(
        default=False,
        description="If true, bypass query-evidence cache and use LightRAG graph retrieval directly"
    )
    agent_id: Optional[int] = Field(
        default=1,
        description="Agent ID for workspace/agent-specific LLM routing (default: 1)"
    )

    @validator("query")
    def validate_query(cls, v):
        """Validate query is not empty after stripping"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()

    @validator("answer")
    def validate_answer(cls, v):
        """Validate answer is not empty after stripping"""
        if not v.strip():
            raise ValueError("Answer cannot be empty or whitespace")
        return v.strip()

    @validator("mode")
    def validate_mode(cls, v):
        """Validate mode is valid"""
        valid_modes = ["naive", "local", "global", "hybrid", "mix"]
        if v.lower() not in valid_modes:
            raise ValueError(f"Mode must be one of: {', '.join(valid_modes)}")
        return v.lower()

    @validator("agent_id")
    def validate_agent_id(cls, v):
        """Validate agent id when provided."""
        if v is None:
            return 1
        if v < 1:
            raise ValueError("agent_id must be >= 1")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is asset management?",
                "answer": "Asset management is the process of...",
                "workspace_id": 123
            }
        }


class GraphNodeModel(BaseModel):
    """Single graph node"""
    id: str = Field(..., description="Node ID (value after last colon)")
    element_id: Optional[str] = Field(None, description="Full Neo4j element ID")
    labels: List[str] = Field(default_factory=list, description="Labels for the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")


class GraphEdgeModel(BaseModel):
    """Single graph edge"""
    id: str = Field(..., description="Edge ID (value after last colon)")
    element_id: Optional[str] = Field(None, description="Full Neo4j element ID")
    type: str = Field(default="DIRECTED", description="Edge type")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge properties")

# Alias for backward compatibility if imported elsewhere
GraphRelationshipModel = GraphEdgeModel


class FilteredGraphDataModel(BaseModel):
    """Filtered graph data with nodes, edges, knowledge bases, and metadata"""
    knowledge_bases: List[str] = Field(default_factory=list, description="Knowledge bases from which answer/graph was retrieved")
    nodes: List[GraphNodeModel] = Field(default_factory=list, description="Filtered graph nodes")
    edges: List[GraphEdgeModel] = Field(default_factory=list, description="Filtered graph edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FetchGraphResponse(BaseModel):
    """Response model for fetch_graph endpoint"""

    knowledge_bases: List[str] = Field(default_factory=list, description="Knowledge bases from which answer/graph was retrieved")
    nodes: List[GraphNodeModel] = Field(default_factory=list, description="Filtered graph nodes")
    edges: List[GraphEdgeModel] = Field(default_factory=list, description="Filtered graph edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    query: str = Field(..., description="Original query")
    workspace_id: int = Field(..., description="Workspace ID")
    graph_only: bool = Field(default=False, description="Whether graph_only mode was enabled")
    cached: bool = Field(..., description="Whether result was from cache")

    class Config:
        json_schema_extra = {
            "example": {
                "knowledge_bases": [
                    "Other/Demo Instances/New knowledgebase 12"
                ],
                "nodes": [
                    {
                        "id": "21",
                        "labels": ["Product Data"],
                        "properties": {
                            "file_path": "Other/Demo Instances/New knowledgebase 12/OOSD 1 1 (1).docx",
                            "entity_type": "data",
                            "truncate": "",
                            "description": "Product Data is information persisted during order creation.",
                            "created_at": 1786082306,
                            "source_id": "chunk-6962bb82e73d6ef4e82c18ce0d6e3398",
                            "entity_id": "Product Data"
                        }
                    }
                ],
                "edges": [
                    {
                        "id": "32",
                        "type": "DIRECTED",
                        "source": "0",
                        "target": "30",
                        "properties": {
                            "file_path": "Other/Demo Instances/New knowledgebase 12/OOSD 1 1 (1).docx",
                            "truncate": "",
                            "keywords": "process initiation,refund handling",
                            "weight": 1,
                            "description": "Refund Process is triggered during refund handling in the Order Workflow.",
                            "created_at": 1786082312,
                            "source_id": "chunk-6962bb82e73d6ef4e82c18ce0d6e3398"
                        }
                    }
                ],
                "metadata": {
                    "total_nodes": 1,
                    "total_edges": 1
                },
                "query": "What is product data?",
                "workspace_id": 123,
                "graph_only": False,
                "cached": False
            }
        }

 
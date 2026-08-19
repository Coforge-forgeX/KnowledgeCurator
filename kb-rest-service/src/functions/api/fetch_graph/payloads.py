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
    """Single graph node/entity"""
    element_id: Optional[str] = Field(None, description="Neo4j element ID")
    entity_name: str = Field(..., description="Name of the entity")
    entity_type: str = Field(..., description="Type of the entity")
    created_at: Optional[Any] = Field(None, description="Creation timestamp")
    description: Optional[str] = Field(None, description="Description of the entity")
    file_path: Optional[Any] = Field(None, description="Source file path(s)")
    source_id: Optional[Any] = Field(None, description="Source chunk ID(s)")


class GraphRelationshipModel(BaseModel):
    """Single graph relationship/edge"""
    element_id: Optional[str] = Field(None, description="Neo4j element ID for relationship")
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation: str = Field(..., description="Relationship type")
    created_at: Optional[Any] = Field(None, description="Creation timestamp")
    description: Optional[str] = Field(None, description="Relationship description")
    file_path: Optional[Any] = Field(None, description="Source file path(s)")
    keywords: Optional[Any] = Field(None, description="Relationship keywords")
    source_id: Optional[Any] = Field(None, description="Source chunk ID(s)")
    weight: Optional[Any] = Field(None, description="Relationship weight")


class FilteredGraphDataModel(BaseModel):
    """Filtered graph data with only relevant nodes and relationships"""
    entities: List[GraphNodeModel] = Field(default_factory=list, description="Filtered entities/nodes")
    relationships: List[GraphRelationshipModel] = Field(default_factory=list, description="Filtered relationships/edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FetchGraphResponse(BaseModel):
    """Response model for fetch_graph endpoint"""

    graph_data: FilteredGraphDataModel = Field(..., description="Filtered graph data")
    query: str = Field(..., description="Original query")
    workspace_id: int = Field(..., description="Workspace ID")
    cached: bool = Field(..., description="Whether result was from cache")

    class Config:
        json_schema_extra = {
            "example": {
                "graph_data": {
                    "entities": [
                        {
                            "entity_name": "Asset Management",
                            "entity_type": "Concept",
                            "description": "Process of managing financial assets",
                            "source_id": "chunk_1",
                            "file_path": "documents/asset_guide.pdf"
                        }
                    ],
                    "relationships": [
                        {
                            "source": "Asset Management",
                            "target": "Portfolio",
                            "relation": "manages",
                            "description": "Asset management manages portfolios"
                        }
                    ],
                    "metadata": {
                        "total_entities": 1,
                        "total_relationships": 1
                    }
                },
                "query": "What is asset management?",
                "workspace_id": 123,
                "cached": False
            }
        }

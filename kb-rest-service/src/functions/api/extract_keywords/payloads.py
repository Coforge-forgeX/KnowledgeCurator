"""
Extract Keywords API Payloads

Request and response models for extract_keywords_from_query endpoint.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ExtractKeywordsRequest(BaseModel):
    """
    Request payload model for extract_keywords_from_query endpoint.
    """

    user_query: str = Field(
        ...,
        alias="query",
        description="User query or input text to extract keywords from",
        min_length=1,
        max_length=5000
    )
    node_labels: Optional[List[str]] = Field(
        default_factory=list,
        description="Candidate node labels in the knowledge graph to match against"
    )
    history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional conversation history for context"
    )
    workspace_id: int = Field(
        ...,
        description="Workspace ID for LLM configuration routing",
        ge=1
    )

    agent_id: Optional[int] = Field(
        default=1,
        description="Agent ID for LLM routing (default: 1)"
    )

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "user_query": "What is the order creation process and product data stored?",
                "node_labels": ["Order Workflow", "Product Data", "Customer Info", "Payment Processing"],
                "workspace_id": 123
            }
        }

    @validator("user_query", pre=True)
    def validate_query(cls, v):
        """Ensure text is a non-empty string."""
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("user_query must be a non-empty string.")
        return v.strip()


class ExtractKeywordsResponse(BaseModel):
    """
    Response model for extract_keywords_from_query endpoint.
    """

    keywords: List[str] = Field(
        ...,
        description="Extracted node labels relevant to the user query"
    )
    user_query: str = Field(
        ...,
        description="Original input user query"
    )
    node_labels: List[str] = Field(
        default_factory=list,
        description="Original candidate node labels provided"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["Order Workflow", "Product Data"],
                "user_query": "What is the order creation process and product data stored?",
                "node_labels": ["Order Workflow", "Product Data", "Customer Info", "Payment Processing"]
            }
        }
 
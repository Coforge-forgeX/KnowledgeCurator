"""Request payload models for query_kb endpoint"""
from src.shared.payloads import BasePayload, NonEmptyStr, VALID_QUERY_MODES
from pydantic import Field, validator


class QueryKBRequest(BasePayload):
    """Query knowledge base request payload"""

    query: NonEmptyStr = Field(..., description="Query text")
    workspace_id: int = Field(..., description="Workspace ID", gt=0)
    mode: str = Field(default="hybrid", description="Query mode")
    only_need_context: bool = Field(
        default=False, description="Return only context"
    )

    @validator("mode")
    def validate_mode(cls, v):
        if v not in VALID_QUERY_MODES:
            raise ValueError(f"mode must be one of {VALID_QUERY_MODES}")
        return v

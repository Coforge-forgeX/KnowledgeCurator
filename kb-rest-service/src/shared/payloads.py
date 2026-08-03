"""Shared request-payload validation primitives"""
import json
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, validator
from typing_extensions import Annotated

from src.core.abstractions import AbstractRequest, AbstractResponse

# Required, non-empty string (leading/trailing whitespace stripped)
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Optional non-empty string
OptionalNonEmptyStr = Annotated[
    Optional[str], StringConstraints(strip_whitespace=True, min_length=1)
]

# Valid query modes for LightRAG
VALID_QUERY_MODES = ["naive", "local", "global", "hybrid"]


class BasePayload(BaseModel):
    """Base for all request payloads: reject unknown fields explicitly."""

    model_config = ConfigDict(extra="forbid")


def _extract_data(req: AbstractRequest) -> dict:
    """Collect raw payload dict from request body (POST) or query params."""
    if req.method in ("POST", "DELETE", "PUT", "PATCH"):
        try:
            body = req.get_json()
            if isinstance(body, dict):
                return body
        except (ValueError, TypeError):
            pass
        return {}
    # GET: use query string parameters
    return req.get_query_params()


def _validation_error_response(exc: ValidationError) -> AbstractResponse:
    """Build 400 response describing pydantic validation failures."""
    errors = [
        {
            "field": ".".join(str(p) for p in err.get("loc", ())),
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    first = errors[0] if errors else {"field": "", "message": "Invalid payload"}
    field = first["field"] or "payload"
    error_response = {
        "success": False,
        "message": f"Invalid request: {field} - {first['message']}",
        "error": "VALIDATION_ERROR",
        "validation_errors": errors,
    }
    return AbstractResponse(
        body=json.dumps(error_response, default=str),
        status_code=400,
        mimetype="application/json",
    )


def parse_request(
    req: AbstractRequest, model: type[BaseModel]
) -> Tuple[Optional[BaseModel], Optional[AbstractResponse]]:
    """Validate request against model.

    Returns (payload, None) on success or (None, error_response) with
    400 AbstractResponse when validation fails.
    """
    data = _extract_data(req)
    try:
        return model.model_validate(data), None
    except ValidationError as exc:
        return None, _validation_error_response(exc)


# Common payload models


class PaginationPayload(BasePayload):
    """Pagination parameters"""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class QueryRequestPayload(BasePayload):
    """Query request payload"""

    query: NonEmptyStr = Field(..., description="Query text")
    mode: str = Field(default="hybrid", description="Query mode")
    workspace_id: Optional[int] = Field(default=None, description="Workspace ID")
    only_need_context: bool = Field(
        default=False, description="Return only context without answer"
    )

    @validator("mode")
    def validate_mode(cls, v):
        if v not in VALID_QUERY_MODES:
            raise ValueError(f"mode must be one of {VALID_QUERY_MODES}")
        return v


class DocumentIndexPayload(BasePayload):
    """Document indexing payload"""

    text: NonEmptyStr = Field(..., description="Document text to index")
    workspace_id: int = Field(..., description="Workspace ID")
    metadata: Optional[dict] = Field(default=None, description="Document metadata")


class BulkDocumentIndexPayload(BasePayload):
    """Bulk document indexing payload"""

    documents: List[dict] = Field(..., description="List of documents to index")
    workspace_id: int = Field(..., description="Workspace ID")


class DocumentDeletePayload(BasePayload):
    """Document deletion payload"""

    doc_ids: List[str] = Field(..., description="List of document IDs to delete")
    workspace_id: int = Field(..., description="Workspace ID")

"""
Pydantic Request & Response Payloads for User Configuration Operations in V2 API.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class GetConfigRequest(BaseModel):
    workspace_id: Union[int, str] = Field(..., description="Workspace identifier")
    fields: Optional[List[str]] = Field(None, description="Optional list of configuration fields to retrieve")

    @model_validator(mode="before")
    @classmethod
    def reject_user_id_in_payload(cls, data: Any) -> Any:
        if isinstance(data, dict) and "user_id" in data:
            raise ValueError(
                "user_id must not be passed in request payload; identity is extracted securely from JWT token."
            )
        return data


class UpdateConfigRequest(BaseModel):
    workspace_id: Union[int, str] = Field(..., description="Workspace identifier")
    data: Dict[str, Any] = Field(..., description="Configuration key-value updates")

    @model_validator(mode="before")
    @classmethod
    def reject_user_id_in_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "user_id" in data:
                raise ValueError(
                    "user_id must not be passed in request payload; identity is extracted securely from JWT token."
                )
            if "data" in data and isinstance(data["data"], dict):
                forbidden_keys = {"user_id", "_id", "created_at"}
                found = forbidden_keys.intersection(data["data"].keys())
                if found:
                    raise ValueError(f"Forbidden system keys in configuration update data: {', '.join(found)}")
        return data


class ConfigResponse(BaseModel):
    status: str = Field(..., description="Response status ('success' or 'error')")
    data: Optional[Dict[str, Any]] = Field(None, description="Configuration dictionary result")
    message: Optional[str] = Field(None, description="Informational or error message")
    details: Optional[str] = Field(None, description="Error details if applicable")
 
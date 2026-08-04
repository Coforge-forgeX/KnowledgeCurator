"""Payload models for Check Indexing Status API"""
from typing import List
from pydantic import Field

from src.shared.payloads import BasePayload


class CheckIndexingStatusRequest(BasePayload):
    """Request payload for checking indexing status"""

    task_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of task IDs to check")

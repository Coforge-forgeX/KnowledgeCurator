"""
WebSocket Message Protocol

Defines the message types exchanged between server and browser clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


# ============================================================================
# Server -> Browser Events
# ============================================================================


@dataclass
class StatusEvent:
    """Connection status update."""

    type: Literal["status"] = "status"
    text: str = ""
    tab_id: Optional[str] = None


@dataclass
class ProgressEvent:
    """Progress update for long-running operations."""

    type: Literal["progress"] = "progress"
    status: str = ""
    message: str = ""
    progress_percent: Optional[float] = None
    job_id: Optional[str] = None
    conversation_id: Optional[str] = None
    run_id: Optional[str] = None
    event_seq: Optional[int] = None


@dataclass
class AssistantMessageEvent:
    """Message from the assistant/agent."""

    type: Literal["assistant_message"] = "assistant_message"
    text: str = ""
    role: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PingEvent:
    """Server heartbeat ping."""

    type: Literal["ping"] = "ping"
    ts: int = 0


@dataclass
class FieldDef:
    """Field definition for elicitation requests."""

    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ElicitationRequestEvent:
    """Request for user input from the server."""

    type: Literal["elicitation_request"] = "elicitation_request"
    elicitation_id: str = ""
    message: str = ""
    fields: List[FieldDef] = field(default_factory=list)
    draft: Optional[Dict[str, Any]] = None


# Union type for all server events
ServerEvent = Union[
    StatusEvent,
    ProgressEvent,
    AssistantMessageEvent,
    PingEvent,
    ElicitationRequestEvent,
]


# ============================================================================
# Browser -> Server Commands
# ============================================================================


@dataclass
class UserMessageCommand:
    """User message sent to the agent."""

    type: Literal["user_message"] = "user_message"
    text: str = ""
    role: Literal["developer", "product_owner", "business_analyst"] = "business_analyst"
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ElicitationResponseCommand:
    """Response to an elicitation request."""

    type: Literal["elicitation_response"] = "elicitation_response"
    elicitation_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    # payload.action: "accept" | "decline" | "cancel"
    # payload.data: { field_name: value, ... }


@dataclass
class PongCommand:
    """Client response to server ping."""

    type: Literal["pong"] = "pong"
    ts: int = 0


# Union type for all client commands
ClientCommand = Union[
    UserMessageCommand,
    ElicitationResponseCommand,
    PongCommand,
]


# ============================================================================
# Serialization helpers
# ============================================================================


def event_to_dict(event: ServerEvent) -> Dict[str, Any]:
    """Convert a server event to a dictionary for JSON serialization."""
    from dataclasses import asdict

    result = asdict(event)
    # Filter out None values for cleaner JSON
    return {k: v for k, v in result.items() if v is not None}


def parse_client_command(data: Dict[str, Any]) -> Optional[ClientCommand]:
    """Parse a dictionary into a ClientCommand."""
    msg_type = data.get("type")

    if msg_type == "user_message":
        return UserMessageCommand(
            text=data.get("text", ""),
            role=data.get("role", "business_analyst"),
            conversation_id=data.get("conversation_id"),
            metadata=data.get("metadata"),
        )
    elif msg_type == "elicitation_response":
        return ElicitationResponseCommand(
            elicitation_id=data.get("elicitation_id", ""),
            payload=data.get("payload", {}),
        )
    elif msg_type == "pong":
        return PongCommand(ts=data.get("ts", 0))

    return None

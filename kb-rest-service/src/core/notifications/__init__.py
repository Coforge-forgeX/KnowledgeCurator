"""
Push Notification Service

Real-time event delivery to browser clients via WebSocket:
- Connection management with session/tab routing
- Broadcast modes (tab-only vs session-wide)
- Heartbeat/keepalive for proxy compatibility
- Backpressure handling with bounded queues
"""

from .connection_manager import ConnectionManager, Connection
from .protocol import (
    ServerEvent,
    ClientCommand,
    StatusEvent,
    ProgressEvent,
    AssistantMessageEvent,
    PingEvent,
    ElicitationRequestEvent,
    UserMessageCommand,
    ElicitationResponseCommand,
    PongCommand,
)
from .broadcast import BroadcastMode, load_broadcast_mode

__all__ = [
    # Connection management
    "ConnectionManager",
    "Connection",
    # Protocol types
    "ServerEvent",
    "ClientCommand",
    "StatusEvent",
    "ProgressEvent",
    "AssistantMessageEvent",
    "PingEvent",
    "ElicitationRequestEvent",
    "UserMessageCommand",
    "ElicitationResponseCommand",
    "PongCommand",
    # Broadcast
    "BroadcastMode",
    "load_broadcast_mode",
]

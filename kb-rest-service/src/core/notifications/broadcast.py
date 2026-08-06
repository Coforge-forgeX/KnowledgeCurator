"""
Broadcast Mode Configuration

Controls how messages are delivered to browser tabs:
- "tab": Send only to the initiating tab
- "session": Broadcast to all tabs in the same user session

Elicitation requests always go to the initiating tab to avoid collisions.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal


class BroadcastMode(str, Enum):
    """
    Broadcast mode for server-to-client messages.

    TAB: Send messages only to the initiating browser tab.
         Best for tab-specific operations and avoiding duplicate UI updates.

    SESSION: Broadcast messages to all tabs sharing the same user session.
             Best for keeping multiple tabs in sync (e.g., notifications).
    """

    TAB = "tab"
    SESSION = "session"


def load_broadcast_mode() -> BroadcastMode:
    """
    Load broadcast mode from environment.

    Environment variable:
    - BROADCAST_MODE: "tab" (default) or "session"

    Returns:
        BroadcastMode enum value
    """
    raw = os.getenv("BROADCAST_MODE", "tab").lower().strip()

    if raw == "session":
        return BroadcastMode.SESSION

    return BroadcastMode.TAB


def get_broadcast_mode_str() -> Literal["tab", "session"]:
    """Get broadcast mode as a string literal."""
    return load_broadcast_mode().value

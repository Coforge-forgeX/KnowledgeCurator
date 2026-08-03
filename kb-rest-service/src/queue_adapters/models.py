"""Queue models and data classes"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class QueueMessage:
    """
    Queue message wrapper.

    Attributes:
        content: Message content (dict or string)
        message_id: Unique message ID (provider-specific)
        receipt_handle: Receipt handle for deletion (provider-specific, optional)
    """

    content: Dict[str, Any]
    message_id: Optional[str] = None
    receipt_handle: Optional[str] = None

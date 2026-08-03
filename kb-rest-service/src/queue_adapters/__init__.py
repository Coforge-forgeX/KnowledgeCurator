"""Queue abstraction layer for multi-cloud deployment"""

from .factory import QueueFactory, get_queue_adapter
from .models import QueueMessage
from .protocols import QueueAdapter

__all__ = [
    "QueueAdapter",
    "QueueFactory",
    "QueueMessage",
    "get_queue_adapter",
]

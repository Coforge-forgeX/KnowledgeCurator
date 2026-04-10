from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatbotContext:
    """ Track conversation context and pending operations. """
    session_id: str
    conversation_history: list[dict] = None
    pending_confirmation: Optional[dict] = None
    last_intent: Optional[str] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "pending_confirmation": self.pending_confirmation,
            "last_intent": self.last_intent
        }

    @staticmethod
    def from_dict(data):
        return ChatbotContext(
            session_id=data["session_id"],
            conversation_history=data.get("conversation_history", []),
            pending_confirmation=data.get("pending_confirmation"),
            last_intent=data.get("last_intent")
        )
import logging
import os
import uuid
from datetime import datetime
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from configparser import ConfigParser

load_dotenv(os.path.abspath(os.path.join(os.getcwd(),'.env')))

class SessionHistoryManager:
    def __init__(self, mongo_client):
        try:
            self.chat_collection = mongo_client.chatbot_db["kb_chat_history"]
            self.context_collection = mongo_client.chatbot_db["contexts"]
        except Exception as e:
            logging.error(f"Error in MongoDB connection: {e}")
            raise

    def save_context(self, context):
        # context should be a ChatbotContext or dict with session_id
        self.context_collection.update_one(
            {"session_id": context.session_id},
            {"$set": context.to_dict() if hasattr(context, 'to_dict') else context},
            upsert=True
        )

    def load_context(self, session_id):
        doc = self.context_collection.find_one({"session_id": session_id})
        if doc:
            try:
                from agent_search.utils.chatbot_context import ChatbotContext  # if you want to avoid circular import, or import at top if safe
                return ChatbotContext.from_dict(doc)
            except Exception:
                return doc
        return None

    @staticmethod
    def create_session():
        """Generate a new session ID."""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def append_message(self, workspace_id, user_id, session_id, role, content, task_ids):
        try:
            doc = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "tasks": task_ids,
                "timestamp": datetime.utcnow()
            }
            insert_result = self.chat_collection.insert_one(doc)
            return insert_result.inserted_id
        except Exception as e:
            logging.error(f"Error in append_message: {e}")
            return None

    def get_recent_sessions(self, workspace_id, user_id, limit=5):
        try:
            
            query = {"workspace_id": workspace_id, "user_id": user_id}
            sessions = self.chat_collection.distinct("session_id", query)
            sessions = [str(s) for s in sessions if s]
            return sessions[-limit:] if sessions else ["No sessions found"]
        except Exception as e:
            logging.error(f"Error in get_recent_sessions: {e}")
            return ["Error fetching sessions"]
        

    def load_history(self, workspace_id, user_id, session_id):
        try:
            query = {"workspace_id": workspace_id, "user_id": user_id, "session_id": session_id}
            messages = list(self.chat_collection.find(query).sort("timestamp", 1))
            return [{"role": m["role"], "content": m["content"], "timestamp": m["timestamp"], "task_ids":m.get("task_ids",None)} for m in messages]
        except Exception as e:
            logging.error(f"Error in load_history: {e}")
            return []
        
    def delete_session(self, workspace_id, user_id, session_id):
        try:
            query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id
            }
            
            delete_result = self.chat_collection.delete_many(query)
            return {
                "deleted_count": delete_result.deleted_count,
                "status": "success" if delete_result.deleted_count > 0 else "no records found"
            }
        except Exception as e:
            logging.error(f"Error in delete_session: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
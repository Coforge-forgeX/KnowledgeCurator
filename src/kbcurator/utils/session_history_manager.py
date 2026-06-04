import logging
import os
import uuid
from datetime import datetime, timedelta 
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from configparser import ConfigParser

# Load .env file if it exists (for local development)
env_path = os.path.abspath(os.path.join(os.getcwd(), '.env'))
if os.path.exists(env_path):
    load_dotenv(env_path)

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
                from kbcurator.utils.chatbot_context import ChatbotContext
                return ChatbotContext.from_dict(doc)
            except Exception:
                return doc
        return None

    @staticmethod
    def create_session():
        """Generate a new session ID."""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # def append_message(self, workspace_id, user_id, session_id, role, content, task_ids):
    #     try:
    #         doc = {
    #             "workspace_id": workspace_id,
    #             "user_id": user_id,
    #             "session_id": session_id,
    #             "role": role,
    #             "content": content,
    #             "tasks": task_ids,
    #             "timestamp": datetime.utcnow()
    #         }
    #         insert_result = self.chat_collection.insert_one(doc)
    #         return insert_result.inserted_id
    #     except Exception as e:
    #         logging.error(f"Error in append_message: {e}")
    #         return None

    def append_message(self, workspace_id, user_id, session_id, role, content, task_ids_or_sources):
        try:
            # Detect if it's sources (list of dicts with download_url) or task_ids
            is_sources = (
                isinstance(task_ids_or_sources, list) and 
                task_ids_or_sources and 
                isinstance(task_ids_or_sources[0], dict) and 
                'download_url' in task_ids_or_sources[0]
            )
            
            doc = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "sources": task_ids_or_sources if is_sources else [],
                "tasks": task_ids_or_sources if not is_sources else [],
                "timestamp": datetime.utcnow()
            }
            insert_result = self.chat_collection.insert_one(doc)
            return insert_result.inserted_id
        except Exception as e:
            logging.error(f"Error in append_message: {e}")
            return None
    def get_recent_sessions_by_ttl(self, workspace_id, user_id, current_time: datetime, ttl_seconds: float = 900):
        # Compute cutoff time
        cutoff_time = current_time - timedelta(seconds=ttl_seconds)
        
        pipeline = [
            {"$match": {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "timestamp": {"$gte": cutoff_time}
            }},
            {"$group": {
                "_id": "$session_id",
                "latest_timestamp": {"$max": "$timestamp"}
            }},
            {"$sort": {"latest_timestamp": -1}}
        ]

        try:
            sessions = list(self.chat_collection.aggregate(pipeline))
            # print(sessions)
            return [str(s["_id"]) for s in sessions]
        except Exception as e:
            logging.error(f"Error in get_recent_sessions_by_ttl: {e}")
            raise

    def get_recent_sessions(self, workspace_id, user_id, limit=5):
        try:
            
            query = {"workspace_id": workspace_id, "user_id": user_id}
            sessions = self.chat_collection.distinct("session_id", query)
            sessions = [str(s) for s in sessions if s]
            return sessions[-limit:] if sessions else ["No sessions found"]
        except Exception as e:
            logging.error(f"Error in get_recent_sessions: {e}")
            return ["Error fetching sessions"]
        

    # def load_history(self, workspace_id, user_id, session_id):
    #     try:
    #         query = {"workspace_id": workspace_id, "user_id": user_id, "session_id": session_id}
    #         messages = list(self.chat_collection.find(query).sort("timestamp", 1))
    #         return [{"role": m["role"], "content": m["content"], "timestamp": m["timestamp"], "task_ids":m.get("task_ids",None)} for m in messages]
    #     except Exception as e:
    #         logging.error(f"Error in load_history: {e}")
    #         return []

    def load_history(self, workspace_id, user_id, session_id):
        try:
            query = {"workspace_id": workspace_id, "user_id": user_id, "session_id": session_id}
            messages = list(self.chat_collection.find(query).sort("timestamp", 1))
            return [
                {
                    "role": m["role"], 
                    "content": m["content"], 
                    "timestamp": m["timestamp"], 
                    "session_id": m["session_id"],
                    "task_ids": m.get("tasks", None),  # Fixed: read from "tasks" not "task_ids"
                    "sources": m.get("sources", [])     # Added: return sources field
                } 
                for m in messages
            ]
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

    def set_conversation_title(self, workspace_id, user_id, session_id, title):
        """Set/update the conversation title in the context collection."""
        try:
            filter_query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id
            }
            update_data = {
                "$set": {
                    "title": title,
                    "timestamp": datetime.utcnow()
                }
            }
            result = self.context_collection.update_one(filter_query, update_data, upsert=True)
            
            if result.upserted_id:
                return {
                    "status": "success",
                    "operation": "created",
                    "message": "Title created successfully"
                }
            else:
                return {
                    "status": "success",
                    "operation": "updated",
                    "message": "Title updated successfully",
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count
                }
        except Exception as e:
            logging.error(f"Error in set_conversation_title: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def get_conversation_title(self, workspace_id, user_id, session_id):
        """Retrieve the conversation title from the context collection.
        
        Args:
            workspace_id: The workspace identifier
            user_id: The user identifier
            session_id: The session identifier
            
        Returns:
            str or None: The conversation title if found, None otherwise
        """
        try:
            query = {
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id
            }
            context_doc = self.context_collection.find_one(query)
            if context_doc and "title" in context_doc:
                return context_doc["title"]
            return None
        except Exception as e:
            logging.error(f"Error in get_conversation_title for session {session_id}: {e}")
            return None


class UserConfigManager:
    def __init__(self, mongo_client):
        """
        Initialize UserConfigManager with a MongoDB client.
        
        Args:
            mongo_client: MongoDBSingleton instance from mongodb_singleton.py
        """
        try:
            self.config_collection = mongo_client.chatbot_db["kb_user_config"]
        except Exception as e:
            logging.error(f"Error in MongoDB connection: {e}")
            raise

    def set_config(self, workspace_id: str, user_id: str, config: dict):
        """
        update existing fields or create new fields for a user config in the workspace. 
        config fields:
       {}
        """ 
        
        # Build filter to match user and workspace
        filter = {"workspace_id": workspace_id, "user_id": user_id}
        update = {"$set": config}
        
        try:
            result = self.config_collection.update_one(filter, update, upsert=True)
            if result.upserted_id:
                return {
                    "status": "success",
                    "operation": "created",
                    "upserted_id": str(result.upserted_id)
                }
            else:
                return {
                    "status": "success",
                    "operation": "updated",
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count
                }
        except Exception as e:
            logging.error(f"Error in set_config: {e}")
            raise
    
    def get_config(self, workspace_id, user_id, fields: list = None):
        try:
            query = {"workspace_id": workspace_id, "user_id": user_id}
            config_doc = self.config_collection.find_one(query)
            if config_doc:
                # Remove MongoDB internal _id field
                config_doc.pop("_id", None)
                if fields is None:
                    return config_doc
                else:
                    return {field: config_doc.get(field) for field in fields}
            else:
                # Return structure with None values if fields are specified
                if fields is not None:
                    return {field: None for field in fields}
                else:
                    return {}
        except Exception as e:
            logging.error(f"Error in get_config: {e}")
            raise 

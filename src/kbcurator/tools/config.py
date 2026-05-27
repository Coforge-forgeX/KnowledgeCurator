from ..server.main import os
import sys

# # Add forgex root to Python path to import from packages/
# forgex_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# if forgex_root not in sys.path:
#     sys.path.insert(0, forgex_root)
from ..server import server
from datetime import datetime , timezone
import json
from ..server.server import mcp
import logging
from common_adapters.sharepoint import sharepoint_update_config
# from utils.azure_devops_update_config import azure_devops_update_config
logger = logging.getLogger("config_tool")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s@ %(name)s@ %(levelname)s @%(message)s"
    )

@mcp.tool()
async def update_config(user_id: str,workspace_id:str ,data: dict) -> dict:
    """
    Endpoint to update the configuration of a specific agent.
    user_id : str : unique identifier for the user
    workspace_id : str : unique identifier for the workspace
    conversation_id : str : unique identifier for the conversation to update its session config
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}

    try:
        data=sharepoint_update_config(data)
        if data:
            response = server.user_config_manager.set_config(workspace_id , user_id , config = data)
            updated_user_config = server.user_config_manager.get_config(workspace_id , user_id )
            sessions = server.session.get_recent_sessions_by_ttl(
                workspace_id,
                user_id,
                current_time=datetime.now(timezone.utc),
                ttl_seconds=float(os.getenv("REDIS_EXPIRY_SECONDS", "3600"))  # Default to 3600 seconds if not set
            )
            for sess in sessions:
                await server.r.store_data({f"{sess}-config": json.dumps(updated_user_config)})
                
            return response
        else:
            return {"status":"success","message":"No data provided for update"}
    except Exception as e:
        logger.info(f"Error storing updated user configuration : {e}")
        return {"status": "error", "message": "Failed to store updated user configuration", "details": str(e)}
    


@mcp.tool()
async def get_config(user_id: str , workspace_id:str , fields:list|None = None) -> dict:
    """
    Endpoint to retrieve the configuration of a specific agent.
    """
    if user_id is None:
        return {"status": "error", "error": "user_id cannot be null"}
    try:
        result = server.user_config_manager.get_config(workspace_id , user_id , fields)
        
        return {"status":"success",**result}
    except Exception as e:
        logger.info(f"Error retrieving user configuration: {e}")
        return {"status": "error", "message": "Failed to fetch user configuration", "details": str(e)}
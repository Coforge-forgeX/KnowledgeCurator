"""
Simplified Agent LLM Configuration Service

This module provides service layer functionality for managing LLM provider selection
for agents within workspaces. It handles only provider selection and current provider,
with all credentials coming from environment variables.
"""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_, text

from ..utils.db import db
import logging

logger = logging.getLogger(__name__)


class AgentLLMConfigurationService:
    """Simplified service class for managing agent LLM provider selection."""
    
    def __init__(self):
        self.db = db
    
    def get_configuration(self, workspace_id: int, agent_id: Optional[int] = None) -> Optional[dict]:
        """
        Get LLM configuration for a workspace/agent.
        
        Args:
            workspace_id: ID of the workspace
            agent_id: ID of the agent (None for workspace default)
        
        Returns:
            Configuration dictionary or None if not found
        """
        session = self.db.Session()
        try:
            # Use raw SQL since we're using automap
            query = text("""
                SELECT id, workspace_id, agent_id, configured_providers, current_provider,
                       created_at, updated_at, created_by, updated_by
                FROM agent_llm_configuration 
                WHERE workspace_id = :workspace_id 
                  AND (agent_id = :agent_id OR (:agent_id IS NULL AND agent_id IS NULL))
                  AND is_active = true
            """)
            
            result = session.execute(query, {
                'workspace_id': workspace_id, 
                'agent_id': agent_id
            }).fetchone()
            
            if result:
                return {
                    'id': result[0],
                    'workspace_id': result[1],
                    'agent_id': result[2],
                    'configured_providers': result[3] or [],
                    'current_provider': result[4],
                    'created_at': result[5],
                    'updated_at': result[6],
                    'created_by': result[7],
                    'updated_by': result[8]
                }
            
            return None
        except Exception as e:
            logger.error(f"Failed to get configuration: {e}")
            return None
        finally:
            session.close()
    
    def get_effective_configuration(self, workspace_id: int, agent_id: int) -> Optional[dict]:
        """
        Get effective configuration for an agent, falling back to workspace default.
        
        Args:
            workspace_id: ID of the workspace
            agent_id: ID of the agent
        
        Returns:
            Configuration dictionary (agent-specific or workspace default)
        """
        # First try to get agent-specific configuration
        agent_config = self.get_configuration(workspace_id, agent_id)
        if agent_config:
            return agent_config
        
        # Fall back to workspace default configuration
        workspace_config = self.get_configuration(workspace_id, None)
        return workspace_config
    
    def create_or_update_configuration(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
        configured_providers: Optional[List[str]] = None,
        current_provider: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> dict:
        """
        Create or update LLM configuration.
        
        Args:
            workspace_id: ID of the workspace
            agent_id: ID of the agent (None for workspace default)
            configured_providers: List of configured provider names
            current_provider: Set as current active provider
            user_id: ID of the user making the change
        
        Returns:
            Updated configuration dictionary
        """
        session = self.db.Session()
        try:
            # Check if configuration already exists
            existing = self.get_configuration(workspace_id, agent_id)
            
            if existing:
                # Update existing configuration
                update_query = text("""
                    UPDATE agent_llm_configuration 
                    SET configured_providers = :configured_providers,
                        current_provider = :current_provider,
                        updated_by = :user_id,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE workspace_id = :workspace_id 
                      AND (agent_id = :agent_id OR (:agent_id IS NULL AND agent_id IS NULL))
                      AND is_active = true
                    RETURNING id
                """)
                
                result = session.execute(update_query, {
                    'workspace_id': workspace_id,
                    'agent_id': agent_id,
                    'configured_providers': configured_providers or existing['configured_providers'],
                    'current_provider': current_provider or existing['current_provider'],
                    'user_id': user_id
                }).fetchone()
                
                config_id = result[0] if result else existing['id']
                
            else:
                # Use UPSERT to handle race conditions
                upsert_query = text("""
                    INSERT INTO agent_llm_configuration 
                    (workspace_id, agent_id, configured_providers, current_provider, created_by, updated_by)
                    VALUES (:workspace_id, :agent_id, :configured_providers, :current_provider, :user_id, :user_id)
                    ON CONFLICT (workspace_id, agent_id)
                    DO UPDATE SET
                        configured_providers = EXCLUDED.configured_providers,
                        current_provider = EXCLUDED.current_provider,
                        updated_by = EXCLUDED.updated_by,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """)
                
                result = session.execute(upsert_query, {
                    'workspace_id': workspace_id,
                    'agent_id': agent_id,
                    'configured_providers': configured_providers or [],
                    'current_provider': current_provider,
                    'user_id': user_id
                }).fetchone()
                
                config_id = result[0]
            
            session.commit()
            
            # Return updated configuration
            updated_config = self.get_configuration(workspace_id, agent_id)
            
            logger.info(f"LLM configuration {'updated' if existing else 'created'} for workspace {workspace_id}, agent {agent_id}")
            
            return updated_config
            
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Failed to create/update LLM configuration: {e}")
            raise ValueError(f"Invalid workspace_id or agent_id: {e}")
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in create_or_update_configuration: {e}")
            raise
        finally:
            session.close()
    
    def switch_provider(
        self,
        workspace_id: int,
        provider: str,
        agent_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> dict:
        """
        Switch the current LLM provider for a workspace/agent.
        
        Args:
            workspace_id: ID of the workspace
            provider: Provider name to switch to
            agent_id: ID of the agent (None for workspace default)
            user_id: ID of the user making the change
        
        Returns:
            Updated configuration dictionary
        """
        valid_providers = ['azure', 'openai', 'gcp', 'quasar', 'aws_bedrock']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}")
        
        # Get current configuration
        config = self.get_configuration(workspace_id, agent_id)
        
        if config:
            # Add provider to configured_providers if not already there
            configured_providers = list(config['configured_providers']) if config['configured_providers'] else []
            if provider not in configured_providers:
                configured_providers.append(provider)
            
            return self.create_or_update_configuration(
                workspace_id=workspace_id,
                agent_id=agent_id,
                configured_providers=configured_providers,
                current_provider=provider,
                user_id=user_id
            )
        else:
            # Create new configuration
            return self.create_or_update_configuration(
                workspace_id=workspace_id,
                agent_id=agent_id,
                configured_providers=[provider],
                current_provider=provider,
                user_id=user_id
            )
    
    def add_provider(
        self,
        workspace_id: int,
        provider: str,
        agent_id: Optional[int] = None,
        set_as_current: bool = False,
        user_id: Optional[int] = None
    ) -> dict:
        """
        Add a provider to the configured providers list.
        
        Args:
            workspace_id: ID of the workspace
            provider: Provider name to add
            agent_id: ID of the agent (None for workspace default)
            set_as_current: Whether to set as current provider
            user_id: ID of the user making the change
        
        Returns:
            Updated configuration dictionary
        """
        valid_providers = ['azure', 'openai', 'gcp', 'quasar', 'aws_bedrock']
        if provider not in valid_providers:
            raise ValueError(f"Invalid provider: {provider}")
        
        # Get current configuration
        config = self.get_configuration(workspace_id, agent_id)
        
        if config:
            # Add provider to configured_providers if not already there
            configured_providers = list(config['configured_providers']) if config['configured_providers'] else []
            if provider not in configured_providers:
                configured_providers.append(provider)
            
            current_provider = provider if set_as_current else config['current_provider']
            
            return self.create_or_update_configuration(
                workspace_id=workspace_id,
                agent_id=agent_id,
                configured_providers=configured_providers,
                current_provider=current_provider,
                user_id=user_id
            )
        else:
            # Create new configuration
            return self.create_or_update_configuration(
                workspace_id=workspace_id,
                agent_id=agent_id,
                configured_providers=[provider],
                current_provider=provider if set_as_current else None,
                user_id=user_id
            )
    
    def get_current_provider(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Get the current active provider for a workspace/agent.
        
        Args:
            workspace_id: ID of the workspace
            agent_id: ID of the agent (None for workspace default)
        
        Returns:
            Current provider name or None
        """
        config = self.get_effective_configuration(workspace_id, agent_id) if agent_id else self.get_configuration(workspace_id, agent_id)
        
        if config:
            return config['current_provider']
        return None
    
    def list_configured_providers(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None
    ) -> List[str]:
        """
        List all configured providers for a workspace/agent.
        
        Args:
            workspace_id: ID of the workspace
            agent_id: ID of the agent (None for workspace default)
        
        Returns:
            List of configured provider names
        """
        config = self.get_effective_configuration(workspace_id, agent_id) if agent_id else self.get_configuration(workspace_id, agent_id)
        
        if config and config['configured_providers']:
            return list(config['configured_providers'])
        return []
    
    def delete_configuration(
        self,
        workspace_id: int,
        agent_id: Optional[int] = None,
        user_id: Optional[int] = None
    ) -> bool:
        """
        Soft delete LLM configuration.
        
        Args:
            workspace_id: ID of the workspace
            agent_id: ID of the agent (None for workspace default)
            user_id: ID of the user making the change
        
        Returns:
            True if configuration was deleted, False if not found
        """
        session = self.db.Session()
        try:
            delete_query = text("""
                UPDATE agent_llm_configuration 
                SET is_active = false,
                    updated_by = :user_id,
                    updated_at = CURRENT_TIMESTAMP
                WHERE workspace_id = :workspace_id 
                  AND (agent_id = :agent_id OR (:agent_id IS NULL AND agent_id IS NULL))
                  AND is_active = true
                RETURNING id
            """)
            
            result = session.execute(delete_query, {
                'workspace_id': workspace_id,
                'agent_id': agent_id,
                'user_id': user_id
            }).fetchone()
            
            if result:
                session.commit()
                logger.info(f"LLM configuration deleted for workspace {workspace_id}, agent {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete configuration: {e}")
            raise
        finally:
            session.close()
    
    def bulk_create_agent_configurations(
        self,
        workspace_id: int,
        agent_ids: List[int],
        configured_providers: Optional[List[str]] = None,
        current_provider: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> List[dict]:
        """
        Bulk create LLM configurations for multiple agents in a workspace.
        
        Args:
            workspace_id: ID of the workspace
            agent_ids: List of agent IDs to create configurations for
            configured_providers: List of configured provider names (default: ['azure'])
            current_provider: Set as current active provider (default: 'azure')
            user_id: ID of the user making the change
        
        Returns:
            List of created configuration dictionaries
        """
        if not agent_ids:
            return []
            
        configured_providers = configured_providers or ['azure']
        current_provider = current_provider or 'azure'
        
        session = self.db.Session()
        created_configs = []
        
        try:
            for agent_id in agent_ids:
                # Check if configuration already exists
                existing = self.get_configuration(workspace_id, agent_id)
                
                if not existing:
                    # Use UPSERT to handle race conditions
                    upsert_query = text("""
                        INSERT INTO agent_llm_configuration 
                        (workspace_id, agent_id, configured_providers, current_provider, created_by, updated_by)
                        VALUES (:workspace_id, :agent_id, :configured_providers, :current_provider, :user_id, :user_id)
                        ON CONFLICT (workspace_id, agent_id)
                        DO UPDATE SET
                            configured_providers = EXCLUDED.configured_providers,
                            current_provider = EXCLUDED.current_provider,
                            updated_by = EXCLUDED.updated_by,
                            updated_at = CURRENT_TIMESTAMP
                        RETURNING id
                    """)
                    
                    result = session.execute(upsert_query, {
                        'workspace_id': workspace_id,
                        'agent_id': agent_id,
                        'configured_providers': configured_providers,
                        'current_provider': current_provider,
                        'user_id': user_id
                    }).fetchone()
                    
                    if result:
                        created_configs.append({
                            'id': result[0],
                            'workspace_id': workspace_id,
                            'agent_id': agent_id,
                            'configured_providers': configured_providers,
                            'current_provider': current_provider
                        })
                        
                        logger.info(f"Created LLM configuration for workspace {workspace_id}, agent {agent_id}")
            
            session.commit()
            return created_configs
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to bulk create agent LLM configurations: {e}")
            raise
        finally:
            session.close()

    def get_workspace_configurations(self, workspace_id: int) -> List[dict]:
        """
        Get all LLM configurations for a workspace (both default and agent-specific).
        
        Args:
            workspace_id: ID of the workspace
        
        Returns:
            List of configuration dictionaries
        """
        session = self.db.Session()
        try:
            query = text("""
                SELECT id, workspace_id, agent_id, configured_providers, current_provider,
                       created_at, updated_at, created_by, updated_by
                FROM agent_llm_configuration 
                WHERE workspace_id = :workspace_id AND is_active = true
                ORDER BY agent_id NULLS FIRST
            """)
            
            results = session.execute(query, {'workspace_id': workspace_id}).fetchall()
            
            configs = []
            for result in results:
                configs.append({
                    'id': result[0],
                    'workspace_id': result[1],
                    'agent_id': result[2],
                    'configured_providers': result[3] or [],
                    'current_provider': result[4],
                    'created_at': result[5],
                    'updated_at': result[6],
                    'created_by': result[7],
                    'updated_by': result[8]
                })
            
            return configs
        except Exception as e:
            logger.error(f"Failed to get workspace configurations: {e}")
            return []
        finally:
            session.close()
    
    def delete_workspace_configurations(self, workspace_id: int, user_id: Optional[int] = None) -> int:
        """
        Delete all LLM configurations for a workspace (soft delete).
        
        Args:
            workspace_id: ID of the workspace
            user_id: ID of the user making the change
        
        Returns:
            Number of configurations deleted
        """
        session = self.db.Session()
        try:
            delete_query = text("""
                UPDATE agent_llm_configuration 
                SET is_active = false,
                    updated_by = :user_id,
                    updated_at = CURRENT_TIMESTAMP
                WHERE workspace_id = :workspace_id 
                  AND is_active = true
                RETURNING id
            """)
            
            results = session.execute(delete_query, {
                'workspace_id': workspace_id,
                'user_id': user_id
            }).fetchall()
            
            deleted_count = len(results)
            
            if deleted_count > 0:
                session.commit()
                logger.info(f"Deleted {deleted_count} LLM configurations for workspace {workspace_id}")
            
            return deleted_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete workspace configurations: {e}")
            raise
        finally:
            session.close()


# Global service instance
agent_llm_config_service = AgentLLMConfigurationService()
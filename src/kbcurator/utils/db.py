from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from urllib.parse import quote_plus
from threading import RLock
from .config import settings

class Database:
	"""
	Singleton-style DB manager for SQLAlchemy engine, session, and automapped tables.
	Usage: db = Database(); db.Session(); db.Base; db.<TableName>
	"""
	_instance = None
	_lock = RLock()

	def __new__(cls):
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = super().__new__(cls)
					cls._instance._init_db()
		return cls._instance

	def _init_db(self):
		# Build connection string from config
		conn_str = (
			f"postgresql+psycopg2://{settings.POSTGRES_USER}:{quote_plus(settings.POSTGRES_PASSWORD)}"
			f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
		)
		
		self.engine = create_engine(conn_str)
		self.Session = sessionmaker(bind=self.engine)
		self.metadata = MetaData()
		self.metadata.reflect(self.engine)
		self.Base = automap_base(metadata=self.metadata)
		self.Base.prepare()
		
		# Table/class mappings
		self.AgentIndustryMap = self.Base.classes.agent_industry_mapping
		self.AgentRegionMap = self.Base.classes.agent_region_mapping
		self.AgentSubIndustryMap = self.Base.classes.agent_subindustry_mapping
		self.AgentIntentMap = self.Base.classes.agent_intent_mapping
		self.ToolIndustryMap = self.Base.classes.tool_industry_mapping
		self.ToolRegionMap = self.Base.classes.tool_region_mapping
		self.ToolIntentMap = self.Base.classes.tool_intent_mapping
		self.Workspace = self.Base.classes.workspace_master
		self.AgentMap = self.Base.classes.workspace_agents_mapping_2
		self.ToolMap = self.Base.classes.workspace_tools_mapping
		self.UserMap = self.Base.classes.workspace_users_mapping
		self.Agent = self.Base.classes.agents_details
		self.Tool = self.Base.classes.tools_details
		self.User = self.Base.classes.users
		self.Category = self.Base.classes.category_master
		self.Industry = self.Base.classes.industry_master
		self.SubIndustry = self.Base.classes.subindustry_master
		self.AgentsCMS = self.Base.classes.agents_cms
		self.ToolsCMS = self.Base.classes.tool_cms
		self.Integrations = self.Base.classes.integrations
		self.Intent = self.Base.classes.intent_master
		self.KnowledgeBase = self.Base.classes.knowledge_base_master
		self.AgentCMSIntegrationMap = self.Base.classes.agent_cms_integration_mapping
		self.FavouriteMappingAgent = self.Base.classes.favourite_mapping_agent
		self.FavouriteMappingTool = self.Base.classes.favourite_mapping_tool
		self.WorkspaceIndustrySubIndustryMap = self.Base.classes.workspace_industry_intent_mapping
		self.Role = self.Base.classes.role_master
		self.UserRoleMap = self.Base.classes.user_role_mapping
		self.TMUIntegrationMapping = self.Base.classes.tool_workspace_user_integration_mapping
		self.AMUIntegrationMapping = self.Base.classes.agent_workspace_user_integration_mapping
		
		# Agent LLM Configuration table (new)
		self.AgentLLMConfiguration = getattr(self.Base.classes, 'agent_llm_configuration', None)
		
		# Optionals
		self.ToolSubIndustryMap = getattr(self.Base.classes, 'tool_subindustry_mapping', None)
		self.ToolCMSIntegrationMap = getattr(self.Base.classes, 'tool_cms_integration_mapping', None)
		self.WorkspaceRegionMap = getattr(self.Base.classes, 'workspace_region_mapping', None)
		self.WorkspaceIntentMap = getattr(self.Base.classes, 'workspace_intent_mapping', None)
		self.WorkspaceKeywordMap = getattr(self.Base.classes, 'workspace_keyword_mapping', None)

# Usage: from .db import db; session = db.Session(); db.Workspace

db = Database()
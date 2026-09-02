from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from urllib.parse import quote_plus
from threading import RLock
from .config import settings


def _require_table(base_classes, table_name):
	table = getattr(base_classes, table_name, None)
	if table is None:
		available = sorted(name for name in dir(base_classes) if not name.startswith('_'))
		raise RuntimeError(
			f"Required table '{table_name}' was not reflected from PostgreSQL. "
			f"Available reflected tables: {available}"
		)
	return table

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
		
		# Configure connection pool with health checks and recycling
		self.engine = create_engine(
			conn_str,
			pool_pre_ping=True,  # Verify connections are alive before using them
			pool_size=10,  # Maximum number of connections to keep open
			max_overflow=20,  # Additional connections when pool is exhausted
			pool_recycle=3600,  # Recycle connections after 1 hour
			pool_timeout=30,  # Timeout for getting a connection from the pool
			echo=False  # Set to True for SQL query logging
		)
		self.Session = sessionmaker(bind=self.engine)
		self.metadata = MetaData()
		self.metadata.reflect(self.engine)
		self.Base = automap_base(metadata=self.metadata)
		self.Base.prepare()
		
		# Table/class mappings
		# Dropped from the DB as unused redundant tables; kept optional in case they're recreated.
		self.AgentIndustryMap = getattr(self.Base.classes, 'agent_industry_mapping', None)
		self.AgentRegionMap = getattr(self.Base.classes, 'agent_region_mapping', None)
		self.AgentSubIndustryMap = getattr(self.Base.classes, 'agent_subindustry_mapping', None)
		self.AgentIntentMap = self.Base.classes.agent_intent_mapping
		self.ToolIndustryMap = getattr(self.Base.classes, 'tool_industry_mapping', None)
		self.ToolRegionMap = getattr(self.Base.classes, 'tool_region_mapping', None)
		self.ToolIntentMap = self.Base.classes.tool_intent_mapping
		self.Workspace = _require_table(self.Base.classes, 'workspace_master')
		self.AgentMap = _require_table(self.Base.classes, 'workspace_agents_mapping_2')
		self.ToolMap = _require_table(self.Base.classes, 'workspace_tools_mapping')
		self.UserMap = _require_table(self.Base.classes, 'workspace_users_mapping')
		self.Agent = _require_table(self.Base.classes, 'agents_details')
		self.Tool = _require_table(self.Base.classes, 'tools_details')
		self.User = _require_table(self.Base.classes, 'users')
		self.Category = _require_table(self.Base.classes, 'category_master')
		self.Industry = _require_table(self.Base.classes, 'industry_master')
		self.SubIndustry = _require_table(self.Base.classes, 'subindustry_master')
		self.AgentsCMS = _require_table(self.Base.classes, 'agents_cms')
		self.ToolsCMS = _require_table(self.Base.classes, 'tool_cms')
		self.Integrations = _require_table(self.Base.classes, 'integrations')
		self.Intent = _require_table(self.Base.classes, 'intent_master')
		self.KnowledgeBase = _require_table(self.Base.classes, 'knowledge_base_master')
		self.AgentCMSIntegrationMap = _require_table(self.Base.classes, 'agent_cms_integration_mapping')
		self.FavouriteMappingAgent = _require_table(self.Base.classes, 'favourite_mapping_agent')
		self.FavouriteMappingTool = _require_table(self.Base.classes, 'favourite_mapping_tool')
		self.WorkspaceIndustrySubIndustryMap = _require_table(self.Base.classes, 'workspace_industry_intent_mapping')
		# Not yet created in the DB; code treats these as optional (falsy checks before use).
		self.TMUIntegrationMapping = getattr(self.Base.classes, 'tool_workspace_user_integration_mapping', None)
		self.AMUIntegrationMapping = getattr(self.Base.classes, 'agent_workspace_user_integration_mapping', None)
		self.Role = _require_table(self.Base.classes, 'role_master')
		self.UserRoleMap = _require_table(self.Base.classes, 'user_role_mapping')
		# Dropped as unused redundant tables; kept optional in case they're recreated.
		self.RoleAgentMap = getattr(self.Base.classes, 'role_agent_mapping', None)
		self.RoleToolMap = getattr(self.Base.classes, 'role_tool_mapping', None)
		self.RolePermissionMap = getattr(self.Base.classes, 'role_permission_mapping', None)
		self.Permission = getattr(self.Base.classes, 'permissions', None)
		self.ClientMaster = getattr(self.Base.classes, 'client_master', None)
		self.RegionMaster = getattr(self.Base.classes, 'region_master', None)
		self.WorkspaceUsersRoleMap = getattr(self.Base.classes, 'workspace_users_role_mapping', None)
		self.WorkspaceAgentsMap = getattr(self.Base.classes, 'workspace_agents_mapping', None)

		# Optionals
		self.ToolSubIndustryMap = getattr(self.Base.classes, 'tool_subindustry_mapping', None)
		self.ToolCMSIntegrationMap = getattr(self.Base.classes, 'tool_cms_integration_mapping', None)
		self.WorkspaceRegionMap = getattr(self.Base.classes, 'workspace_region_mapping', None)
		self.WorkspaceIntentMap = getattr(self.Base.classes, 'workspace_intent_mapping', None)
		self.WorkspaceKeywordMap = getattr(self.Base.classes, 'workspace_keyword_mapping', None)

# Usage: from .db import db; session = db.Session(); db.Workspace

db = Database()

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlatformRole(str, Enum):
    ADMIN = "Forge-X Admin"
    USER = "User"

class WorkspaceRole(str, Enum):
    WS_ADMIN = "Workspace Admin"
    WS_MANAGER = "Workspace Manager"

class SdlcRoles(str, Enum):
    PRODUCT_OWNER = "Product Owner"
    DEVELOPER = "Developer"
    ARCHITECT = "Architect"
    DATA_ENGINEER = "Data Engineer"
    SME = "Subject Matter Expert"
    KNOWLEDGE_CURATOR = "Knowledge Curator"
    BUSINESS_ANALYST = "Business Analyst"
    TESTER = "Tester"
    SUPPORT = "Support"

class WorkspaceType(str, Enum):
    KG = "Knowledge Graph"
    DM = "Demo"
    TR = "Trial"
    PR = "Product"

class DefaultValue(str, Enum):
    PASSWORD = "forge-X@coforge"
    EMAIL_ENDS_WITH_COFORGE = "@coforge.com"

@dataclass(frozen=True)
class RoleData:
    name: str
    id: int

class Role(Enum):
    ADMIN = RoleData(PlatformRole.ADMIN.value, 0)
    USER = RoleData(PlatformRole.USER.value, 1)
    WS_ADMIN = RoleData(WorkspaceRole.WS_ADMIN.value, 3)
    WS_MANAGER = RoleData(WorkspaceRole.WS_MANAGER.value, 11)
    PRODUCT_OWNER = RoleData(SdlcRoles.PRODUCT_OWNER.value, 10)
    DEVELOPER = RoleData(SdlcRoles.DEVELOPER.value, 5)
    ARCHITECT = RoleData(SdlcRoles.ARCHITECT.value, 7)
    DATA_ENGINEER = RoleData(SdlcRoles.DATA_ENGINEER.value, 9)
    SME = RoleData(SdlcRoles.SME.value, 34)
    KNOWLEDGE_CURATOR = RoleData(SdlcRoles.KNOWLEDGE_CURATOR.value, 4)
    BUSINESS_ANALYST = RoleData(SdlcRoles.BUSINESS_ANALYST.value, 35)
    TESTER = RoleData(SdlcRoles.TESTER.value, 6)
    SUPPORT = RoleData(SdlcRoles.SUPPORT.value, 8)

    
    @classmethod
    def get_by_id(cls, id) -> "Role | None":
        return next((r for r in cls if r.value.id == id), None)
    
    
    @classmethod
    def get_by_name(cls, name: str) -> "Role | None":
        return next((r for r in cls if r.value.name == name), None)

    
    @property
    def id(self):
        return self.value.id

    @property
    def name(self):
        return self.value.name
    
    
    @classmethod
    def _validate_unique_ids(cls):
        ids = [r.value.id for r in cls]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate role IDs detected")


# Call once after class definition
Role._validate_unique_ids()






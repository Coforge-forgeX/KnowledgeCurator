"""
RAG Domain Models

Clean domain models following Domain-Driven Design principles.
No infrastructure dependencies - pure business logic.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class QueryMode(str, Enum):
    """RAG query execution modes"""
    NAIVE = "naive"
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    MIX = "mix"  # Legacy compatibility


@dataclass(frozen=True)
class QueryContext:
    """Immutable query context for RAG operations"""
    query: str
    workspace_id: int
    role_id: int
    mode: QueryMode = QueryMode.HYBRID
    history: List[Dict[str, str]] = field(default_factory=list)
    only_context: bool = False
    agent_id: Optional[int] = None


@dataclass(frozen=True)
class KnowledgeBase:
    """Knowledge base configuration"""
    domain: str
    name: str
    workspace_id: Optional[int] = None

    @property
    def full_name(self) -> str:
        """Get fully qualified KB name"""
        return f"{self.domain}/{self.name}"


@dataclass
class RetrievedChunk:
    """Retrieved document chunk from RAG"""
    chunk_id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class DocumentReference:
    """Parsed document reference from RAG response"""
    citation_number: str  # e.g., "[1]"
    file_path: str
    file_name: str

    def __post_init__(self):
        """Validate reference after initialization"""
        if not self.citation_number.startswith('['):
            raise ValueError(f"Invalid citation format: {self.citation_number}")
        if not self.file_path:
            raise ValueError("file_path cannot be empty")


@dataclass
class EnrichedSource:
    """Document source with download URL"""
    file_name: str
    download_url: str
    container_name: str
    blob_path: str
    download_name: str
    citation: Optional[str] = None


@dataclass
class RAGQueryResult:
    """Complete RAG query result"""
    answer: str
    sources: List[EnrichedSource] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)

    def has_sources(self) -> bool:
        """Check if result has any sources"""
        return len(self.sources) > 0

    def has_chunks(self) -> bool:
        """Check if result has retrieved chunks"""
        return len(self.retrieved_chunks) > 0


@dataclass
class MultiKBResult:
    """Result from querying multiple knowledge bases"""
    kb_results: Dict[str, RAGQueryResult]
    aggregated_answer: str
    sources: List[EnrichedSource] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    def successful_count(self) -> int:
        """Count of successful KB queries"""
        return len(self.kb_results) - len(self.errors)

    def has_errors(self) -> bool:
        """Check if any KB queries failed"""
        return len(self.errors) > 0

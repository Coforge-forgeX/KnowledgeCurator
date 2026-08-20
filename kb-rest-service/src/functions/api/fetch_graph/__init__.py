"""
Fetch Graph API Endpoint

Fetches filtered graph data relevant to a query and answer.
Uses LLM to validate which nodes are related to the answer.
"""
import json
import time
import hashlib
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.lightrag_service import get_lightrag_service
from src.core.logging import get_logger
from src.core.redis import redis_manager
from src.functions.api.fetch_graph.payloads import (
    FetchGraphRequest,
    FilteredGraphDataModel,
    GraphEdgeModel,
    GraphNodeModel,
    GraphRelationshipModel,
)
from src.helpers.graph_parser import parse_graph_context
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.helpers.workspace_resolver import WorkspaceResolver
from src.services.workspace_service import get_workspace_service
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from shared.lightrag import build_azure_openai_chat_completion_func

try:
    from common_adapters.configurableAI.llm_router_config_store import (
        llm_router_config_store,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    llm_router_config_store = None

logger = get_logger(__name__)

# Cache TTL for filtered graph data (30 minutes)
GRAPH_CACHE_TTL_SECONDS = 30 * 60


def _normalize_text(text: str) -> str:
    """Normalize text for consistent cache keys."""
    return " ".join((text or "").strip().split())


def _make_graph_cache_key(
    workspace_id: int,
    query: str,
    answer: str,
    mode: str,
    graph_only: bool = False,
) -> str:
    """Create deterministic Redis key for cached graph data."""
    normalized_query = _normalize_text(query)
    normalized_answer = _normalize_text(answer)
    normalized_mode = (mode or "").strip().lower()
    key_material = f"{workspace_id}|{normalized_mode}|{normalized_query}|{normalized_answer}|graph_only={int(graph_only)}"
    query_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    return f"graph_filtered:{workspace_id}:{query_hash}"


def _make_query_evidence_key(workspace_id: int, query: str, mode: str) -> str:
    """Create query evidence cache key compatible with query_rag endpoint."""
    normalized_query = _normalize_text(query)
    normalized_mode = (mode or "").strip().lower()
    key_material = f"{workspace_id}|{normalized_mode}|{normalized_query}"
    query_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    return f"query_evidence:{workspace_id}:{query_hash}"


def _normalize_mode(mode: str) -> str:
    """Normalize query mode for LightRAG usage."""
    normalized = (mode or "").strip().lower()
    return "mix" if normalized == "hybrid" else normalized


def _has_non_empty_graph_content(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> bool:
    """Return True when at least one graph section contains data."""
    return bool(entities) or bool(relationships)


def _resolve_additional_kb_paths(kb_name: str, additional_kbs: Optional[List[str]]) -> List[str]:
    """Resolve additional KB paths similarly to existing query strategy behavior."""
    normalized_primary = (kb_name or "").strip().strip("/")
    primary_parts = [part for part in normalized_primary.split("/") if part]
    subindustry = primary_parts[0] if primary_parts else ""

    resolved: List[str] = []
    seen: set[str] = set()

    for kb_entry in additional_kbs or []:
        title = (kb_entry or "").strip().strip("/")
        if not title:
            continue
        candidate = title if "/" in title else (f"{subindustry}/{title}" if subindustry else title)
        dedupe_key = candidate.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        resolved.append(candidate)

    return resolved


def _extract_graph_context_item(item: Any) -> Optional[Dict[str, Any]]:
    """Convert a raw LightRAG context item into structured graph payload."""
    if isinstance(item, dict):
        entities = item.get("entities")
        relationships = item.get("relationships") or item.get("relationship")
        if isinstance(entities, list) or isinstance(relationships, list):
            return {
                "entities": entities if isinstance(entities, list) else [],
                "relationships": relationships if isinstance(relationships, list) else [],
                "metadata": item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
            }

        content = item.get("content") or item.get("text")
        if isinstance(content, str) and "Knowledge Graph Data" in content:
            parsed = parse_graph_context(content)
            return {
                "entities": parsed.get("entities", []),
                "relationships": parsed.get("relationships", []),
                "metadata": parsed.get("metadata", {}),
            }
        return None

    if isinstance(item, str) and "Knowledge Graph Data" in item:
        parsed = parse_graph_context(item)
        return {
            "entities": parsed.get("entities", []),
            "relationships": parsed.get("relationships", []),
            "metadata": parsed.get("metadata", {}),
        }

    return None


def _get_cached_query_evidence(
    workspace_id: int,
    query: str,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """Get query_rag cached evidence payload by workspace/query/mode."""
    if not redis_manager.is_available or not getattr(settings.cache, "REDIS_ENABLED", True):
        return None

    evidence_key = _make_query_evidence_key(workspace_id, query, mode)
    cached_data = redis_manager.get(evidence_key)
    if not cached_data:
        return None

    try:
        parsed = json.loads(cached_data)
        if not isinstance(parsed, dict):
            return None
        return {
            "kb_results": parsed.get("kb_results", []),
            "metadata": {
                "source": "query_evidence_cache",
                "requested_mode": parsed.get("requested_mode"),
            },
        }
    except Exception as cache_error:
        logger.warning("Failed to parse query evidence cache", error=str(cache_error), key=evidence_key)
        return None


async def _fetch_context_from_lightrag(
    workspace_id: int,
    role_id: int,
    domain: str,
    kb_name: str,
    query: str,
    mode: str,
    additional_kbs: Optional[List[str]],
    is_kg: Optional[bool],
    agent_id: Optional[int],
) -> Dict[str, Any]:
    """Fetch structured graph data from LightRAG without answer generation."""
    del role_id  # role_id is validated upstream, not needed for retrieval calls

    resolved_mode = _normalize_mode(mode)

    kb_targets: List[str] = [kb_name]
    if not is_kg:
        kb_targets.extend(_resolve_additional_kb_paths(kb_name, additional_kbs))

    kb_results: Dict[str, Any] = {}
    for kb_target in kb_targets:
        target_workspace = WorkspaceResolver.build_workspace_name(domain, kb_target)
        lightrag = get_lightrag_service(workspace=target_workspace)
        lightrag.working_dir = settings.lightrag.LIGHTRAG_WORKING_DIR
        lightrag.set_runtime_context(workspace_id=workspace_id, agent_id=agent_id)

        data_response = await lightrag.query_data(
            query=query,
            mode=resolved_mode,
            top_k=20,
            chunk_top_k=5,
            stream=False,
        )

        data_payload = data_response.get("data", {}) if isinstance(data_response, dict) else {}
        entities_payload = data_payload.get("entities", []) if isinstance(data_payload.get("entities"), list) else []
        relationships_payload = data_payload.get("relationships", []) if isinstance(data_payload.get("relationships"), list) else []

        # Retrieve true Neo4j KnowledgeGraph subgraphs (with element_id) for candidate entity labels
        graph_entities: List[Dict[str, Any]] = []
        graph_relationships: List[Dict[str, Any]] = []

        candidate_labels = [
            _extract_entity_name(e)
            for e in entities_payload
            if isinstance(e, dict) and _extract_entity_name(e)
        ]

        if candidate_labels:
            semaphore = asyncio.Semaphore(4)

            async def _fetch_kg_for_label(lbl: str):
                async with semaphore:
                    try:
                        return await lightrag.get_knowledge_graph(node_label=lbl, max_depth=2)
                    except Exception as kg_err:
                        logger.warning(f"Failed to fetch knowledge graph for label '{lbl}'", error=kg_err)
                        return None

            kg_batch_results = await asyncio.gather(
                *[_fetch_kg_for_label(lbl) for lbl in candidate_labels],
                return_exceptions=True,
            )

            seen_nodes: set[str] = set()
            seen_edges: set[str] = set()

            for res in kg_batch_results:
                if isinstance(res, dict):
                    nodes_list = res.get("nodes", [])
                    edges_list = res.get("edges", [])
                    if isinstance(nodes_list, list):
                        for n in nodes_list:
                            n_id = str(n.get("element_id") or n.get("id") or _extract_entity_name(n))
                            if n_id not in seen_nodes:
                                seen_nodes.add(n_id)
                                graph_entities.append(n)
                    if isinstance(edges_list, list):
                        for e in edges_list:
                            e_id = str(e.get("element_id") or e.get("id") or f"{e.get('source')}-{e.get('target')}")
                            if e_id not in seen_edges:
                                seen_edges.add(e_id)
                                graph_relationships.append(e)

        final_entities = graph_entities if graph_entities else entities_payload
        final_relationships = graph_relationships if graph_relationships else relationships_payload

        parsed_context: List[Dict[str, Any]] = [
            {
                "entities": final_entities,
                "relationships": final_relationships,
                "metadata": data_response.get("metadata", {}) if isinstance(data_response, dict) else {},
            }
        ]

        kb_results[f"{domain}/{kb_target}"] = {
            "_raw_context": parsed_context,
        }

    return {
        "kb_results": kb_results,
        "metadata": {
            "source": "lightrag_query_data",
            "mode": resolved_mode,
        },
    }


def _build_llm_filter_func(workspace_id: int, agent_id: Optional[int]):
    """Build LLM callable using common_adapters workspace/agent routing with safe fallback."""
    if llm_router_config_store is not None:
        try:
            effective = llm_router_config_store.get_effective_configuration(workspace_id, agent_id or 1)
            current_provider = (effective or {}).get("current_provider", "").strip().lower()
            current_model = (effective or {}).get("current_model")
            if current_provider:
                provider_config = llm_router_config_store.build_config_dict(
                    workspace_id,
                    current_provider,
                    model_override=current_model,
                )
                if isinstance(provider_config, dict) and (provider_config.get("provider_name") or "").strip().lower() == "azure":
                    api_key = provider_config.get("api_key")
                    api_base = provider_config.get("endpoint")
                    api_version = provider_config.get("api_version") or settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION
                    deployment = provider_config.get("deployment_name") or provider_config.get("model")
                    if all([api_key, api_base, deployment]):
                        logger.info(
                            "Using common_adapters LLM config for graph filtering",
                            workspace_id=workspace_id,
                            agent_id=agent_id,
                            provider=current_provider,
                            model=provider_config.get("model") or deployment,
                        )
                        return build_azure_openai_chat_completion_func(
                            api_key=str(api_key or ""),
                            api_base=str(api_base or ""),
                            api_version=str(api_version or ""),
                            deployment=str(deployment or ""),
                        )
        except Exception as route_error:
            logger.warning(
                "Failed to resolve common_adapters LLM route for graph filtering",
                workspace_id=workspace_id,
                agent_id=agent_id,
                error=route_error,
            )

    api_key = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_KEY
    api_base = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_BASE or getattr(settings.lightrag, "AZURE_OPENAI_LLM_MODEL_ENDPOINT", None)
    api_version = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION
    deployment = settings.lightrag.AZURE_OPENAI_LLM_MODEL_LLM_MODEL or getattr(settings.lightrag, "AZURE_OPENAI_LLM_MODEL_NAME", None)

    if all([api_key, api_base, deployment]):
        logger.info(
            "Using fallback Azure LLM config for graph filtering",
            workspace_id=workspace_id,
            agent_id=agent_id,
            model=deployment,
        )
        return build_azure_openai_chat_completion_func(
            api_key=str(api_key or ""),
            api_base=str(api_base or ""),
            api_version=str(api_version or ""),
            deployment=str(deployment or ""),
        )

    raise ValidationException(message="No LLM configuration found for graph filtering")


def _parse_llm_json(response_text: str) -> Dict[str, Any]:
    """Parse JSON payload from LLM response text."""
    text = (response_text or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        payload = json.loads(match.group(0))

    return payload if isinstance(payload, dict) else {}


def _parse_relevant_indices(response_text: str) -> List[int]:
    """Parse relevant indices from LLM JSON response text."""
    payload = _parse_llm_json(response_text)

    indices = payload.get("relevant_indices", [])
    if not isinstance(indices, list):
        return []

    normalized_indices: List[int] = []
    for idx in indices:
        try:
            normalized_indices.append(int(idx))
        except (TypeError, ValueError):
            continue
    return normalized_indices


def _parse_relevant_entity_ids(response_text: str) -> List[str]:
    """Parse relevant entity IDs from LLM JSON response text."""
    payload = _parse_llm_json(response_text)

    for key in ("relevant_entity_ids", "entity_ids", "relevant_ids"):
        value = payload.get(key)
        if isinstance(value, list):
            ids = [str(item).strip() for item in value if str(item).strip()]
            if ids:
                return ids
    return []


def _canonical_label(value: Any) -> str:
    """Normalize labels for resilient graph entity/relationship matching."""
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    return " ".join(text.lower().split())


def _extract_entity_name(entity: Dict[str, Any], idx: Optional[int] = None) -> str:
    """Extract entity label from heterogeneous payload shapes."""
    for key in ("entity_name", "name", "entity", "label", "title", "id"):
        value = entity.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    for key in ("entity1", "entity2", "source", "target"):
        value = entity.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    # As a last resort, infer name from a JSON-ish description block.
    raw_description = entity.get("description")
    if isinstance(raw_description, str):
        for line in raw_description.splitlines():
            line_text = line.strip()
            if not line_text:
                continue
            try:
                payload = json.loads(line_text)
                if isinstance(payload, dict):
                    for key in ("Entity", "entity", "Name", "name"):
                        value = payload.get(key)
                        if value is not None and str(value).strip():
                            return str(value).strip()
            except json.JSONDecodeError:
                continue

    if idx is not None:
        return f"Entity_{idx}"
    return ""


def _extract_entity_type(entity: Dict[str, Any]) -> str:
    """Extract entity type from heterogeneous payload shapes."""
    for key in ("entity_type", "type", "category", "label_type"):
        value = entity.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "Unknown"


def _split_source_ids(value: Any) -> List[str]:
    """Split source id payloads into normalized tokens."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    normalized = (
        text.replace("<SEP>", ",")
        .replace(";", ",")
        .replace("|", ",")
    )
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _extract_relationship_label(relationship: Dict[str, Any]) -> str:
    """Extract relationship label from heterogeneous payload shapes."""
    for key in ("relation", "relationship", "edge", "predicate", "type", "label"):
        value = relationship.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _extract_entity_id(entity: Dict[str, Any]) -> Any:
    """Extract entity id from heterogeneous payload shapes."""
    for key in ("element_id", "entity_id", "id", "uid", "node_id"):
        value = entity.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _ensure_entity_id(entity: Dict[str, Any], idx: Optional[int] = None) -> str:
    """Return graph entity id, deriving a stable fallback when upstream id is missing."""
    existing_id = _extract_entity_id(entity)
    if existing_id is not None and str(existing_id).strip():
        return str(existing_id).strip()

    name = _extract_entity_name(entity, idx)
    entity_type = _extract_entity_type(entity)
    file_path = str(_extract_entity_file_path(entity) or "")
    source_id = str(_extract_entity_source_id(entity) or "")
    material = f"{_canonical_label(name)}|{_canonical_label(entity_type)}|{file_path}|{source_id}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"node-{digest}"


def _extract_prop(obj: Dict[str, Any], key: str) -> Any:
    """Extract property checking top level, properties dict, and metadata dict."""
    if not isinstance(obj, dict):
        return None
    val = obj.get(key)
    if val is not None and str(val).strip() != "":
        return val
    props = obj.get("properties")
    if isinstance(props, dict):
        val = props.get(key)
        if val is not None and str(val).strip() != "":
            return val
    meta = obj.get("metadata")
    if isinstance(meta, dict):
        val = meta.get(key)
        if val is not None and str(val).strip() != "":
            return val
    return None


def _extract_entity_file_path(entity: Dict[str, Any]) -> Any:
    """Extract file path provenance from entity payload."""
    for key in ("file_path", "source_file", "source_path"):
        val = _extract_prop(entity, key)
        if val:
            return val
    return None


def _extract_entity_source_id(entity: Dict[str, Any]) -> Any:
    """Extract source id provenance from entity payload."""
    for key in ("source_id", "chunk_id", "doc_id"):
        val = _extract_prop(entity, key)
        if val is not None and str(val).strip():
            return val
    return None


def _extract_relationship_id(relationship: Dict[str, Any]) -> Any:
    """Extract relation id from heterogeneous payload shapes."""
    for key in ("relation_id", "id", "edge_id", "uid"):
        value = relationship.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _ensure_relationship_id(relationship: Dict[str, Any]) -> str:
    """Return relationship id, deriving stable fallback when upstream id is missing."""
    existing_id = _extract_relationship_id(relationship)
    if existing_id is not None and str(existing_id).strip():
        return str(existing_id).strip()

    src, dst = _extract_relationship_endpoints(relationship)
    rel = _canonical_label(_extract_relationship_label(relationship))
    source_id = str(_extract_relationship_source_id(relationship) or "")
    file_path = str(_extract_relationship_file_path(relationship) or "")
    material = f"{src}|{rel}|{dst}|{source_id}|{file_path}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"rel-{digest}"


def _extract_relationship_file_path(relationship: Dict[str, Any]) -> Any:
    """Extract file path provenance from relationship payload."""
    for key in ("file_path", "source_file", "source_path"):
        val = _extract_prop(relationship, key)
        if val:
            return val
    return None


def _extract_relationship_source_id(relationship: Dict[str, Any]) -> Any:
    """Extract source id provenance from relationship payload."""
    for key in ("source_id", "chunk_id", "doc_id"):
        val = _extract_prop(relationship, key)
        if val is not None and str(val).strip():
            return val
    return None


def _extract_relationship_endpoints(relationship: Dict[str, Any]) -> Tuple[str, str]:
    """Extract source/target labels from relationship payload with flexible key support."""
    source = (
        relationship.get("source")
        or relationship.get("entity1")
        or relationship.get("entity_1")
        or relationship.get("from")
        or relationship.get("src")
        or relationship.get("head")
        or relationship.get("start")
    )
    target = (
        relationship.get("target")
        or relationship.get("entity2")
        or relationship.get("entity_2")
        or relationship.get("to")
        or relationship.get("dst")
        or relationship.get("tail")
        or relationship.get("end")
    )
    return _canonical_label(source), _canonical_label(target)


def _extract_relationship_endpoints_raw(relationship: Dict[str, Any]) -> Tuple[str, str]:
    """Extract source/target labels as-is for API responses."""
    source = (
        relationship.get("source")
        or relationship.get("entity1")
        or relationship.get("entity_1")
        or relationship.get("from")
        or relationship.get("src")
        or relationship.get("head")
        or relationship.get("start")
        or ""
    )
    target = (
        relationship.get("target")
        or relationship.get("entity2")
        or relationship.get("entity_2")
        or relationship.get("to")
        or relationship.get("dst")
        or relationship.get("tail")
        or relationship.get("end")
        or ""
    )
    return str(source).strip(), str(target).strip()


def _resolve_relationship_endpoints_for_response(
    relationship: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Resolve missing relation endpoints from source overlap and description mentions."""
    source, target = _extract_relationship_endpoints_raw(relationship)
    if source and target:
        return source, target

    # Build candidate entity names and their source ids.
    candidate_names: List[str] = []
    candidate_source_ids: Dict[str, set[str]] = {}
    for idx, entity in enumerate(entities):
        name = _extract_entity_name(entity, idx).strip()
        if not name:
            continue
        if name not in candidate_source_ids:
            candidate_names.append(name)
            candidate_source_ids[name] = set()
        candidate_source_ids[name].update(_split_source_ids(_extract_entity_source_id(entity)))

    if not candidate_names:
        return source or "Unknown", target or "Unknown"

    rel_source_ids = set(_split_source_ids(_extract_relationship_source_id(relationship)))
    desc = str(relationship.get("description") or "")
    desc_lower = desc.lower()

    # Score entity relevance for this relationship.
    # Higher score means a stronger endpoint candidate.
    scored: Dict[str, Dict[str, Any]] = {}
    for name in candidate_names:
        score = 0
        first_pos = 10**9

        if rel_source_ids and candidate_source_ids.get(name, set()).intersection(rel_source_ids):
            score += 5

        name_l = name.lower()
        pos = desc_lower.find(name_l) if name_l else -1
        if pos >= 0:
            score += 3
            first_pos = min(first_pos, pos)

        if score > 0:
            scored[name] = {"score": score, "pos": first_pos}

    ranked = sorted(
        scored.keys(),
        key=lambda n: (-int(scored[n]["score"]), int(scored[n]["pos"]), n.lower()),
    )

    resolved_source = source
    resolved_target = target

    # Fill missing endpoints from ranked candidates.
    if not resolved_source and ranked:
        resolved_source = ranked[0]

    if not resolved_target:
        for candidate in ranked:
            if candidate != resolved_source:
                resolved_target = candidate
                break

    # Last-resort deterministic fill keeps payload mappable for clients.
    if not resolved_source:
        resolved_source = "Unknown"
    if not resolved_target:
        resolved_target = "Unknown"

    return resolved_source, resolved_target


def _build_graph_relationship_model(
    relationship: Dict[str, Any],
    entities: List[Dict[str, Any]],
) -> GraphRelationshipModel:
    """Build relationship response model with resilient endpoint resolution."""
    source, target = _resolve_relationship_endpoints_for_response(relationship, entities)
    relation = _extract_relationship_label(relationship) or "related_to"
    rel_element_id = str(relationship.get("element_id") or relationship.get("id") or _ensure_relationship_id(relationship))

    return GraphRelationshipModel(
        element_id=rel_element_id,
        source=source,
        target=target,
        relation=relation,
        created_at=_extract_prop(relationship, "created_at") or _extract_prop(relationship, "create_time"),
        description=_extract_prop(relationship, "description"),
        file_path=_extract_relationship_file_path(relationship),
        keywords=_extract_prop(relationship, "keywords"),
        source_id=_extract_relationship_source_id(relationship),
        weight=_extract_prop(relationship, "weight"),
    )


def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate entities using stable entity ids."""
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx, entity in enumerate(entities):
        entity_id = _ensure_entity_id(entity, idx)
        if entity_id in seen:
            continue
        seen.add(entity_id)
        deduped.append(entity)

    return deduped


def _dedupe_relationships(
    relationships: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate relationships by canonicalized endpoint+relation signature."""
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for rel in relationships:
        src, dst = _resolve_relationship_endpoints_for_response(rel, entities)
        rel_label = _extract_relationship_label(rel) or "related_to"
        rel_id = _ensure_relationship_id(rel)

        signature = "|".join(
            [
                _canonical_label(src),
                _canonical_label(rel_label),
                _canonical_label(dst),
                str(rel_id).strip(),
            ]
        )

        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(rel)

    return deduped


def _build_entity_match_tokens(entity: Dict[str, Any], idx: Optional[int] = None) -> set[str]:
    """Build comparable tokens for matching entities to relationship endpoints."""
    tokens: set[str] = set()
    for candidate in (
        _extract_entity_name(entity, idx),
        entity.get("entity_name"),
        entity.get("name"),
        entity.get("entity"),
        entity.get("label"),
        entity.get("id"),
        entity.get("entity_id"),
        entity.get("node_id"),
        entity.get("uid"),
    ):
        normalized = _canonical_label(candidate)
        if normalized:
            tokens.add(normalized)
    return tokens


async def _filter_graph_with_llm(
    query: str,
    answer: str,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    workspace_id: int,
    agent_id: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Use LLM to filter graph entities and relationships based on relevance to the answer.

    Args:
        answer: The generated answer text
        entities: List of all entities from LightRAG
        relationships: List of all relationships from LightRAG

    Returns:
        Tuple of (filtered_entities, filtered_relationships)
    """
    if not entities:
        return [], []

    # Build stable entity catalog so LLM always selects by entity_id.
    entity_catalog: List[Dict[str, Any]] = []
    entity_catalog_lines: List[str] = []

    for idx, entity in enumerate(entities):
        entity_id = _ensure_entity_id(entity, idx)
        entity_name = _extract_entity_name(entity, idx)
        entity_type = _extract_entity_type(entity)
        entity_desc = str(entity.get("description") or "")
        line = f"entity_id={entity_id} | name={entity_name} | type={entity_type}"
        if entity_desc:
            line += f" | description={entity_desc[:120]}"
        entity_catalog_lines.append(line)

        entity_catalog.append(
            {
                "entity": entity,
                "entity_id": entity_id,
                "match_tokens": _build_entity_match_tokens({**entity, "entity_id": entity_id}, idx),
            }
        )

    available_entity_ids = {entry["entity_id"] for entry in entity_catalog}

    # Prepare LLM prompt
    prompt = f"""Given the following user query, generated answer, and list of graph entities, identify which entities are directly mentioned or relevant.

User Query:
{query}

Answer:
{answer}

Available Entities:
{chr(10).join(entity_catalog_lines)}

Task: Return a JSON object with a list of entity_id values that are relevant to the answer. Only include IDs from Available Entities.

Response format:
{{
    "relevant_entity_ids": ["node-abc123", "E42", ...]
}}
"""

    # Call LLM to filter entities
    try:
        llm_filter_func = _build_llm_filter_func(workspace_id, agent_id)
        response = await llm_filter_func(
            prompt=prompt,
            temperature=0.1,
            top_p=1,
            n=1,
        )

        relevant_entity_ids = {
            entity_id
            for entity_id in _parse_relevant_entity_ids(response)
            if entity_id in available_entity_ids
        }

        # Backward compatibility: if model still returns indices, map them to entity IDs.
        if not relevant_entity_ids:
            for idx in _parse_relevant_indices(response):
                if 0 <= idx < len(entity_catalog):
                    relevant_entity_ids.add(entity_catalog[idx]["entity_id"])

        # Safety fallback: keep original graph when the LLM cannot select valid IDs.
        # This preserves recall and avoids empty responses due to JSON/format drift.
        if not relevant_entity_ids:
            logger.warning(
                "LLM returned no valid entity IDs; using unfiltered graph fallback",
                workspace_id=workspace_id,
                agent_id=agent_id,
                total_entities=len(entities),
                total_relationships=len(relationships),
            )
            return entities, relationships

        filtered_entities = [
            entry["entity"]
            for entry in entity_catalog
            if entry["entity_id"] in relevant_entity_ids
        ]

        # Build robust token set (entity labels + ids) for endpoint matching.
        relevant_entity_tokens: set[str] = set()
        for entry in entity_catalog:
            if entry["entity_id"] in relevant_entity_ids:
                relevant_entity_tokens.update(entry["match_tokens"])

        # Primary relation filter: both endpoints map to selected entities.
        filtered_relationships = []
        for rel in relationships:
            src_token, dst_token = _extract_relationship_endpoints(rel)
            if src_token in relevant_entity_tokens or dst_token in relevant_entity_tokens:
                filtered_relationships.append(rel)

        # Fallback A: source_id overlap with selected entities (when endpoint naming differs).
        if relationships and filtered_entities and not filtered_relationships:
            selected_source_ids: set[str] = set()
            for entity in filtered_entities:
                selected_source_ids.update(_split_source_ids(_extract_entity_source_id(entity)))

            if selected_source_ids:
                for rel in relationships:
                    rel_source_ids = set(_split_source_ids(_extract_relationship_source_id(rel)))
                    if rel_source_ids.intersection(selected_source_ids):
                        filtered_relationships.append(rel)

        logger.info(
            f"LLM filtering complete",
            total_entities=len(entities),
            filtered_entities=len(filtered_entities),
            total_relationships=len(relationships),
            filtered_relationships=len(filtered_relationships),
            selected_entity_ids=len(relevant_entity_ids),
        )

        return filtered_entities, filtered_relationships

    except Exception as e:
        logger.error(
            "LLM filtering failed, returning unfiltered graph",
            error=e,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )
        # Fallback: return all entities if LLM filtering fails
        return entities, relationships


def _parse_graph_from_context(context_data: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse graph entities and relationships from LightRAG context data.

    Args:
        context_data: Raw context data from LightRAG

    Returns:
        Tuple of (entities, relationships)
    """
    all_entities = []
    all_relationships = []

    kb_results = context_data.get("kb_results", {})

    if isinstance(kb_results, list):
        for kb_payload in kb_results:
            if not isinstance(kb_payload, dict):
                continue
            graph_data = kb_payload.get("graph_data", {})
            if not isinstance(graph_data, dict):
                continue
            entities = graph_data.get("entities", [])
            relationships = graph_data.get("relationship") or graph_data.get("relationships") or []
            if isinstance(entities, list):
                all_entities.extend(entities)
            if isinstance(relationships, list):
                all_relationships.extend(relationships)
        return all_entities, all_relationships

    for kb_source, kb_payload in kb_results.items():
        if not isinstance(kb_payload, dict) or "error" in kb_payload:
            continue

        # Get raw context
        raw_context = kb_payload.get("_raw_context", [])
        if not isinstance(raw_context, list):
            raw_context = [raw_context]

        for context_item in raw_context:
            if not isinstance(context_item, dict):
                continue

            # Extract entities
            entities = context_item.get("entities", [])
            if isinstance(entities, list):
                all_entities.extend(entities)

            # Extract relationships
            relationships = context_item.get("relationships", [])
            if isinstance(relationships, list):
                all_relationships.extend(relationships)

    return all_entities, all_relationships


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Fetch filtered graph data endpoint.

    POST /api/v2/kb/graph-data
    Headers: Authorization: Bearer <token>
    Body: {
        "query": "What is asset management?",
        "answer": "Asset management is...",
        "workspace_id": 123,
        "mode": "hybrid"
    }

    Flow:
    1. Check Redis cache for existing filtered graph
    2. If not cached:
       a. Fetch context from LightRAG (only_context mode)
       b. Use LLM to filter relevant entities/relationships
       c. Save filtered result to Redis
    3. Return filtered graph data

    Returns:
        200: FetchGraphResponse with filtered graph data
        400: Validation error
        403: Not authorized for workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)
    start_time = time.time()

    logger.info(
        "Fetch graph request received",
        correlation_id=correlation_id,
        user_id=user_id
    )

    try:
        # Parse and validate request payload
        raw_payload, error_response = parse_request(req, FetchGraphRequest)
        if error_response or not isinstance(raw_payload, FetchGraphRequest):
            return error_response or create_error_response("Invalid request payload", status_code=400)

        payload: FetchGraphRequest = raw_payload

        workspace_id = payload.workspace_id
        query = payload.query
        answer = payload.answer
        mode = payload.mode
        graph_only = payload.graph_only
        agent_id = payload.agent_id

        # ===========================================
        # STEP 1: Try Cache First
        # ===========================================
        cache_key = _make_graph_cache_key(
            workspace_id,
            query,
            answer,
            mode,
            graph_only=graph_only,
        )
        cached_result = None

        if redis_manager.is_available and getattr(settings.cache, 'REDIS_ENABLED', True):
            cached_data = redis_manager.get(cache_key)
            if cached_data:
                try:
                    cached_result = json.loads(cached_data)
                    cached_nodes = cached_result.get("nodes", []) if isinstance(cached_result, dict) else []
                    cached_edges = cached_result.get("edges", []) if isinstance(cached_result, dict) else []

                    if not cached_nodes and not cached_edges:
                        logger.info(
                            "Evicting empty graph cache entry so request queries fresh graph",
                            cache_key=cache_key,
                            workspace_id=workspace_id,
                        )
                        redis_manager.delete(cache_key)
                        cached_result = None
                    else:
                        cache_elapsed = time.time() - start_time

                        logger.info(
                            "Fetch graph completed from cache",
                            correlation_id=correlation_id,
                            workspace_id=workspace_id,
                            cache_hit=True,
                            response_time_ms=round(cache_elapsed * 1000, 2),
                        )

                        return create_success_response(
                            message="Graph data retrieved successfully (cached)",
                            data=cached_result,
                            status_code=200,
                            correlation_id=correlation_id
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse cached graph data: {e}")
                    cached_result = None

        # ===========================================
        # STEP 2: Validate User Access
        # ===========================================
        workspace_service = get_workspace_service()

        is_authorized, role_id = await workspace_service.validate_user_workspace_access(
            user_id=user_id,
            workspace_id=workspace_id
        )

        if not is_authorized:
            logger.warning(
                "User not authorized for workspace",
                user_id=user_id,
                workspace_id=workspace_id,
                correlation_id=correlation_id
            )
            raise AuthorizationException(
                message=f"You are not authorized to access workspace {workspace_id}"
            )

        # ===========================================
        # STEP 3: Get Workspace Configuration
        # ===========================================
        storage_paths = await get_workspace_storage_paths(workspace_id)

        if not storage_paths:
            logger.error(
                "Failed to retrieve workspace storage paths",
                workspace_id=workspace_id,
                correlation_id=correlation_id
            )
            raise ValidationException(
                message=f"Failed to retrieve workspace configuration for workspace {workspace_id}"
            )

        domain = str(storage_paths.get("domain", ""))
        kb_name = str(storage_paths.get("kb_name", ""))
        raw_all_kbs = storage_paths.get("all_kb_titles")
        raw_is_kg = storage_paths.get("is_kg")
        is_kg: bool = bool(raw_is_kg) if raw_is_kg is not None else False

        additional_kbs: Optional[List[str]] = None
        if not is_kg and isinstance(raw_all_kbs, list) and raw_all_kbs:
            additional_kbs = [str(title) for title in raw_all_kbs if title]

        logger.info(
            "Workspace configuration retrieved",
            workspace_id=workspace_id,
            domain=domain,
            kb_name=kb_name,
            kb_count=1 + (len(additional_kbs) if additional_kbs else 0),
            role_id=role_id,
            correlation_id=correlation_id
        )

        # ===========================================
        # STEP 4: Query Evidence Cache -> LightRAG Fallback
        # ===========================================
        context_data = None
        if not graph_only:
            context_data = _get_cached_query_evidence(workspace_id, query, mode)

        if context_data:
            logger.info(
                "Graph context loaded from query evidence cache",
                workspace_id=workspace_id,
            )
        else:
            if graph_only:
                logger.info(
                    "graph_only=true, skipping query evidence cache and using LightRAG graph retrieval",
                    workspace_id=workspace_id,
                )
            logger.info(
                "Query evidence cache miss, fetching context from LightRAG",
                workspace_id=workspace_id,
            )
            context_data = await _fetch_context_from_lightrag(
                workspace_id=workspace_id,
                role_id=role_id,
                domain=domain,
                kb_name=kb_name,
                query=query,
                mode=mode,
                additional_kbs=additional_kbs,
                is_kg=is_kg,
                agent_id=agent_id,
            )

        # ===========================================
        # STEP 5: Parse Graph Data from Context
        # ===========================================
        entities, relationships = _parse_graph_from_context(context_data)

        if graph_only and not _has_non_empty_graph_content(entities, relationships):
            logger.warning(
                "graph_only retrieval returned insufficient graph evidence",
                workspace_id=workspace_id,
                entity_count=len(entities),
                relationship_count=len(relationships),
                graph_source=(context_data.get("metadata", {}) if isinstance(context_data, dict) else {}).get("source"),
            )
            raise ValidationException(
                message=(
                    "graph_only mode requires graph-backed entities or relationships, "
                    "but the current retrieval returned insufficient graph evidence"
                )
            )

        logger.info(
            "Graph data parsed from context",
            entity_count=len(entities),
            relationship_count=len(relationships)
        )

        # ===========================================
        # STEP 6: Filter with LLM
        # ===========================================
        filtered_entities, filtered_relationships = await _filter_graph_with_llm(
            query=query,
            answer=answer,
            entities=entities,
            relationships=relationships,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

        filtered_entities = _dedupe_entities(filtered_entities)
        filtered_relationships = _dedupe_relationships(filtered_relationships, filtered_entities)

        # ===========================================
        # STEP 7: Build Response
        # ===========================================
        kb_targets: List[str] = [kb_name]
        if not is_kg:
            kb_targets.extend(_resolve_additional_kb_paths(kb_name, additional_kbs))

        knowledge_bases: List[str] = []
        for target in kb_targets:
            t = str(target).strip()
            if not t:
                continue
            if domain and not t.startswith(f"{domain}/"):
                knowledge_bases.append(f"{domain}/{t}")
            else:
                knowledge_bases.append(t)

        nodes_list: List[GraphNodeModel] = []
        node_id_map: Dict[str, str] = {}

        for idx, e in enumerate(filtered_entities):
            raw_element_id = str(e.get("element_id") or e.get("id") or _ensure_entity_id(e, idx))
            short_id = raw_element_id.split(":")[-1] if ":" in raw_element_id else raw_element_id

            entity_name = _extract_entity_name(e, idx)
            entity_type = _extract_entity_type(e)
            file_path = _extract_entity_file_path(e) or ""
            source_id = _extract_entity_source_id(e) or ""
            created_at = _extract_prop(e, "created_at") or _extract_prop(e, "create_time") or ""
            description = _extract_prop(e, "description") or ""
            truncate = _extract_prop(e, "truncate") or ""

            raw_labels = e.get("labels")
            clean_labels = []
            if isinstance(raw_labels, list):
                clean_labels = [
                    str(lbl).strip()
                    for lbl in raw_labels
                    if str(lbl).strip() and str(lbl).strip().lower() not in ("knowledgegraph", "knowledge graph", "knowledge_graph")
                ]

            if clean_labels:
                labels = clean_labels
            elif entity_name:
                labels = [entity_name]
            elif entity_type and entity_type.lower() != "unknown":
                labels = [entity_type]
            else:
                labels = ["Entity"]

            properties: Dict[str, Any] = {
                "file_path": file_path,
                "entity_type": entity_type,
                "truncate": truncate,
                "description": description,
                "created_at": created_at,
                "source_id": source_id,
                "entity_id": entity_name,
            }

            existing_props = e.get("properties")
            if isinstance(existing_props, dict):
                for k, v in existing_props.items():
                    if k not in properties:
                        properties[k] = v

            node_model = GraphNodeModel(
                id=short_id,
                element_id=raw_element_id,
                labels=labels,
                properties=properties,
            )
            nodes_list.append(node_model)

            for key in (raw_element_id, short_id, entity_name, e.get("entity_id"), e.get("element_id"), e.get("id")):
                if key and str(key).strip():
                    k_str = str(key).strip()
                    node_id_map[k_str] = short_id
                    node_id_map[k_str.lower()] = short_id

        edges_list: List[GraphEdgeModel] = []
        for idx, r in enumerate(filtered_relationships):
            raw_edge_element_id = str(r.get("element_id") or r.get("id") or _ensure_relationship_id(r))
            short_edge_id = raw_edge_element_id.split(":")[-1] if ":" in raw_edge_element_id else raw_edge_element_id

            raw_src, raw_dst = _resolve_relationship_endpoints_for_response(r, filtered_entities)

            src_id = node_id_map.get(raw_src) or node_id_map.get(raw_src.lower()) or raw_src
            dst_id = node_id_map.get(raw_dst) or node_id_map.get(raw_dst.lower()) or raw_dst

            edge_type = str(r.get("type") or r.get("relation") or _extract_prop(r, "type") or "DIRECTED")
            file_path = _extract_relationship_file_path(r) or ""
            source_id = _extract_relationship_source_id(r) or ""
            created_at = _extract_prop(r, "created_at") or _extract_prop(r, "create_time") or ""
            description = _extract_prop(r, "description") or ""
            truncate = _extract_prop(r, "truncate") or ""
            keywords = _extract_prop(r, "keywords") or ""
            weight = _extract_prop(r, "weight")
            if weight is None:
                weight = 1

            edge_properties: Dict[str, Any] = {
                "file_path": file_path,
                "truncate": truncate,
                "keywords": keywords,
                "weight": weight,
                "description": description,
                "created_at": created_at,
                "source_id": source_id,
            }

            existing_props = r.get("properties")
            if isinstance(existing_props, dict):
                for k, v in existing_props.items():
                    if k not in edge_properties:
                        edge_properties[k] = v

            edge_model = GraphEdgeModel(
                id=short_edge_id,
                element_id=raw_edge_element_id,
                type=edge_type,
                source=src_id,
                target=dst_id,
                properties=edge_properties,
            )
            edges_list.append(edge_model)

        graph_data = FilteredGraphDataModel(
            knowledge_bases=knowledge_bases,
            nodes=nodes_list,
            edges=edges_list,
            metadata={
                "total_nodes": len(nodes_list),
                "total_edges": len(edges_list),
                "total_entities": len(nodes_list),
                "total_relationships": len(edges_list),
                "original_entity_count": len(entities),
                "original_relationship_count": len(relationships),
                "graph_source": (context_data.get("metadata", {}) if isinstance(context_data, dict) else {}).get("source"),
                "graph_only": graph_only,
            }
        )

        response_data = {
            "knowledge_bases": knowledge_bases,
            "nodes": [n.dict() for n in nodes_list],
            "edges": [e.dict() for e in edges_list],
            "metadata": graph_data.metadata,
            "query": query,
            "workspace_id": workspace_id,
            "graph_only": graph_only,
            "cached": False
        }


        # ===========================================
        # STEP 8: Cache Result (Only if graph data contains nodes or edges)
        # ===========================================
        has_results = bool(nodes_list or edges_list)
        if has_results and redis_manager.is_available and getattr(settings.cache, 'REDIS_ENABLED', True):
            cached_payload = {**response_data, "cached": True}
            redis_manager.setex(
                cache_key,
                GRAPH_CACHE_TTL_SECONDS,
                json.dumps(cached_payload)
            )
        else:
            logger.info(
                "Skipping cache write for graph request because no nodes/edges were returned",
                workspace_id=workspace_id,
                has_results=has_results,
            )


        total_elapsed = time.time() - start_time

        logger.info(
            "Fetch graph completed successfully",
            correlation_id=correlation_id,
            workspace_id=workspace_id,
            entity_count=len(filtered_entities),
            relationship_count=len(filtered_relationships),
            cache_hit=False,
            total_time_ms=round(total_elapsed * 1000, 2),
        )

        return create_success_response(
            message="Graph data retrieved successfully",
            data=response_data,
            status_code=200,
            correlation_id=correlation_id
        )

    except ValidationException as e:
        logger.warning(
            "Validation error",
            error=e.message,
            correlation_id=correlation_id
        )
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id
        )

    except AuthorizationException as e:
        logger.warning(
            "Authorization error",
            error=e.message,
            user_id=user_id,
            workspace_id=payload.workspace_id if payload is not None else None,
            correlation_id=correlation_id
        )
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id
        )

    except Exception as e:
        logger.error(
            "Fetch graph failed",
            error=e,
            correlation_id=correlation_id,
            exc_info=True
        )
        return create_internal_error_response(
            message="An error occurred while fetching graph data",
            error=e,
            error_code="INTERNAL_ERROR",
            correlation_id=correlation_id
        )

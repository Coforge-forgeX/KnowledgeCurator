"""Mutate knowledge graph nodes/relationships with workspace-safe dual updates."""
import json
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, text

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.database import DocumentMetadata, FileTask, get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.core.neo4j_driver import get_neo4j_driver
from src.helpers.workspace_permissions import require_workspace_admin_curator
from src.shared import create_error_response, create_success_response, parse_request

from .payloads import (
    MutateKnowledgeGraphRequest,
    NodeMutationPayload,
    RelationshipMutationPayload,
)

logger = get_logger(__name__)

CHUNK_TABLES = ["lightrag_vdb_chunks"]
RELATION_TABLES = ["lightrag_vdb_relation", "lightrag_vdb_relations"]


async def _validate_admin_curate_permission(user_id: int, workspace_id: int) -> None:
    await require_workspace_admin_curator(
        user_id=user_id,
        workspace_id=workspace_id,
        action_description="mutate knowledge graph",
    )


def _norm(value: Optional[str]) -> str:
    return str(value or "").strip().replace("\\", "/")


async def _validate_workspace_scope(
    workspace_id: int,
    file_path: str,
    source_id: Optional[str],
    full_doc_id: Optional[str],
) -> Dict[str, Any]:
    """Ensure requested scope maps to indexed rows in the same workspace."""
    normalized_path = _norm(file_path)
    normalized_source_id = _norm(source_id)
    normalized_full_doc_id = _norm(full_doc_id)

    async with get_async_session() as session:
        task_stmt = select(FileTask).where(
            FileTask.workspace_id == workspace_id,
            FileTask.file_path == normalized_path,
        )
        task_result = await session.execute(task_stmt)
        task = task_result.scalar_one_or_none()

        if not task:
            raise ValidationException(
                message=(
                    f"File path '{normalized_path}' is not indexed in workspace {workspace_id}"
                )
            )

        metadata_stmt = select(DocumentMetadata).where(
            DocumentMetadata.workspace_id == workspace_id,
            DocumentMetadata.file_path == normalized_path,
        )
        metadata_result = await session.execute(metadata_stmt)
        metadata_rows = metadata_result.scalars().all()

    doc_ids = [row.full_doc_id for row in metadata_rows if row.full_doc_id]
    if normalized_full_doc_id and normalized_full_doc_id not in doc_ids:
        raise ValidationException(
            message=(
                f"full_doc_id '{normalized_full_doc_id}' does not belong to workspace {workspace_id} "
                f"for file_path '{normalized_path}'"
            )
        )

    return {
        "file_path": normalized_path,
        "source_id": normalized_source_id or None,
        "full_doc_id": normalized_full_doc_id or None,
        "full_doc_ids": doc_ids,
    }


def _merge_properties(base: Dict[str, Any], additional: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if additional:
        for key, value in additional.items():
            if value is not None:
                merged[str(key)] = value
    return merged


async def _mutate_node_neo4j(
    action: str,
    node: NodeMutationPayload,
    scoped: Dict[str, Any],
) -> Dict[str, Any]:
    neo4j_driver = get_neo4j_driver()
    entity_name = _norm(node.entity_name)
    if not entity_name:
        raise ValidationException(message="entity_name is required")

    scoped_source_id = scoped.get("source_id")
    scoped_file_path = scoped["file_path"]

    params: Dict[str, Any] = {
        "entity_name": entity_name,
        "file_path": scoped_file_path,
        "scope_source_id": scoped_source_id,
    }

    if action == "create":
        properties = _merge_properties(
            {
                "entity_name": entity_name,
                "entity_type": node.entity_type,
                "description": node.description,
                "source_id": node.source_id or scoped_source_id,
                "file_path": scoped_file_path,
                "source": scoped_file_path,
            },
            node.additional_properties,
        )
        create_query = """
        CREATE (n:Entity)
        SET n += $properties
        RETURN n.entity_name AS entity_name
        """
        result = await neo4j_driver.execute_write_query(create_query, {"properties": properties})
        return {"neo4j_rows": len(result)}

    if action == "update":
        new_name = _norm(node.new_entity_name) or entity_name
        properties = _merge_properties(
            {
                "entity_name": new_name,
                "entity_type": node.entity_type,
                "description": node.description,
                "source_id": node.source_id or scoped_source_id,
            },
            node.additional_properties,
        )
        update_query = """
        MATCH (n)
        WHERE n.entity_name = $entity_name
          AND n.file_path = $file_path
          AND ($scope_source_id IS NULL OR n.source_id = $scope_source_id)
        SET n += $properties
        RETURN count(n) AS updated_count
        """
        result = await neo4j_driver.execute_write_query(update_query, {**params, "properties": properties})
        updated_count = int(result[0].get("updated_count", 0)) if result else 0
        return {"neo4j_rows": updated_count}

    delete_query = """
    MATCH (n)
    WHERE n.entity_name = $entity_name
      AND n.file_path = $file_path
      AND ($scope_source_id IS NULL OR n.source_id = $scope_source_id)
    DETACH DELETE n
    RETURN count(n) AS deleted_count
    """
    result = await neo4j_driver.execute_write_query(delete_query, params)
    deleted_count = int(result[0].get("deleted_count", 0)) if result else 0
    return {"neo4j_rows": deleted_count}


async def _mutate_relationship_neo4j(
    action: str,
    relationship: RelationshipMutationPayload,
    scoped: Dict[str, Any],
) -> Dict[str, Any]:
    neo4j_driver = get_neo4j_driver()
    source = _norm(relationship.source)
    target = _norm(relationship.target)
    relation = _norm(relationship.relation)
    if not source or not target or not relation:
        raise ValidationException(message="source, target, and relation are required")

    scoped_file_path = scoped["file_path"]
    scoped_source_id = scoped.get("source_id")

    if action == "create":
        props = _merge_properties(
            {
                "relation": relation,
                "description": relationship.description,
                "source_id": relationship.source_id or scoped_source_id,
                "file_path": scoped_file_path,
                "source": scoped_file_path,
            },
            relationship.additional_properties,
        )
        create_query = """
        MERGE (s:Entity {entity_name: $source_name, file_path: $file_path})
        MERGE (t:Entity {entity_name: $target_name, file_path: $file_path})
        CREATE (s)-[r:RELATED_TO]->(t)
        SET r += $properties
        RETURN count(r) AS created_count
        """
        result = await neo4j_driver.execute_write_query(
            create_query,
            {
                "source_name": source,
                "target_name": target,
                "file_path": scoped_file_path,
                "properties": props,
            },
        )
        created_count = int(result[0].get("created_count", 0)) if result else 0
        return {"neo4j_rows": created_count}

    if action == "update":
        new_relation = _norm(relationship.new_relation) or relation
        props = _merge_properties(
            {
                "relation": new_relation,
                "description": relationship.description,
                "source_id": relationship.source_id or scoped_source_id,
            },
            relationship.additional_properties,
        )
        update_query = """
        MATCH (s:Entity {entity_name: $source_name, file_path: $file_path})-[r:RELATED_TO]->(t:Entity {entity_name: $target_name, file_path: $file_path})
        WHERE r.relation = $relation
          AND ($scope_source_id IS NULL OR r.source_id = $scope_source_id)
        SET r += $properties
        RETURN count(r) AS updated_count
        """
        result = await neo4j_driver.execute_write_query(
            update_query,
            {
                "source_name": source,
                "target_name": target,
                "file_path": scoped_file_path,
                "relation": relation,
                "scope_source_id": scoped_source_id,
                "properties": props,
            },
        )
        updated_count = int(result[0].get("updated_count", 0)) if result else 0
        return {"neo4j_rows": updated_count}

    delete_query = """
    MATCH (s:Entity {entity_name: $source_name, file_path: $file_path})-[r:RELATED_TO]->(t:Entity {entity_name: $target_name, file_path: $file_path})
    WHERE r.relation = $relation
      AND ($scope_source_id IS NULL OR r.source_id = $scope_source_id)
    DELETE r
    RETURN count(r) AS deleted_count
    """
    result = await neo4j_driver.execute_write_query(
        delete_query,
        {
            "source_name": source,
            "target_name": target,
            "file_path": scoped_file_path,
            "relation": relation,
            "scope_source_id": scoped_source_id,
        },
    )
    deleted_count = int(result[0].get("deleted_count", 0)) if result else 0
    return {"neo4j_rows": deleted_count}


def _build_doc_ids(scoped: Dict[str, Any]) -> List[str]:
    if scoped.get("full_doc_id"):
        return [str(scoped["full_doc_id"])]
    return [str(doc_id) for doc_id in scoped.get("full_doc_ids", []) if doc_id]


async def _update_chunk_tables(
    action: str,
    node: NodeMutationPayload,
    scoped: Dict[str, Any],
) -> Dict[str, int]:
    """Best-effort updates for chunk table metadata fields."""
    affected = 0
    doc_ids = _build_doc_ids(scoped)
    source_id = scoped.get("source_id")

    if not doc_ids and not source_id:
        return {"chunks_affected": 0}

    async with get_async_session() as session:
        for table_name in CHUNK_TABLES:
            try:
                if action == "delete":
                    metadata_patch = {"deleted_entity": _norm(node.entity_name)}
                    if doc_ids:
                        total = 0
                        for doc_id in doc_ids:
                            stmt = text(
                                f"UPDATE {table_name} "
                                "SET metadata = COALESCE(metadata, '{}'::jsonb) || :metadata_patch::jsonb "
                                "WHERE file_path = :file_path AND full_doc_id = :doc_id"
                            )
                            result = await session.execute(
                                stmt,
                                {
                                    "metadata_patch": json.dumps(metadata_patch),
                                    "file_path": scoped["file_path"],
                                    "doc_id": doc_id,
                                },
                            )
                            if result.rowcount and result.rowcount > 0:
                                total += int(result.rowcount)
                        affected += total
                        continue

                    stmt = text(
                        f"UPDATE {table_name} "
                        "SET metadata = COALESCE(metadata, '{}'::jsonb) || :metadata_patch::jsonb "
                        "WHERE file_path = :file_path AND source_id = :source_id"
                    )
                    result = await session.execute(
                        stmt,
                        {
                            "metadata_patch": json.dumps(metadata_patch),
                            "file_path": scoped["file_path"],
                            "source_id": source_id,
                        },
                    )
                else:
                    entity_name = _norm(node.entity_name)
                    new_entity_name = _norm(node.new_entity_name) or entity_name
                    description = node.description
                    metadata_patch = {
                        "entity_name": new_entity_name,
                        "entity_type": node.entity_type,
                        "entity_description": description,
                    }
                    if doc_ids:
                        total = 0
                        for doc_id in doc_ids:
                            stmt = text(
                                f"UPDATE {table_name} "
                                "SET metadata = COALESCE(metadata, '{}'::jsonb) || :metadata_patch::jsonb "
                                "WHERE file_path = :file_path AND full_doc_id = :doc_id"
                            )
                            result = await session.execute(
                                stmt,
                                {
                                    "metadata_patch": json.dumps(metadata_patch),
                                    "file_path": scoped["file_path"],
                                    "doc_id": doc_id,
                                },
                            )
                            if result.rowcount and result.rowcount > 0:
                                total += int(result.rowcount)
                        affected += total
                        continue

                    stmt = text(
                        f"UPDATE {table_name} "
                        "SET metadata = COALESCE(metadata, '{}'::jsonb) || :metadata_patch::jsonb "
                        "WHERE file_path = :file_path AND source_id = :source_id"
                    )
                    result = await session.execute(
                        stmt,
                        {
                            "metadata_patch": json.dumps(metadata_patch),
                            "file_path": scoped["file_path"],
                            "source_id": source_id,
                        },
                    )

                if result.rowcount and result.rowcount > 0:
                    affected += int(result.rowcount)
            except Exception as exc:
                logger.warning(
                    "Chunk table sync failed",
                    table_name=table_name,
                    action=action,
                    error=str(exc),
                )

    return {"chunks_affected": affected}


async def _update_relation_tables(
    action: str,
    relationship: RelationshipMutationPayload,
    scoped: Dict[str, Any],
) -> Dict[str, int]:
    """Best-effort updates for relation tables in LightRAG VDB."""
    affected = 0
    source = _norm(relationship.source)
    target = _norm(relationship.target)
    relation = _norm(relationship.relation)
    new_relation = _norm(relationship.new_relation) or relation
    source_id = relationship.source_id or scoped.get("source_id")
    doc_ids = _build_doc_ids(scoped)

    async with get_async_session() as session:
        for table_name in RELATION_TABLES:
            try:
                if action == "create":
                    insert_stmt = text(
                        f"INSERT INTO {table_name} (source, target, relation, source_id, file_path, full_doc_id) "
                        "VALUES (:source, :target, :relation, :source_id, :file_path, :full_doc_id)"
                    )
                    full_doc_id = doc_ids[0] if doc_ids else None
                    result = await session.execute(
                        insert_stmt,
                        {
                            "source": source,
                            "target": target,
                            "relation": relation,
                            "source_id": source_id,
                            "file_path": scoped["file_path"],
                            "full_doc_id": full_doc_id,
                        },
                    )
                elif action == "update":
                    if doc_ids:
                        total = 0
                        for doc_id in doc_ids:
                            update_stmt = text(
                                f"UPDATE {table_name} SET relation = :new_relation, description = :description "
                                "WHERE source = :source AND target = :target AND relation = :relation "
                                "AND file_path = :file_path AND full_doc_id = :doc_id"
                            )
                            result = await session.execute(
                                update_stmt,
                                {
                                    "new_relation": new_relation,
                                    "description": relationship.description,
                                    "source": source,
                                    "target": target,
                                    "relation": relation,
                                    "file_path": scoped["file_path"],
                                    "doc_id": doc_id,
                                },
                            )
                            if result.rowcount and result.rowcount > 0:
                                total += int(result.rowcount)
                        affected += total
                        continue
                    else:
                        update_stmt = text(
                            f"UPDATE {table_name} SET relation = :new_relation, description = :description "
                            "WHERE source = :source AND target = :target AND relation = :relation "
                            "AND file_path = :file_path"
                        )
                        result = await session.execute(
                            update_stmt,
                            {
                                "new_relation": new_relation,
                                "description": relationship.description,
                                "source": source,
                                "target": target,
                                "relation": relation,
                                "file_path": scoped["file_path"],
                            },
                        )
                else:
                    if doc_ids:
                        total = 0
                        for doc_id in doc_ids:
                            delete_stmt = text(
                                f"DELETE FROM {table_name} "
                                "WHERE source = :source AND target = :target AND relation = :relation "
                                "AND file_path = :file_path AND full_doc_id = :doc_id"
                            )
                            result = await session.execute(
                                delete_stmt,
                                {
                                    "source": source,
                                    "target": target,
                                    "relation": relation,
                                    "file_path": scoped["file_path"],
                                    "doc_id": doc_id,
                                },
                            )
                            if result.rowcount and result.rowcount > 0:
                                total += int(result.rowcount)
                        affected += total
                        continue
                    else:
                        delete_stmt = text(
                            f"DELETE FROM {table_name} "
                            "WHERE source = :source AND target = :target AND relation = :relation "
                            "AND file_path = :file_path"
                        )
                        result = await session.execute(
                            delete_stmt,
                            {
                                "source": source,
                                "target": target,
                                "relation": relation,
                                "file_path": scoped["file_path"],
                            },
                        )

                if result.rowcount and result.rowcount > 0:
                    affected += int(result.rowcount)
            except Exception as exc:
                logger.warning(
                    "Relation table sync failed",
                    table_name=table_name,
                    action=action,
                    error=str(exc),
                )

    return {"relations_affected": affected}


async def _delete_node_related_relations(node: NodeMutationPayload, scoped: Dict[str, Any]) -> Dict[str, int]:
    """Delete relation table rows where the node appears as source or target in this scope."""
    affected = 0
    entity_name = _norm(node.entity_name)
    doc_ids = _build_doc_ids(scoped)

    async with get_async_session() as session:
        for table_name in RELATION_TABLES:
            try:
                if doc_ids:
                    total = 0
                    for doc_id in doc_ids:
                        stmt = text(
                            f"DELETE FROM {table_name} "
                            "WHERE file_path = :file_path AND full_doc_id = :doc_id "
                            "AND (source = :entity_name OR target = :entity_name)"
                        )
                        result = await session.execute(
                            stmt,
                            {
                                "file_path": scoped["file_path"],
                                "doc_id": doc_id,
                                "entity_name": entity_name,
                            },
                        )
                        if result.rowcount and result.rowcount > 0:
                            total += int(result.rowcount)
                    affected += total
                    continue

                stmt = text(
                    f"DELETE FROM {table_name} "
                    "WHERE file_path = :file_path AND (source = :entity_name OR target = :entity_name)"
                )
                result = await session.execute(
                    stmt,
                    {
                        "file_path": scoped["file_path"],
                        "entity_name": entity_name,
                    },
                )
                if result.rowcount and result.rowcount > 0:
                    affected += int(result.rowcount)
            except Exception as exc:
                logger.warning(
                    "Relation cleanup for node delete failed",
                    table_name=table_name,
                    error=str(exc),
                )

    return {"relations_affected": affected}


async def _apply_mutation(payload: MutateKnowledgeGraphRequest) -> Tuple[Dict[str, Any], List[str]]:
    scoped = await _validate_workspace_scope(
        workspace_id=payload.workspace_id,
        file_path=payload.scope.file_path,
        source_id=payload.scope.source_id,
        full_doc_id=payload.scope.full_doc_id,
    )

    warnings: List[str] = []
    result: Dict[str, Any] = {
        "workspace_id": payload.workspace_id,
        "action": payload.action,
        "target": payload.target,
        "scope": {
            "file_path": scoped["file_path"],
            "source_id": scoped.get("source_id"),
            "full_doc_id": scoped.get("full_doc_id"),
            "matched_full_doc_ids": scoped.get("full_doc_ids", []),
        },
    }

    if payload.target == "node":
        assert payload.node is not None
        neo = await _mutate_node_neo4j(payload.action, payload.node, scoped)
        chunk_sync = await _update_chunk_tables(payload.action, payload.node, scoped)
        relation_sync = {"relations_affected": 0}
        if payload.action == "delete":
            relation_sync = await _delete_node_related_relations(payload.node, scoped)
        result["neo4j"] = neo
        result["lightrag_vdb"] = {
            **chunk_sync,
            **relation_sync,
        }

        if chunk_sync.get("chunks_affected", 0) == 0:
            warnings.append("No rows changed in lightrag_vdb_chunks for this scoped mutation")
        if payload.action == "delete" and relation_sync.get("relations_affected", 0) == 0:
            warnings.append("No relation rows changed in lightrag_vdb_relation(s) for node delete")

    else:
        assert payload.relationship is not None
        neo = await _mutate_relationship_neo4j(payload.action, payload.relationship, scoped)
        relation_sync = await _update_relation_tables(payload.action, payload.relationship, scoped)
        result["neo4j"] = neo
        result["lightrag_vdb"] = relation_sync

        if relation_sync.get("relations_affected", 0) == 0:
            warnings.append("No rows changed in lightrag_vdb_relation(s) for this scoped mutation")

    if result.get("neo4j", {}).get("neo4j_rows", 0) == 0:
        warnings.append("No Neo4j rows matched the scoped mutation filter")

    return result, warnings


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """Mutate nodes/relationships with strict workspace scoping and admin checks."""
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    payload, error_response = parse_request(req, MutateKnowledgeGraphRequest)
    if error_response:
        return error_response

    try:
        await _validate_admin_curate_permission(user_id=user_id, workspace_id=payload.workspace_id)
        mutation_result, warnings = await _apply_mutation(payload)

        response_data: Dict[str, Any] = {
            "mutation": mutation_result,
            "status": "success",
        }
        if warnings:
            response_data["warnings"] = warnings

        return create_success_response(
            message="Knowledge graph mutation completed",
            data=response_data,
            correlation_id=correlation_id,
        )

    except AuthorizationException as exc:
        return create_error_response(
            message=exc.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id,
        )
    except ValidationException as exc:
        return create_error_response(
            message=exc.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )
    except Exception as exc:
        logger.error(
            "Knowledge graph mutation failed",
            error=exc,
            workspace_id=getattr(payload, "workspace_id", None),
            user_id=user_id,
            exc_info=True,
        )
        return create_error_response(
            message="Failed to mutate knowledge graph",
            error_code="MUTATE_KNOWLEDGE_GRAPH_FAILED",
            details={"error": str(exc)},
            status_code=500,
            correlation_id=correlation_id,
        )

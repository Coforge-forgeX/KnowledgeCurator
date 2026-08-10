# Mutate Knowledge Graph API

## Overview

The mutate endpoint updates graph data in a workspace-scoped and permission-checked way.

- Endpoint: `POST /api/v2/kb/graph/mutate`
- Handler: `src/functions/api/mutate_knowledge_graph`
- Auth: Required (`Authorization: Bearer <jwt>`)
- Permission: User must have admin/curator access for the workspace

This endpoint supports all graph mutation operations through one contract:

- Create node
- Update node
- Delete node
- Create relationship
- Update relationship
- Delete relationship

It applies mutations in both:

- Neo4j graph
- LightRAG VDB tables (`lightrag_vdb_chunks`, `lightrag_vdb_relation`, `lightrag_vdb_relations`)

---

## Legacy Tool Mapping

The following legacy operations map to this endpoint:

- `insert_node_to_kg` -> `action=create`, `target=node`
- `insert_edge_to_kg` -> `action=create`, `target=relationship`
- `delete_entity_from_kg` -> `action=delete`, `target=node`
- `delete_relation_from_kg` -> `action=delete`, `target=relationship`
- `edit_entity_in_kg` -> `action=update`, `target=node`
- `edit_relation_in_kg` -> `action=update`, `target=relationship`

---

## Request

### Headers

- `Content-Type: application/json`
- `Authorization: Bearer <jwt-token>`

### Body Schema

```json
{
  "workspace_id": 1017,
  "action": "create | update | delete",
  "target": "node | relationship",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf",
    "source_id": "chunk-optional",
    "full_doc_id": "doc-optional"
  },
  "node": {
    "entity_name": "string",
    "new_entity_name": "string-optional",
    "entity_type": "string-optional",
    "description": "string-optional",
    "source_id": "string-optional",
    "additional_properties": {}
  },
  "relationship": {
    "source": "string",
    "target": "string",
    "relation": "string",
    "new_relation": "string-optional",
    "description": "string-optional",
    "source_id": "string-optional",
    "additional_properties": {}
  }
}
```

### Field Rules

- `workspace_id` must be `> 0`
- `action` must be one of `create`, `update`, `delete`
- `target` must be one of `node`, `relationship`
- `scope.file_path` is required and must belong to indexed data in the given workspace
- If `target=node`, `node` is required and `relationship` must be omitted
- If `target=relationship`, `relationship` is required and `node` must be omitted

Node payload:

- `entity_name` required for create/update/delete
- `new_entity_name` optional (used for rename in update)

Relationship payload:

- `source`, `target`, and `relation` are required for create/update/delete
- `new_relation` optional (used for rename/update)

Scope behavior:

- `scope.source_id` narrows mutation to a chunk/source scope when provided
- `scope.full_doc_id` narrows to a specific indexed doc id when provided
- if `scope.full_doc_id` is provided, it must belong to the workspace + file_path

---

## Response

All responses use the service-standard wrapper.

### Success (`200`)

```json
{
  "success": true,
  "message": "Knowledge graph mutation completed",
  "timestamp": "2026-08-10T13:00:00.000000",
  "correlation_id": "optional-correlation-id",
  "data": {
    "status": "success",
    "mutation": {
      "workspace_id": 1017,
      "action": "update",
      "target": "relationship",
      "scope": {
        "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf",
        "source_id": "chunk-...",
        "full_doc_id": null,
        "matched_full_doc_ids": ["doc-1", "doc-2"]
      },
      "neo4j": {
        "neo4j_rows": 1
      },
      "lightrag_vdb": {
        "relations_affected": 2,
        "chunks_affected": 0
      }
    },
    "warnings": [
      "No rows changed in lightrag_vdb_relation(s) for this scoped mutation"
    ]
  }
}
```

Notes:

- `warnings` appears when mutation is valid but no rows matched in one or more backends
- `neo4j.neo4j_rows` indicates matched/created/updated/deleted rows in Neo4j
- `lightrag_vdb` counts vary by mutation target/action

### Validation Error (`400`)

```json
{
  "success": false,
  "error": "VALIDATION_ERROR",
  "message": "...",
  "timestamp": "...",
  "correlation_id": "..."
}
```

Typical causes:

- invalid payload shape (`node`/`relationship` mismatch for target)
- missing required fields
- `file_path` not indexed in workspace
- `full_doc_id` not associated with workspace + file_path

### Authorization Error (`403`)

```json
{
  "success": false,
  "error": "AUTHORIZATION_ERROR",
  "message": "...",
  "timestamp": "...",
  "correlation_id": "..."
}
```

Typical causes:

- missing/invalid JWT
- user lacks admin/curator permission for workspace

### Server Error (`500`)

```json
{
  "success": false,
  "error": "MUTATE_KNOWLEDGE_GRAPH_FAILED",
  "message": "Failed to mutate knowledge graph",
  "timestamp": "...",
  "correlation_id": "..."
}
```

---

## Examples

### 1) Create Node

```json
{
  "workspace_id": 1017,
  "action": "create",
  "target": "node",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
  },
  "node": {
    "entity_name": "Test Entity A",
    "entity_type": "concept",
    "description": "created via mutate api"
  }
}
```

### 2) Update Node (Rename)

```json
{
  "workspace_id": 1017,
  "action": "update",
  "target": "node",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
  },
  "node": {
    "entity_name": "Test Entity A",
    "new_entity_name": "Test Entity A Updated",
    "description": "renamed via mutate api"
  }
}
```

### 3) Delete Node

```json
{
  "workspace_id": 1017,
  "action": "delete",
  "target": "node",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
  },
  "node": {
    "entity_name": "Test Entity A Updated"
  }
}
```

### 4) Create Relationship

```json
{
  "workspace_id": 1017,
  "action": "create",
  "target": "relationship",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
  },
  "relationship": {
    "source": "Test Entity A Updated",
    "target": "Insured",
    "relation": "relates_to",
    "description": "created relation"
  }
}
```

### 5) Update Relationship

```json
{
  "workspace_id": 1017,
  "action": "update",
  "target": "relationship",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
  },
  "relationship": {
    "source": "Test Entity A Updated",
    "target": "Insured",
    "relation": "relates_to",
    "new_relation": "depends_on",
    "description": "updated relation"
  }
}
```

### 6) Delete Relationship

```json
{
  "workspace_id": 1017,
  "action": "delete",
  "target": "relationship",
  "scope": {
    "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
  },
  "relationship": {
    "source": "Test Entity A Updated",
    "target": "Insured",
    "relation": "depends_on"
  }
}
```

---

## cURL Template

```bash
curl -X POST "http://localhost:8080/api/v2/kb/graph/mutate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <jwt-token>" \
  -d '{
    "workspace_id": 1017,
    "action": "create",
    "target": "node",
    "scope": {
      "file_path": "Other/Demo Instances/1017/Chapter 11_Speciality Coverages.pdf"
    },
    "node": {
      "entity_name": "Test Entity A",
      "entity_type": "concept"
    }
  }'
```

---

## Testing Recommendations

- Prefer passing `scope.source_id` (or `scope.full_doc_id`) for precise mutations.
- After each mutation, verify using `POST /api/v2/kb/graph` with a focused query.
- Check `data.warnings` even on `success=true` to catch no-op updates.
- Use a dedicated test entity/relation name prefix for easy cleanup.

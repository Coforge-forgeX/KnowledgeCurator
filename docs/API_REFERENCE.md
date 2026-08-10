# KB REST Service API Reference

## Scope

This document describes all currently exposed FastAPI routes in the service.

Base prefix for application APIs:

- `/api/v2`

System route:

- `GET /health`

## Common Response Envelope

Most endpoints return the standard envelope:

```json
{
  "success": true,
  "message": "...",
  "timestamp": "2026-08-10T13:00:00.000000",
  "correlation_id": "optional",
  "data": {}
}
```

Error envelope:

```json
{
  "success": false,
  "error": "ERROR_CODE",
  "message": "...",
  "timestamp": "2026-08-10T13:00:00.000000",
  "correlation_id": "optional",
  "details": {}
}
```

## Authentication

- JWT bearer token is expected for API endpoints.
- Header: `Authorization: Bearer <token>`
- Many handlers also validate workspace membership and/or admin-curator permissions.

## Endpoints

---

### 1) Health

- Method: `GET`
- Path: `/health`
- Purpose: Service liveness and runtime metadata.

Response fields:

- `status`
- `service`
- `version`
- `cloud_provider`
- `storage_provider`
- `queue_provider`

---

### 2) Query RAG

- Method: `POST`
- Path: `/api/v2/query`
- Handler: query_rag
- Purpose: Query KB with caching, workspace resolution, and source references.

Request body:

```json
{
  "query": "What is asset management?",
  "workspace_id": 123,
  "mode": "hybrid",
  "history": [],
  "agent_id": 1
}
```

Request params:

- `query` (string, required)
- `workspace_id` (int > 0, required)
- `mode` (optional: naive | local | global | hybrid | mix, default hybrid)
- `history` (optional list of objects)
- `agent_id` (optional int, default 1)

Success response data:

- `final_answer` (string)
- `source` (list of source references)
  - each item includes: `file_id`, `file_name`, optional `citation`
- `requested_mode`
- `effective_mode`

Status codes:

- `200` success
- `400` validation error
- `403` authorization error
- `500` internal error

---

### 3) Source Download URL

- Method: `GET`
- Path: `/api/v2/files/{file_id}/download`
- Handler: query_source_download_url
- Purpose: Generate signed 5-minute download URL for query source references.

Path params:

- `file_id` (string, required)

Success response data:

- `file_id`
- `file_name`
- `download_url`
- `expires_in_seconds`

Status codes:

- `200` success
- `400` invalid/expired file reference
- `403` authorization error
- `500` internal error

---

### 4) KB Index Job

- Method: `POST`
- Path: `/api/v2/kb/index`
- Handler: kb_index
- Purpose: Enqueue a background indexing job by document URL.

Request body:

```json
{
  "workspace_id": 1,
  "document_url": "https://example.com/file.pdf",
  "kb_id": 10
}
```

Request params:

- `workspace_id` (int, required)
- `document_url` (non-empty string, required)
- `kb_id` (optional int)

Success response data:

- `job_id`
- `status` (queued)

Status codes:

- `202` accepted
- `500` queue/internal error

---

### 5) Upload and Index Files

- Method: `POST`
- Path: `/api/v2/documents/upload`
- Handler: upload_and_index
- Purpose: Upload files and asynchronously enqueue indexing tasks.

Request body:

```json
{
  "workspace_id": 123,
  "files": [
    {
      "file_name": "document.pdf",
      "file_content": "<base64>"
    }
  ],
  "idempotency_key": "optional-key"
}
```

Request params:

- `workspace_id` (int > 0, required)
- `files` (list, required, 1-10)
  - `file_name` (required, valid supported extension)
  - `file_content` (required, base64)
- `idempotency_key` (optional string)
- Header alternative: `Idempotency-Key`

Success response data:

- `success`
- `message`
- `workspace_id`
- `total_files`
- `tasks` (list of queued file task info)
- `failed_files` (list)

Status codes:

- `202` accepted
- `400` invalid payload / KB mapping missing
- `404` workspace not found
- `500` upload/index failure

---

### 6) Index Existing Workspace Files

- Method: `POST`
- Path: `/api/v2/workspaces/index-files`
- Handler: index_workspace_files
- Purpose: Queue indexing for all existing blobs under workspace upload path.

Request body:

```json
{
  "workspace_id": 123
}
```

Success response data:

- `success`
- `message`
- `workspace_id`
- `total_blobs_scanned`
- `queued_files`
- `tasks`
- `failed_files`
- `skipped_files`
- `kb_id`

Status codes:

- `200` no files found
- `202` queued successfully
- `400` invalid KG/KB mapping
- `404` workspace not found
- `500` list/queue failure

---

### 7) File Task Status

- Method: `GET`
- Path: `/api/v2/documents/status`
- Handler: file_tasks_status
- Purpose: Fetch status by task IDs or by workspace.

Query params:

- `file_tasks_id` (preferred; supports comma-separated values or JSON list)
  - aliases accepted: `task_ids`, `task_id`
- `workspace_id` (required only when file_tasks_id is not provided)

Response data:

- `mode` (`file_tasks_id` or `workspace`)
- `requested_file_tasks_id`
- `workspace_id`
- `count`
- `statuses` (list with `file_task_id`, `workspace_id`, `file_name`, `file_path`, `status`, timestamps)

Status codes:

- `200` success
- `400` validation error
- `403` authorization error
- `500` internal error

---

### 8) Workspace Documents (Grouped)

- Method: `GET`
- Path: `/api/v2/workspaces/documents`
- Handler: workspace_documents_grouped
- Purpose: Return documents grouped by workspace and linked KBs.

Query params:

- `workspace_id` (int > 0, required)

Response data:

- `workspace_id`
- `group_count`
- `groups` (list)
  - each group includes: `key`, `label`, `kb_id`, `count`, `documents`
  - document entries include: `file_id`, `file_name`, `file_path`, `status`, `file_task_id`, `indexed_at`, `updated_at`

Status codes:

- `200` success
- `403` authorization error
- `500` internal error

---

### 9) Delete Files by File ID / File Path

- Method: `DELETE`
- Path: `/api/v2/files`
- Handler: delete_files_by_id
- Purpose: Delete indexed content and cleanup storage, graph, VDB, and metadata.

Request body:

```json
{
  "workspace_id": 123,
  "file_id": ["token1", "token2"],
  "file_path": ["path/optional/fallback.pdf"]
}
```

Request params:

- `workspace_id` (int > 0, required)
- `file_id` (optional string list)
- `file_path` (optional string list)
- At least one of `file_id` or `file_path` is required

Response data:

- `workspace_id`
- `requested`
- `deleted_count`
- `failed_count`
- `deleted` (per-file cleanup summary)
- `failed` (per-file errors)

Status codes:

- `200` all requested files deleted
- `207` partial success (warnings)
- `400` validation or all requested deletes failed
- `403` authorization error
- `500` internal error

---

### 10) Fetch Filtered Graph Data

- Method: `POST`
- Path: `/api/v2/kb/graph`
- Alias Path: `/api/v2/kb/graph-data`
- Handler: fetch_graph
- Purpose: Retrieve graph context and filter nodes/relationships relevant to query and answer.

Request body:

```json
{
  "query": "What is asset management?",
  "answer": "Asset management is ...",
  "workspace_id": 123,
  "mode": "hybrid",
  "graph_only": false,
  "agent_id": 1
}
```

Request params:

- `query` (string, required)
- `answer` (string, required)
- `workspace_id` (int > 0, required)
- `mode` (optional: naive | local | global | hybrid | mix)
- `graph_only` (optional bool; bypass query evidence cache)
- `agent_id` (optional int >= 1)

Response data:

- `graph_data`
  - `entities` list
  - `relationships` list
  - `metadata`
- `query`
- `workspace_id`
- `graph_only`
- `cached`

Status codes:

- `200` success
- `400` validation error
- `403` authorization error
- `500` internal error

---

### 11) Mutate Knowledge Graph

- Method: `POST`
- Path: `/api/v2/kb/graph/mutate`
- Handler: mutate_knowledge_graph
- Purpose: Create/update/delete nodes or relationships with workspace-scoped validation and dual writes.

Request body (high-level):

```json
{
  "workspace_id": 1017,
  "action": "create | update | delete",
  "target": "node | relationship",
  "scope": {
    "file_path": "indexed/file/path.pdf",
    "source_id": "optional",
    "full_doc_id": "optional"
  },
  "node": {},
  "relationship": {}
}
```

Rules:

- target=node: `node` required and `relationship` omitted
- target=relationship: `relationship` required and `node` omitted

Response data:

- `status`
- `mutation`
  - `workspace_id`, `action`, `target`, `scope`
  - `neo4j` row counts
  - `lightrag_vdb` affected row counts
- optional `warnings`

Status codes:

- `200` success (can still include warnings)
- `400` validation error
- `403` authorization error
- `500` mutation failure

For complete field-by-field contract and examples, see:

- `MUTATE_KNOWLEDGE_GRAPH_API.md`

---

## Quick Endpoint Index

- `GET /health`
- `POST /api/v2/query`
- `GET /api/v2/files/{file_id}/download`
- `POST /api/v2/kb/index`
- `POST /api/v2/documents/upload`
- `POST /api/v2/workspaces/index-files`
- `GET /api/v2/documents/status`
- `GET /api/v2/workspaces/documents`
- `DELETE /api/v2/files`
- `POST /api/v2/kb/graph`
- `POST /api/v2/kb/graph-data`
- `POST /api/v2/kb/graph/mutate`

## Notes

- `POST /api/v2/kb/graph` and `POST /api/v2/kb/graph-data` currently invoke the same handler.
- When `DEBUG=true`, interactive API docs are available at `/docs` and `/redoc`.

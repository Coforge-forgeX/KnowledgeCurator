# KB Backend Fix Notes (2026-07-06)

## Scope

This document summarizes the backend-only fixes applied for KB conversation/search failures.

## Primary Symptoms Observed

1. KB chat returned:
   - `Error: Invalid HTTP/S URL provided for Streamable HTTP.`
2. RAG initialization failed with PostgreSQL config issues:
   - `no PostgreSQL user name specified in startup packet`
3. RAG initialization failed with Neo4j connectivity/routing issues:
   - `Unable to retrieve routing information`
   - `Couldn't connect to localhost:7687`
   - `Database neo4j ... is not available`
4. In some paths, backend errors were collapsed into empty/weak responses.

## Root Causes

1. Internal MCP URL resolution for KB service was fragile when `server_url` was missing/invalid.
2. LightRAG Postgres env mapping used only `LIGHTRAG_POSTGRESQL_DATABASE_*`, while runtime env provided `POSTGRESQL_DATABASE_*`.
3. Neo4j runtime in local setup was unavailable or not reachable on expected URI.
4. Query error payloads were not consistently propagated as readable response text.

## Changes Made

### 1) MCP URL Resolution Hardening

File:
- `KnowledgeCurator/src/kbcurator/utils/mcp_service_client.py`

Changes:
- Added robust server URL resolver/normalizer for internal MCP calls.
- Fallback chain now supports:
  1. explicit `server_url`
  2. `KC_SERVICE_URL`
  3. `KBCURATOR_URL`
  4. safe default `http://127.0.0.1:8002/mcp`
- Added scheme normalization (`http://` prefix when missing).
- Improved logging to print the exact resolved URL used.

Why:
- Prevent invalid transport URL errors in `StreamableHttpTransport`.

---

### 2) LightRAG Postgres Environment Fallbacks

File:
- `KnowledgeCurator/src/kbcurator/tools/ingestion_new.py`

Changes:
- Added env fallbacks so LightRAG runtime vars are populated from either:
  - `LIGHTRAG_POSTGRESQL_DATABASE_*`
  - or `POSTGRESQL_DATABASE_*`

Why:
- Runtime env had standard Postgres variables but LightRAG-specific vars were absent, causing startup packet/user errors.

---

### 3) Neo4j URI and Database Stability for Local Runs

File:
- `KnowledgeCurator/src/kbcurator/tools/ingestion_new.py`

Changes:
- Added Neo4j URI normalization helper:
  - converts `neo4j://localhost...` to `bolt://localhost...` for local single-node scenarios.
- Set stable Neo4j DB default via `NEO4J_DATABASE_NAME` (fallback `neo4j`) instead of per-workspace DB assumptions.
- Added retry/fallback logic in `initialize_rag` for routing/database availability edge cases.

Why:
- Local Neo4j routing and dynamic database availability caused repeated graph init failures.

---

### 4) Graph Storage Fallback When Neo4j Is Down

File:
- `KnowledgeCurator/src/kbcurator/tools/ingestion_new.py`

Changes:
- `initialize_rag` now attempts `Neo4JStorage` first.
- If Neo4j is unreachable/unavailable, automatically falls back to `NetworkXStorage`.

Why:
- Keeps KB query flow operational instead of hard failing when Neo4j is unavailable locally.

---

### 5) Query Error Propagation Improvement

File:
- `KnowledgeCurator/src/kbcurator/utils/mcp_service_client.py`

Changes:
- If `query_rag` returns structured error payload, backend now surfaces it in `response` text.

Why:
- Avoids silent/empty responses and improves debuggability.

## What Was Not Changed

1. No UI component behavior was intentionally modified for these fixes.
2. Existing warnings unrelated to this issue (for example unresolved Windows-only imports in Linux/macOS context) were not part of this change set.

## Operational Notes

1. Restart `KnowledgeCurator` service after these changes.
2. Re-test KB query flow.
3. If fallback is active, logs should indicate graph-storage degradation path rather than hard failure.

## Expected Outcome

1. No more invalid Streamable HTTP URL errors from KB internal MCP calls.
2. LightRAG Postgres should initialize using available environment variables.
3. KB query path should continue even if Neo4j is down (via `NetworkXStorage` fallback).
4. Backend errors should appear as explicit response text, not blank output.

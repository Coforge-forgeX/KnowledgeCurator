# Services Workspace

This folder contains the runtime services for document upload/indexing and knowledge-base query flows.

## Services

### kb-rest-service
Purpose:
- Exposes REST APIs for document upload, indexing orchestration, and KB query/chat.
- Writes uploaded documents to blob storage.
- Pushes indexing jobs to Azure Queue Storage.

Primary responsibilities:
- API/auth/request validation.
- Queue message production for indexing.
- Read/query operations against LightRAG-backed stores.

### indexer-service
Purpose:
- Runs as a background worker.
- Consumes queue messages created by kb-rest-service.
- Downloads documents from blob storage.
- Extracts text and indexes content into LightRAG storages.
- Updates indexing/file-task state in PostgreSQL.

Primary responsibilities:
- Queue polling and retries.
- Document processing and embedding/index pipelines.
- Metadata and status updates.

## Local Architecture

1. Client calls kb-rest-service upload/index API.
2. kb-rest-service stores file in blob container and enqueues job.
3. indexer-service consumes job from queue.
4. indexer-service indexes document and updates task state.

## Prerequisites

- Python 3.10+ (or your team standard).
- Access to local/remote backing services configured in each service `.env`:
  - PostgreSQL
  - Neo4j
  - MongoDB (kb-rest-service)
  - Redis (optional/if enabled)
- Azurite (for local Blob + Queue emulation).

## Run Azurite (Local Storage Emulator)

Run this in a dedicated terminal:

```powershell
azurite --silent --location C:\azurite-data --skipApiVersionCheck
```

Notes:
- This enables local Blob and Queue endpoints.
- Keep this terminal running while both services are running.

## Environment Configuration

Each service uses its own `.env` file:
- `services/kb-rest-service/.env`
- `services/indexer-service/.env`

For local Azurite, ensure both services point to Azurite for storage/queue connection strings.
Typical local emulator connection string:

```text
DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFeqCnf2O==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;
```

Minimum settings to verify:

kb-rest-service:
- `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_QUEUE_STORAGE_CONNECTION_STRING` (as used by your config)
- Blob container names used for uploads
- Queue name (default in your code: `kb-indexing-jobs`)

indexer-service:
- `AZURE_STORAGE_CONNECTION_STRING`
- `INDEXING_QUEUE_NAME` (must match kb-rest-service queue)
- Blob container name settings (must match upload target)

## Start Both Services Locally

Use 3 terminals.

### Terminal 1: Azurite

```powershell
azurite --silent --location C:\azurite-data --skipApiVersionCheck
```

### Terminal 2: kb-rest-service

```powershell
Set-Location "services\kb-rest-service"
python function_app.py
```

Alternative (if you run via Azure Functions host):

```powershell
Set-Location "services\kb-rest-service"
func start
```

### Terminal 3: indexer-service

```powershell
Set-Location "services\indexer-service"
python main.py
```

## Quick Verification

1. Confirm kb-rest-service health endpoint responds.
2. Upload a document through kb-rest-service upload/index API.
3. Confirm a queue message is created in Azurite queue.
4. Watch indexer-service logs for:
- message consumption
- processing/indexing
- final task status update to `indexed`

## Common Local Issues

- Queue name mismatch:
  - Ensure both services use the same indexing queue name.
- Container mismatch:
  - Ensure uploader and indexer resolve the same blob container/path.
- Azurite not running:
  - Both services fail queue/blob operations if Azurite is down.
- Stale local data:
  - Stop services and clear `C:\azurite-data` only if you need a clean slate.

## Suggested Startup Order

1. Start Azurite.
2. Start kb-rest-service.
3. Start indexer-service.
4. Trigger upload/index API.

This order avoids queue/storage connection errors during boot.

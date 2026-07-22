# Retry and Resume Mechanism - Implementation Guide

## 🎯 Overview

The indexer service now includes **checkpoint-based retry and resume** functionality that allows failed indexing jobs to:

1. ✅ **Resume from last successful step** (no re-processing)
2. ✅ **Automatic retry with exponential backoff**
3. ✅ **State persistence** (survives crashes and restarts)
4. ✅ **Cached intermediate results** (avoid re-downloading/re-extracting)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Indexing Pipeline                         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Download │→ │ Extract  │→ │  Index   │→ │ Metadata │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│       ↓             ↓              ↓              ↓          │
│  CHECKPOINT    CHECKPOINT     CHECKPOINT     CHECKPOINT      │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ↓
         ┌─────────────────────────────┐
         │   State Manager             │
         │   - PostgreSQL (primary)    │
         │   - Local files (fallback)  │
         └─────────────────────────────┘
```

---

## 📊 State Machine

```
PENDING
   ↓
DOWNLOADING → [checkpoint] → DOWNLOADED
   ↓
EXTRACTING  → [checkpoint] → EXTRACTED (text cached)
   ↓
INDEXING    → [checkpoint] → INDEXED
   ↓
UPDATING_METADATA → COMPLETED
   ↓
[state deleted]

At any step:
   ↓ [error]
FAILED → RETRYING (if retries left)
   ↓
Resume from last checkpoint
```

---

## 💾 Database Schema

### `indexing_jobs` Table

```sql
CREATE TABLE indexing_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    workspace_id INTEGER NOT NULL,
    document_url TEXT NOT NULL,
    kb_id INTEGER,
    
    -- State tracking
    state VARCHAR(50) NOT NULL,           -- Current state
    checkpoint_data JSONB DEFAULT '{}',   -- Checkpoint data
    
    -- Retry tracking
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

### Checkpoint Data Structure

```json
{
  "file_downloaded": true,
  "file_size": 1048576,
  "content_type": "application/pdf",
  
  "text_extracted": true,
  "extracted_text_path": "document.pdf",
  "extraction_method": "pdfplumber",
  "page_count": 15,
  
  "indexed": true,
  "doc_id": "doc-abc123def456",
  "chunks_processed": 12,
  
  "metadata_updated": true,
  
  "error_message": null,
  "retry_count": 1,
  "last_retry_at": "2026-07-22T10:30:00Z"
}
```

---

## ⚙️ Configuration

Add to `.env`:

```bash
# Retry Configuration
MAX_RETRIES=3                       # Maximum retry attempts (default: 3)
MESSAGE_VISIBILITY_TIMEOUT=300      # Queue message visibility in seconds
MAX_CONCURRENT_JOBS=10              # Concurrent processing limit

# Cache Configuration
INDEXER_CACHE_DIR=./indexer_cache   # Cached extracted text location
INDEXER_STATE_DIR=./indexer_state   # Local state backup location

# PDF Processing (from previous update)
PDF_MIN_TEXT_CHARS=200
PDF_MIN_TEXT_PER_PAGE=100
PDF_PER_PAGE_OCR=true
```

---

## 🔄 Retry Behavior

### Exponential Backoff

Retry delays increase exponentially:

```
Retry 1: 60 seconds  (1 minute)
Retry 2: 120 seconds (2 minutes)
Retry 3: 240 seconds (4 minutes)
```

Formula: `delay = 60 * (2 ^ retry_count)`

### Queue Message Handling

```python
if retry_scheduled:
    # Update message visibility to delay retry
    queue_client.update_message(
        message,
        visibility_timeout=retry_delay_seconds
    )
elif max_retries_exceeded:
    # Delete from queue (moves to poison queue if configured)
    queue_client.delete_message(message)
elif success:
    # Delete from queue
    queue_client.delete_message(message)
else:
    # Leave in queue for automatic retry
    pass
```

---

## 📝 Resume Examples

### Example 1: Failure During Extraction

```
Job starts:
  ✅ Download (2 MB PDF)
  ✅ Checkpoint saved: DOWNLOADED
  ❌ Extraction fails (OCR timeout)

Retry #1 (after 60s):
  ⏭️  Skip download (from checkpoint)
  ✅ Re-download file (no binary cache)
  ✅ Extract with retry
  ✅ Checkpoint saved: EXTRACTED (text cached)
  ✅ Index document
  ✅ Update metadata
  ✅ COMPLETED
```

### Example 2: Failure During Indexing

```
Job starts:
  ✅ Download
  ✅ Extract (text cached to disk)
  ✅ Checkpoint saved: EXTRACTED
  ❌ Indexing fails (LightRAG error)

Retry #1 (after 60s):
  ⏭️  Skip download
  ⏭️  Skip extraction (load cached text)
  ✅ Index document
  ✅ Update metadata
  ✅ COMPLETED
```

### Example 3: Crash During Processing

```
Job starts:
  ✅ Download
  ✅ Extract
  ✅ Checkpoint saved to PostgreSQL
  💥 Service crashes

Service restarts:
  📖 Load state from PostgreSQL
  ℹ️  State: EXTRACTED
  ⏭️  Resume from extraction checkpoint
  📄 Load cached text
  ✅ Index document
  ✅ Update metadata
  ✅ COMPLETED
```

---

## 🧪 Testing

### Test Scenario 1: Simulated Failure

```python
# Add to document_processor_with_retry.py for testing
import os
SIMULATE_FAILURE = os.getenv("SIMULATE_FAILURE_AT")

if SIMULATE_FAILURE == "extraction":
    raise Exception("Simulated extraction failure")
```

Run with:
```bash
SIMULATE_FAILURE_AT=extraction python app.py
```

Expected behavior:
- Job fails at extraction
- Retry scheduled after 60s
- Resumes from checkpoint
- Succeeds on retry

### Test Scenario 2: Manual Resume

```python
# Manually create a failed job state
from core.state_manager import get_state_manager
from shared.indexing_state import IndexingJobState, IndexingState

state = IndexingJobState(
    job_id="test-123",
    workspace_id=1,
    document_url="workspace_1/test.pdf",
    state=IndexingState.EXTRACTED,
    checkpoint={
        "text_extracted": True,
        "extracted_text_path": "./cache/test-123_text.txt",
    }
)

state_manager = get_state_manager()
await state_manager.save_state(state)

# Send queue message with job_id="test-123"
# Should resume from EXTRACTED state
```

---

## 📊 Monitoring

### State Queries

```sql
-- Jobs by state
SELECT state, COUNT(*) 
FROM indexing_jobs 
GROUP BY state;

-- Failed jobs
SELECT job_id, workspace_id, last_error, retry_count
FROM indexing_jobs
WHERE state = 'failed'
ORDER BY updated_at DESC;

-- Jobs being retried
SELECT job_id, workspace_id, retry_count, updated_at
FROM indexing_jobs
WHERE state = 'retrying'
ORDER BY updated_at DESC;

-- Average retries per job
SELECT AVG(retry_count) as avg_retries
FROM indexing_jobs
WHERE state = 'completed';
```

### Logs

Structured logs with context:

```json
{
  "event": "Document processing completed",
  "job_id": "job-abc123",
  "doc_id": "doc-def456",
  "duration_seconds": 45.2,
  "chunks": 12,
  "retry_count": 1,
  "extraction_method": "pdfplumber",
  "timestamp": "2026-07-22T10:30:00Z"
}
```

---

## 🔧 Implementation Files

| File | Purpose |
|------|---------|
| [`src/shared/indexing_state.py`](indexer-service/src/shared/indexing_state.py) | State machine and checkpoint models |
| [`src/core/state_manager.py`](indexer-service/src/core/state_manager.py) | State persistence (PostgreSQL + local files) |
| [`src/workers/document_processor_with_retry.py`](indexer-service/src/workers/document_processor_with_retry.py) | Main processor with retry logic |
| [`migrations/001_create_indexing_jobs_table.sql`](indexer-service/migrations/001_create_indexing_jobs_table.sql) | Database migration |
| [`app.py`](indexer-service/app.py) | Updated queue worker |

---

## ✅ Features

### Implemented

- ✅ State tracking at each pipeline step
- ✅ Checkpoint-based resume
- ✅ Exponential backoff retry
- ✅ Dual persistence (PostgreSQL + local files)
- ✅ Cached extracted text (avoid re-extraction)
- ✅ Queue message visibility management
- ✅ Structured logging with context
- ✅ Max retry limits
- ✅ Crash recovery

### Benefits

- ✅ **No duplicate work** - Resume from where it failed
- ✅ **Cost savings** - Don't re-download or re-OCR on retry
- ✅ **Crash resilient** - Survives service restarts
- ✅ **Automatic recovery** - Self-healing pipeline
- ✅ **Observable** - Full state tracking in database

---

## 🚀 Migration

### Step 1: Run Database Migration

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d forgex_coforge

# Run migration
\i migrations/001_create_indexing_jobs_table.sql
```

### Step 2: Update Configuration

Add retry configs to `.env`:

```bash
MAX_RETRIES=3
INDEXER_CACHE_DIR=./indexer_cache
INDEXER_STATE_DIR=./indexer_state
```

### Step 3: Deploy Updated Service

The updated `app.py` automatically uses the new processor:

```bash
python app.py
```

### Step 4: Monitor

Watch logs for retry behavior:

```bash
# Look for these log events:
# - "Resuming from checkpoint"
# - "Will retry processing"
# - "Loaded cached text"
# - "Job completed successfully" (with retry_count > 0)
```

---

## 💡 Best Practices

1. **Database Monitoring** - Regularly check `indexing_jobs` table for stuck jobs
2. **Cache Cleanup** - The service auto-cleans cache on success, but monitor disk usage
3. **State Cleanup** - Completed jobs are auto-deleted from state table
4. **Retry Limits** - Adjust `MAX_RETRIES` based on your failure patterns
5. **Poison Queue** - Configure Azure Storage Queue to move max-retry-exceeded messages

---

## 🎯 Summary

The retry and resume system ensures:

✅ **Resilience** - Automatic recovery from transient failures  
✅ **Efficiency** - No duplicate work on retry  
✅ **Observability** - Full state tracking  
✅ **Cost Optimization** - Don't re-download or re-OCR  
✅ **Production Ready** - Handles crashes and restarts  

**Every failed job gets up to 3 automatic retries, resuming from the last successful checkpoint!**

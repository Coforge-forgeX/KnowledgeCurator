# Database Migration Guide

## Quick Run (One Command)

```bash
psql -h forgexpostgresql.postgres.database.azure.com \
     -U forgeX \
     -d forgex_coforge \
     -f migrations/001_clean_and_migrate.sql
```

This will:
1. Delete all records with non-integer `workspace_id` (e.g., 'test_workspace')
2. Convert `workspace_id` from VARCHAR to INTEGER
3. Add `file_name` and `full_doc_id` columns
4. Populate the new columns
5. Create indexes

---

## What Changes

### Before Migration
```sql
file_tasks:
├── workspace_id VARCHAR    -- "618", "test_workspace", etc.
├── file_path TEXT
└── (no file_name)
└── (no full_doc_id)
```

### After Migration
```sql
file_tasks:
├── workspace_id INTEGER    -- 618, 734, etc. (cleaned)
├── file_path TEXT
├── file_name VARCHAR       -- "document.pdf"
└── full_doc_id VARCHAR     -- "doc_72_618_abc123..."
```

---

## Full Migration Sequence

Run in order:

```bash
# 1. Clean data and migrate file_tasks
psql ... -f migrations/001_clean_and_migrate.sql

# 2. Create document_metadata tables
psql ... -f migrations/002_create_document_metadata_tables.sql

# 3. Backfill existing data
psql ... -f migrations/003_backfill_document_metadata.sql
```

---

## Update Your Code

### Before (old code):
```python
file_task = FileTask(
    workspace_id="618",  # String
    file_path=blob_path,
)
```

### After (new code):
```python
file_task = FileTask(
    workspace_id=618,           # Integer now
    file_name=file_name,        # Add this
    full_doc_id=generate_doc_id(),  # Add this
    file_path=blob_path,
)
```

---

## What Gets Deleted

**Records with non-integer workspace_id will be DELETED:**
- `workspace_id = 'test_workspace'` → **DELETED**
- `workspace_id = 'abc'` → **DELETED**
- `workspace_id = NULL` → **DELETED**

**Valid records are KEPT:**
- `workspace_id = '618'` → Converted to `618` (INTEGER)
- `workspace_id = '734'` → Converted to `734` (INTEGER)

---

## Rollback (if needed)

```sql
-- No easy rollback since data is deleted!
-- Restore from backup if needed
```

**Recommendation:** Take a backup first:

```bash
pg_dump -h forgexpostgresql.postgres.database.azure.com \
        -U forgeX \
        -d forgex_coforge \
        -t file_tasks > backup_file_tasks.sql
```

---

## Verify After Migration

```sql
-- Check schema
\d file_tasks

-- Should show:
--   workspace_id | integer | NOT NULL
--   file_name    | varchar |
--   full_doc_id  | varchar |

-- Check data
SELECT id, workspace_id, file_name, full_doc_id, status
FROM file_tasks
LIMIT 5;

-- Count records
SELECT COUNT(*) FROM file_tasks;
```

---

## Connection String

```bash
export PGPASSWORD='Postgre@f0rge-X2025'
psql -h forgexpostgresql.postgres.database.azure.com \
     -U forgeX \
     -d forgex_coforge \
     -f migrations/001_clean_and_migrate.sql
```

Or set in `.pgpass` file for convenience.

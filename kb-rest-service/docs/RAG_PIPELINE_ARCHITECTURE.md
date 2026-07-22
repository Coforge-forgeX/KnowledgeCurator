# RAG Pipeline Architecture - Complete Flow

## 🎯 Overview

The RAG (Retrieval-Augmented Generation) pipeline in this system follows a **distributed, queue-based architecture** for scalability and reliability.

**Key Principle**: The kb-rest-service **does NOT perform indexing**. It only:
1. Accepts documents
2. Uploads to blob storage
3. Queues indexing jobs
4. Monitors status

**A separate indexer service** (background worker) performs the actual chunking, embedding, and indexing.

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Frontend/API)                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ POST /api/rag/upload
                                 │ (file_names, file_contents)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        KB-REST-SERVICE                                  │
│                      (Azure Function App)                                │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. Upload Endpoint (upload_and_index_tool)                       │  │
│  │    - Validates request                                            │  │
│  │    - Decodes base64 files                                         │  │
│  │    - Uploads to Azure Blob Storage                               │  │
│  │    - Creates FileTask record (status: pending)                   │  │
│  │    - Sends message to Azure Storage Queue                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  PostgreSQL Tables:                                                      │
│  ├─ file_tasks (status tracking)                                        │
│  └─ document_metadata (indexed docs)                                    │
└────────────┬────────────────────────────────────────────┬───────────────┘
             │                                             │
             │ Stores                                      │ Queues
             ▼                                             ▼
┌─────────────────────────┐               ┌──────────────────────────────┐
│  Azure Blob Storage     │               │  Azure Storage Queue         │
│                         │               │  (kb-indexing-jobs)          │
│  Structure:             │               │                              │
│  workspace_X/           │               │  Message Format:             │
│  ├─ default/            │               │  {                           │
│  │  ├─ doc1.pdf         │               │    "task_id": 123,           │
│  │  ├─ doc2.docx        │               │    "workspace_id": 1,        │
│  │  └─ doc3.txt         │               │    "file_path": "...",       │
│  └─ kb_name/            │               │    "file_name": "doc1.pdf",  │
│     └─ doc4.pdf         │               │    "queued_at": "..."        │
└─────────────────────────┘               │  }                           │
                                          └────────────┬─────────────────┘
                                                       │
                                                       │ Polls/Triggers
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       INDEXER SERVICE                                   │
│                   (Separate Background Worker)                           │
│                   **NOT in kb-rest-service**                             │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 2. Queue Processor                                                │  │
│  │    - Polls Azure Storage Queue                                    │  │
│  │    - Receives indexing job messages                              │  │
│  │    - Downloads file from Blob Storage                            │  │
│  │    - Updates FileTask (status: processing)                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 3. Document Processing Pipeline                                   │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 3.1. Text Extraction                                        │  │  │
│  │  │  - PDF: PyPDF2, pdfplumber                                  │  │  │
│  │  │  - DOCX: python-docx                                        │  │  │
│  │  │  - TXT: direct read                                         │  │  │
│  │  │  - Images: Azure Document Intelligence (OCR)                │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 3.2. Text Chunking (LightRAG)                              │  │  │
│  │  │                                                             │  │  │
│  │  │  Configuration:                                             │  │  │
│  │  │  ├─ chunk_token_size: 1200 tokens                          │  │  │
│  │  │  └─ chunk_overlap_token_size: 100 tokens                   │  │  │
│  │  │                                                             │  │  │
│  │  │  Process:                                                   │  │  │
│  │  │  1. Split text into chunks of ~1200 tokens                 │  │  │
│  │  │  2. Add 100 token overlap between chunks                   │  │  │
│  │  │  3. Preserve context across boundaries                     │  │  │
│  │  │                                                             │  │  │
│  │  │  Example:                                                   │  │  │
│  │  │    Text: "...ABC...DEF...GHI...JKL..."                     │  │  │
│  │  │    Chunk 1: "ABC...DEF...G" (1200 tokens)                  │  │  │
│  │  │    Chunk 2: "...DEF...GHI...J" (overlap + 1200)            │  │  │
│  │  │    Chunk 3: "...GHI...JKL..." (overlap + 1200)             │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 3.3. Entity & Relationship Extraction (LightRAG)           │  │  │
│  │  │                                                             │  │  │
│  │  │  For each chunk:                                            │  │  │
│  │  │  1. Send to Azure OpenAI GPT-4                             │  │  │
│  │  │  2. Extract entities (people, places, concepts)            │  │  │
│  │  │  3. Extract relationships between entities                 │  │  │
│  │  │  4. Create knowledge graph nodes and edges                 │  │  │
│  │  │                                                             │  │  │
│  │  │  Example:                                                   │  │  │
│  │  │    Text: "John works at Microsoft"                         │  │  │
│  │  │    Entities: [John, Microsoft]                             │  │  │
│  │  │    Relationship: John --[works_at]--> Microsoft            │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 3.4. Embedding Generation                                  │  │  │
│  │  │                                                             │  │  │
│  │  │  Model: Ollama mxbai-embed-large                           │  │  │
│  │  │  Dimensions: 1024                                           │  │  │
│  │  │                                                             │  │  │
│  │  │  Process:                                                   │  │  │
│  │  │  1. Send each chunk to Ollama                              │  │  │
│  │  │  2. Generate 1024-dim vector                               │  │  │
│  │  │  3. Vector represents semantic meaning                     │  │  │
│  │  │                                                             │  │  │
│  │  │  Example:                                                   │  │  │
│  │  │    Text: "The quick brown fox..."                          │  │  │
│  │  │    Vector: [0.23, -0.15, 0.87, ..., 0.42] (1024 dims)     │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 4. Storage Layer                                                  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 4.1. Vector Storage (PostgreSQL + pgvector)                │  │  │
│  │  │                                                             │  │  │
│  │  │  Table: lightrag_vdb_chunks                                │  │  │
│  │  │  ├─ chunk_id (PK)                                          │  │  │
│  │  │  ├─ doc_id (FK to document)                                │  │  │
│  │  │  ├─ workspace (workspace identifier)                       │  │  │
│  │  │  ├─ content (text)                                         │  │  │
│  │  │  ├─ embedding (vector[1024])  ← pgvector type             │  │  │
│  │  │  ├─ metadata (JSON)                                        │  │  │
│  │  │  └─ created_at                                             │  │  │
│  │  │                                                             │  │  │
│  │  │  Index: GIN index on embedding for fast similarity search  │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 4.2. Graph Storage (Neo4j)                                 │  │  │
│  │  │                                                             │  │  │
│  │  │  Stores:                                                    │  │  │
│  │  │  - Nodes: Entities (Person, Company, Concept)              │  │  │
│  │  │  - Edges: Relationships (works_at, located_in)             │  │  │
│  │  │  - Properties: Attributes on nodes and edges               │  │  │
│  │  │                                                             │  │  │
│  │  │  Workspace Isolation:                                       │  │  │
│  │  │  - Each workspace has its own label                        │  │  │
│  │  │  - Labels: workspace_1, workspace_2, etc.                  │  │  │
│  │  │                                                             │  │  │
│  │  │  Example Graph:                                             │  │  │
│  │  │    (John:Person) --[works_at]--> (Microsoft:Company)       │  │  │
│  │  │    (Microsoft) --[located_in]--> (Seattle:Location)        │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 5. Completion                                                     │  │
│  │    - Update FileTask (status: completed/failed)                  │  │
│  │    - Create DocumentMetadata record                              │  │
│  │    - Delete message from queue                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

                                    │
                                    │ Status Updates
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PostgreSQL Database                               │
│                                                                           │
│  file_tasks                     document_metadata                        │
│  ├─ id: 123                     ├─ id: 456                              │
│  ├─ file_name: "doc1.pdf"       ├─ doc_id: "doc-abc123"                 │
│  ├─ workspace_id: 1             ├─ file_name: "doc1.pdf"                │
│  ├─ status: "completed" ✓       ├─ workspace_id: 1                      │
│  ├─ file_size: 1024000          ├─ chunk_count: 15                      │
│  ├─ created_at: ...             ├─ indexed_at: ...                      │
│  └─ updated_at: ...             └─ metadata: {...}                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Query Pipeline (How RAG Works)

When a user queries the knowledge base:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUERY: "Tell me about LightRAG"                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ POST /api/rag/query
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        KB-REST-SERVICE                                  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 1. Query Endpoint (query_rag)                                     │  │
│  │    - Receives user query                                          │  │
│  │    - Loads conversation history (last 5 messages)                │  │
│  │    - Augments query with context                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 2. LightRAG Query Processing                                      │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2.1. Query Embedding                                        │  │  │
│  │  │  - Convert query to 1024-dim vector using Ollama           │  │  │
│  │  │  - Query: "Tell me about LightRAG"                         │  │  │
│  │  │  - Vector: [0.12, -0.34, ..., 0.56]                        │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2.2. Vector Similarity Search (PostgreSQL)                 │  │  │
│  │  │                                                             │  │  │
│  │  │  Query:                                                     │  │  │
│  │  │    SELECT chunk_id, content, metadata                      │  │  │
│  │  │    FROM lightrag_vdb_chunks                                │  │  │
│  │  │    WHERE workspace = 'workspace_1'                         │  │  │
│  │  │    ORDER BY embedding <=> query_vector                     │  │  │
│  │  │    LIMIT 5;                                                 │  │  │
│  │  │                                                             │  │  │
│  │  │  Returns: Top 5 most similar chunks                        │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2.3. Graph Traversal (Neo4j) - Optional                   │  │  │
│  │  │                                                             │  │  │
│  │  │  If mode = "local" or "hybrid":                            │  │  │
│  │  │    1. Find entities related to query                       │  │  │
│  │  │    2. Traverse graph relationships                         │  │  │
│  │  │    3. Get connected knowledge                              │  │  │
│  │  │                                                             │  │  │
│  │  │  Example:                                                   │  │  │
│  │  │    Query mentions "LightRAG"                               │  │  │
│  │  │    → Find (LightRAG) node                                  │  │  │
│  │  │    → Get relationships                                      │  │  │
│  │  │    → (LightRAG)-[uses]->(Neo4j)                           │  │  │
│  │  │    → (LightRAG)-[stores]->(PostgreSQL)                    │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2.4. Context Assembly                                      │  │  │
│  │  │                                                             │  │  │
│  │  │  Combine:                                                   │  │  │
│  │  │  - Top 5 similar chunks from vector search                 │  │  │
│  │  │  - Related entities from graph traversal                   │  │  │
│  │  │  - Conversation history                                     │  │  │
│  │  │                                                             │  │  │
│  │  │  Result: Rich context for LLM                              │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │ 2.5. LLM Generation (Azure OpenAI GPT-4)                  │  │  │
│  │  │                                                             │  │  │
│  │  │  Prompt:                                                    │  │  │
│  │  │    System: You are a helpful assistant...                  │  │  │
│  │  │    Context: [Retrieved chunks and graph data]              │  │  │
│  │  │    History: [Last 5 messages]                              │  │  │
│  │  │    User: Tell me about LightRAG                            │  │  │
│  │  │                                                             │  │  │
│  │  │  Response: Generated answer based on context               │  │  │
│  │  └────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ 3. Response Enhancement                                           │  │
│  │    - Add source citations                                         │  │
│  │    - Add document metadata                                        │  │
│  │    - Add confidence scores                                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Returns
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Response:                                                               │
│  {                                                                        │
│    "answer": "LightRAG is a knowledge graph-based RAG system...",       │
│    "sources": [                                                          │
│      {                                                                   │
│        "doc_id": "doc-abc123",                                           │
│        "file_name": "lightrag_intro.pdf",                               │
│        "chunk_id": 5,                                                    │
│        "content": "LightRAG combines vector search...",                 │
│        "similarity_score": 0.92                                          │
│      }                                                                   │
│    ]                                                                     │
│  }                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Points

### 1. **Separation of Concerns** ✅

| Service | Responsibility |
|---------|----------------|
| **kb-rest-service** | API endpoints, validation, queueing, status tracking |
| **Indexer Service** | Document processing, chunking, embedding, indexing |
| **LightRAG** | Query processing, vector search, graph traversal |

### 2. **No Indexing in kb-rest-service** ⚠️

The kb-rest-service **DOES NOT**:
- ❌ Extract text from files
- ❌ Chunk documents
- ❌ Generate embeddings
- ❌ Insert into vector database
- ❌ Create knowledge graph nodes

It **ONLY**:
- ✅ Uploads files to blob storage
- ✅ Creates FileTask records
- ✅ Queues indexing jobs
- ✅ Monitors status
- ✅ Queries indexed data (via LightRAG)

### 3. **Where Does Indexing Happen?** 📍

Indexing happens in a **separate Indexer Service** (background worker):

**Location**: Not in this repository - needs to be created or already exists separately

**Trigger**: Azure Function Queue Trigger or Azure Container Apps Job

**Example Structure**:
```python
# indexer-service/function_app.py

@app.queue_trigger(
    arg_name="msg",
    queue_name="kb-indexing-jobs",
    connection="AzureWebJobsStorage"
)
async def indexer_worker(msg: func.QueueMessage):
    """Process indexing job from queue"""
    
    # 1. Parse message
    job = json.loads(msg.get_body().decode())
    task_id = job["task_id"]
    workspace_id = job["workspace_id"]
    file_path = job["file_path"]
    
    # 2. Download file from blob
    blob_client = BlobServiceClient.from_connection_string(...)
    file_bytes = blob_client.get_blob_client(...).download_blob().readall()
    
    # 3. Extract text
    text = extract_text(file_bytes, file_path)
    
    # 4. Initialize LightRAG for workspace
    working_dir = f"lightrag_data/workspace_{workspace_id}"
    rag = LightRAG(working_dir=working_dir, ...)
    
    # 5. Index document (chunking, embedding, graph creation all happen here)
    await rag.ainsert(text)
    
    # 6. Update status
    update_file_task_status(task_id, "completed")
    
    # 7. Create document metadata
    create_document_metadata(workspace_id, doc_id, ...)
```

### 4. **LightRAG Chunking Configuration** 📐

Configured in [lightrag_service.py:159](d:\OneDrive - Coforge Limited\Desktop\forgex\poly\Kb\kb-rest-service\src\core\lightrag_service.py:159):

```python
self._rag = LightRAG(
    chunk_token_size=1200,           # Each chunk is ~1200 tokens
    chunk_overlap_token_size=100,    # 100 tokens overlap between chunks
)
```

**Why overlap?**
- Preserves context across chunk boundaries
- Ensures related information isn't split
- Improves retrieval accuracy

### 5. **Storage Backend** 💾

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Store** | PostgreSQL + pgvector | Fast similarity search on embeddings |
| **Graph Store** | Neo4j | Entity relationships and graph traversal |
| **Blob Store** | Azure Blob Storage | Raw document files |
| **Metadata** | PostgreSQL | Document and task metadata |
| **Messages** | MongoDB | Conversation history |

---

## 🔄 Complete Flow Summary

### Upload Flow:
1. User uploads file → kb-rest-service API
2. kb-rest-service uploads to blob storage
3. kb-rest-service creates FileTask (pending)
4. kb-rest-service queues message
5. **Indexer service** picks up message
6. **Indexer service** processes document:
   - Extracts text
   - **Chunks into 1200-token pieces with 100-token overlap**
   - **Generates embeddings (1024-dim vectors)**
   - **Stores chunks in PostgreSQL**
   - **Creates knowledge graph in Neo4j**
7. **Indexer service** updates FileTask (completed)

### Query Flow:
1. User sends query → kb-rest-service API
2. kb-rest-service embeds query
3. LightRAG searches PostgreSQL for similar chunks
4. LightRAG traverses Neo4j graph (if hybrid/local mode)
5. LightRAG assembles context
6. Azure OpenAI generates answer
7. kb-rest-service returns answer + sources

---

## 📚 References

- **LightRAG Service**: [lightrag_service.py](d:\OneDrive - Coforge Limited\Desktop\forgex\poly\Kb\kb-rest-service\src\core\lightrag_service.py)
- **RAG Service**: [rag_service.py](d:\OneDrive - Coforge Limited\Desktop\forgex\poly\Kb\kb-rest-service\src\services\rag_service.py)
- **Queue Helper**: [queue_helpers.py](d:\OneDrive - Coforge Limited\Desktop\forgex\poly\Kb\kb-rest-service\src\helpers\queue_helpers.py)
- **Database Models**: [database.py](d:\OneDrive - Coforge Limited\Desktop\forgex\poly\Kb\kb-rest-service\src\core\database.py)

---

## ✅ To Answer Your Question:

**Q: Are we doing the indexing in this service or is it done in the indexer service?**

**A**: **The indexing is done in a separate Indexer Service**, NOT in kb-rest-service.

**kb-rest-service** is purely an API gateway that:
- Accepts requests
- Uploads to blob
- Queues jobs
- Monitors status
- Queries indexed data

**Indexer Service** (separate background worker) does:
- Text extraction
- Chunking (1200 tokens with 100 overlap)
- Embedding generation (1024-dim vectors)
- Vector storage (PostgreSQL)
- Graph creation (Neo4j)

This follows the **microservices pattern** for scalability and separation of concerns.

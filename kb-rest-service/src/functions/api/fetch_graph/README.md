# Fetch Graph Data Endpoint

## Overview

The `/api/v2/kb/graph-data` endpoint fetches filtered graph data that is relevant to a specific query and answer. It uses LLM to validate which nodes are actually related to the answer, providing a cleaner and more focused graph visualization.

## Features

- **Redis Caching**: Results are cached for 30 minutes to improve performance
- **LLM Filtering**: Uses Azure OpenAI to intelligently filter only relevant entities and relationships
- **Workspace Access Control**: Validates user-workspace membership before processing
- **Reuses query_rag Logic**: Leverages existing LightRAG infrastructure with `only_context` behavior

## Endpoint

```
POST /api/v2/kb/graph-data
```

## Authentication

Requires Bearer token in Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## Request Body

```json
{
  "query": "What is asset management?",
  "answer": "Asset management is the process of developing, operating, maintaining...",
  "workspace_id": 123,
  "mode": "hybrid"
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | User's original question (1-5000 chars) |
| `answer` | string | Yes | Generated answer for filtering (1-50000 chars) |
| `workspace_id` | integer | Yes | Workspace ID (must be > 0) |
| `mode` | string | No | Query strategy: "hybrid" (default), "local", "global", "naive", or "mix" |

## Response

### Success (200)

```json
{
  "success": true,
  "message": "Graph data retrieved successfully",
  "data": {
    "graph_data": {
      "entities": [
        {
          "entity_name": "Asset Management",
          "entity_type": "Concept",
          "description": "Process of managing financial assets",
          "source_id": "chunk_1",
          "file_path": "documents/asset_guide.pdf"
        }
      ],
      "relationships": [
        {
          "source": "Asset Management",
          "target": "Portfolio",
          "relation": "manages",
          "description": "Asset management manages portfolios",
          "source_id": "chunk_1"
        }
      ],
      "metadata": {
        "total_entities": 1,
        "total_relationships": 1,
        "original_entity_count": 50,
        "original_relationship_count": 120
      }
    },
    "query": "What is asset management?",
    "workspace_id": 123,
    "cached": false
  },
  "correlation_id": "abc-123-def",
  "timestamp": "2026-08-09T10:30:00.000Z"
}
```

### Error Responses

#### 400 - Validation Error
```json
{
  "success": false,
  "error": "VALIDATION_ERROR",
  "message": "Query cannot be empty or whitespace",
  "correlation_id": "abc-123-def",
  "timestamp": "2026-08-09T10:30:00.000Z"
}
```

#### 403 - Authorization Error
```json
{
  "success": false,
  "error": "AUTHORIZATION_ERROR",
  "message": "You are not authorized to access workspace 123",
  "correlation_id": "abc-123-def",
  "timestamp": "2026-08-09T10:30:00.000Z"
}
```

#### 500 - Internal Error
```json
{
  "success": false,
  "error": "INTERNAL_ERROR",
  "message": "An error occurred while fetching graph data",
  "details": {
    "error": "Connection timeout"
  },
  "correlation_id": "abc-123-def",
  "timestamp": "2026-08-09T10:30:00.000Z"
}
```

## Flow

1. **Check Cache**: First checks Redis for existing filtered graph data based on workspace_id, query, answer, and mode
2. **Validate Access**: Validates user-workspace membership and permissions
3. **Fetch Context**: If not cached, fetches context from LightRAG (reuses query_rag service)
4. **Parse Graph**: Extracts entities and relationships from LightRAG context
5. **Filter with LLM**: Calls Azure OpenAI to identify which entities are relevant to the answer
6. **Filter Relationships**: Only keeps relationships connecting relevant entities
7. **Cache Result**: Saves filtered graph to Redis with 30-minute TTL
8. **Return**: Returns filtered graph data with metadata

## Implementation Details

### Cache Key Format
```
graph_filtered:{workspace_id}:{hash}
```
Where `hash` is SHA256 of `workspace_id|mode|normalized_query|normalized_answer` (first 24 chars)

### LLM Filtering Prompt

The endpoint sends a prompt to Azure OpenAI that:
- Lists all entities with their types and descriptions
- Provides the generated answer
- Asks the LLM to identify relevant entity indices
- Returns only entities mentioned or closely related to the answer

### Performance

- **Cache Hit**: ~10-50ms (Redis lookup)
- **Cache Miss**: ~2-5 seconds (LightRAG + LLM filtering)
- **Typical Reduction**: 60-80% fewer entities/relationships compared to raw graph

## Usage Example

```python
import requests

url = "https://your-api-host/api/v2/kb/graph-data"
headers = {
    "Authorization": "Bearer your-jwt-token",
    "Content-Type": "application/json"
}
payload = {
    "query": "What is portfolio diversification?",
    "answer": "Portfolio diversification is a risk management strategy that mixes...",
    "workspace_id": 42,
    "mode": "hybrid"
}

response = requests.post(url, json=payload, headers=headers)
data = response.json()

if data["success"]:
    entities = data["data"]["graph_data"]["entities"]
    relationships = data["data"]["graph_data"]["relationships"]
    print(f"Found {len(entities)} relevant entities")
else:
    print(f"Error: {data['message']}")
```

## Notes

- The endpoint reuses the same workspace resolution and access control as `query_rag`
- Graph data is fetched using the same LightRAG configuration as regular queries
- LLM filtering uses Azure OpenAI with low temperature (0.1) for deterministic results
- Results are cached separately from `query_rag` results to avoid cache bloat
- The cache key includes the answer text, so different answers to the same query get different filtered graphs

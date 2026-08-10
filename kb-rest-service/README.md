# KB REST Service

**Knowledge Base REST API Service** for the forgeX platform - A microservice for managing knowledge base operations using LightRAG.

## 📋 Overview

The KB REST Service is a standalone microservice extracted from the monolithic KnowledgeCurator service. It provides RESTful APIs for querying knowledge bases, managing documents, and orchestrating indexing operations through Azure Queue Storage.

### Key Features

- ✅ **LightRAG Integration** - Query and manage knowledge bases with hybrid search capabilities
- ✅ **Async Operations** - Full async/await support for optimal performance  
- ✅ **Multiple Storage Backends** - PostgreSQL, MongoDB, Redis, Neo4j, and Azure Blob Storage
- ✅ **Queue-Based Indexing** - Asynchronous document indexing via Azure Queue Storage
- ✅ **JWT Authentication** - Secure API endpoints with JWT token authentication
- ✅ **Structured Logging** - Human-readable console logs for dev, JSON logs for production with correlation ID tracking
- ✅ **Multi-Cloud Container** - Containerized deployment for Azure, AWS, GCP, or Kubernetes

## 🏗️ Architecture

### Service Responsibilities

```
┌─────────────────────────────────────────────────────────────┐
│                     KB REST Service                         │
│                                                             │
│  📥 Query API (Synchronous)                                │
│     - Receives user queries                                 │
│     - Directly accesses LightRAG for read operations       │
│     - Returns answers with sources/chunks                   │
│                                                             │
│  📤 Document Upload API                                     │
│     - Receives document upload requests                     │
│     - Uploads to Azure Blob Storage                         │
│     - Pushes indexing jobs to Azure Queue                   │
│                                                             │
│  📊 Status & Management APIs                               │
│     - Document listing and management                       │
│     - Indexing status tracking                             │
│     - Workspace management                                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ├─→ Azure Queue (indexing jobs)
                           │
                           ├─→ LightRAG (direct read access)
                           │
                           └─→ Azure Blob Storage
                           
┌─────────────────────────────────────────────────────────────┐
│              Indexer Service (Separate)                     │
│  - Consumes queue messages                                  │
│  - Performs document indexing                               │
│  - Writes to LightRAG storage                               │
└─────────────────────────────────────────────────────────────┘
                           │
                           └─→ Shared LightRAG Storage
                               (Neo4j + Vector DB)
```

### Project Structure

```
kb-rest-service/
├── src/
│   ├── core/                      # Core modules
│   │   ├── auth.py               # JWT authentication
│   │   ├── config.py             # Configuration management
│   │   ├── database.py           # Async SQLAlchemy models
│   │   ├── exceptions.py         # Custom exceptions
│   │   ├── lightrag_service.py   # LightRAG integration
│   │   ├── logging.py            # Structured logging
│   │   ├── middleware.py         # Azure Functions middleware
│   │   ├── neo4j_driver.py       # Neo4j connection
│   │   └── redis.py              # Redis management
│   │
│   ├── adapters/                 # Platform adapters (FastAPI)
│   │   └── fastapi_adapter.py   # Request/response wrappers
│   │
│   ├── functions/                # API endpoint handlers
│   │   └── api/
│   │       ├── kb_query/         # Query knowledge base
│   │       ├── kb_index/         # Index documents
│   │       ├── kb_chat/          # Chat interface
│   │       └── ...
│   │
│   ├── shared/                   # Shared utilities
│   │   ├── messages.py           # Message constants
│   │   ├── payloads.py           # Pydantic models
│   │   └── response_utils.py    # Response helpers
│   │
│   └── helpers/                  # Helper modules
│       ├── constants.py          # Application constants
│       └── queue_helpers.py      # Azure Queue operations
│
├── tests/                        # Test suite
├── .env.example                  # Environment variables template
├── pyproject.toml               # Project configuration
├── requirements.txt              # Python dependencies
├── main.py                      # FastAPI application entry point
├── Dockerfile                   # Multi-stage container build
└── docker-compose.yml           # Local development environment
```

## 🚀 Getting Started

### Prerequisites

- **Docker & Docker Compose** (recommended for local development)
- OR Python 3.10+ (for non-containerized development)

**Required Services** (auto-configured in docker-compose):
- PostgreSQL database
- MongoDB instance
- Redis server
- Neo4j database
- Azure Storage Account (Blob + Queue) OR compatible alternative
- Azure OpenAI or compatible LLM endpoint
- Ollama (for embeddings)

### Quick Start with Docker (Recommended)

1. **Clone and navigate**
   ```bash
   git clone <repository-url>
   cd kb-rest-service
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the service**
   ```bash
   docker-compose up --build
   ```

4. **Access the service**
   - API: http://localhost:8081
   - Health: http://localhost:8081/health
   - Docs: http://localhost:8081/docs (if DEBUG=true)

### Alternative: Local Python Installation

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the service**
   ```bash
   python main.py
   # OR
   uvicorn main:app --reload --port 8080
   ```

### Configuration

Edit `.env` file with your configuration:

#### Database Configuration
```bash
POSTGRESQL_DATABASE_HOST=localhost
POSTGRESQL_DATABASE_DATABASE=kbcurator
NEO4J_DATABASE_NEO4J_BOLT_URI=bolt://localhost:7687
MONGODB_URI=mongodb://localhost:27017
REDIS_HOST=localhost
```

#### Azure Configuration
```bash
BLOB_STORAGE_CONNECTION_STRING=<your-connection-string>
AZURE_QUEUE_STORAGE_CONNECTION_STRING=<your-connection-string>
```

#### LightRAG Configuration
```bash
OLLAMA_MODEL_BASE_URL=http://localhost:11434
OLLAMA_MODEL_EMBEDDING_MODEL=nomic-embed-text
AZURE_OPENAI_LLM_MODEL_API_KEY=<your-api-key>
AZURE_OPENAI_LLM_MODEL_API_BASE=https://your-resource.openai.azure.com/
```

### Running Locally

#### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```

#### Using Python directly
```bash
# Development mode with auto-reload
uvicorn main:app --reload --port 8080

# OR using the main.py script
python main.py
```

The service will be available at:
- Docker Compose: `http://localhost:8081`
- Direct Python: `http://localhost:8080`

## Serverless Entrypoints

The service now includes thin runtime adapters so the same business handlers can run on all three providers:

- Azure Functions: `function_app.py` with function `main`
- AWS Lambda: `aws_lambda_handler.py` with function `lambda_handler`
- GCP Cloud Functions: `gcp_function_main.py` with function `entrypoint`

All entrypoints delegate to shared routing in `src/adapters/cloud_function_adapter.py`, which maps HTTP method + path to the existing handler registry.

### Route Support (Shared Across Providers)

- `POST /api/v2/kb/query` -> `kb_query`
- `POST /api/v2/kb/chat` -> `kb_chat`
- `POST /api/v2/kb/index` -> `kb_index`
- `POST /api/v2/documents/upload` -> `upload_and_index`
- `GET /api/v2/documents/status` -> `file_tasks_status`
- `DELETE /api/v2/files` -> `delete_files_by_id`
- `POST /api/v2/kb/graph` -> `get_knowledge_graph`
- `POST /api/v2/llm/route` -> `llm_route`
- `GET /health` -> health response

### Provider Dependency Notes

- Azure Functions runtime requires `azure-functions` (already in dependencies).
- AWS Lambda runtime with SQS queue provider requires `boto3`.
- GCP Function runtime with GCP storage provider requires `google-cloud-storage`.

Install extras when needed:

```bash
# AWS storage/queue support
pip install "kb-rest-service[aws]"

# GCP storage support
pip install "kb-rest-service[gcp]"

# All cloud providers
pip install "kb-rest-service[all-clouds]"
```

## 📚 API Documentation

### Authentication

All API endpoints require JWT authentication via Bearer token:

```bash
Authorization: Bearer <your-jwt-token>
```

### Key Endpoints

Detailed endpoint documentation:

- Full API reference (all current endpoints): see [API_REFERENCE.md](API_REFERENCE.md)
- Graph mutation endpoint (`POST /api/v2/kb/graph/mutate`): see [MUTATE_KNOWLEDGE_GRAPH_API.md](MUTATE_KNOWLEDGE_GRAPH_API.md)

#### Query Knowledge Base
```http
POST /query-kb
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "What is LightRAG?",
  "mode": "hybrid",
  "workspace_id": 1,
  "only_need_context": false
}
```

**Response:**
```json
{
  "success": true,
  "answer": "LightRAG is...",
  "sources": [...],
  "retrieved_chunks": [...],
  "timestamp": "2026-07-21T12:00:00Z",
  "correlation_id": "abc-123-def"
}
```

#### Upload Document
```http
POST /upload-document
Content-Type: application/json
Authorization: Bearer <token>

{
  "file_content": "base64_encoded_content",
  "filename": "document.pdf",
  "workspace_id": 1,
  "metadata": {
    "source": "manual",
    "author": "John Doe"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Document uploaded and queued for indexing",
  "task_id": "abc123",
  "timestamp": "2026-07-21T12:00:00Z",
  "correlation_id": "abc-123-def"
}
```

#### API Documentation
When running with `DEBUG=true`, interactive API documentation is available at:
- Swagger UI: `http://localhost:8081/docs`
- ReDoc: `http://localhost:8081/redoc`

## 🔧 Development

### Code Quality

The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Linting & Formatting

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/test_lightrag_service.py
```

## 📦 Deployment

This service is containerized and can be deployed to any platform that supports Docker containers.

### Local Docker Deployment

```bash
# Build image
docker build -t kb-rest-service:latest .

# Run container
docker run -d \
  --name kb-rest-service \
  -p 8080:8080 \
  --env-file .env \
  kb-rest-service:latest
```

### Cloud Deployment

See **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** for detailed platform-specific instructions:

- **Azure Container Apps** - Managed containers on Azure
- **AWS App Runner** - Fully managed container deployment
- **GCP Cloud Run** - Serverless container platform
- **Kubernetes** - Deploy to any Kubernetes cluster (AKS, EKS, GKE)

Quick example for Azure Container Apps:

```bash
# Build and push to Azure Container Registry
az acr build --registry <acr-name> \
  --image kb-rest-service:v1.0.0 .

# Deploy to Container Apps
az containerapp create \
  --name kb-rest-service \
  --resource-group <rg-name> \
  --environment <env-name> \
  --image <acr-name>.azurecr.io/kb-rest-service:v1.0.0 \
  --target-port 8080 \
  --ingress external
```

See deployment guide for complete instructions and other platforms.

## 🔐 Security

- **JWT Authentication**: All endpoints protected with JWT tokens
- **Token Revocation**: Redis-based token revocation support
- **Input Validation**: Pydantic models for request validation
- **CORS Configuration**: Configurable CORS policies
- **Security Headers**: Automatic security headers on all responses
- **Rate Limiting**: Configurable rate limiting per endpoint

## 📊 Monitoring

### Logging

The service supports two logging formats via the `LOG_FORMAT` environment variable:

**Console Format (Development)**
```bash
LOG_FORMAT=console  # Human-readable with colors
```

Output example:
```
2026-07-21T12:00:00Z [info] Query executed successfully    app=kb-rest-service correlation_id=abc-123-def-456 user_id=42
```

**JSON Format (Production)**
```bash
LOG_FORMAT=json  # Structured for log aggregation
```

Output example:
```json
{
  "app": "kb-rest-service",
  "environment": "production",
  "level": "info",
  "message": "Query executed successfully",
  "correlation_id": "abc-123-def-456",
  "user_id": 42,
  "query_length": 50,
  "timestamp": "2026-07-21T12:00:00Z"
}
```

**Reducing Log Noise**

In development mode, third-party library logs are automatically reduced to WARNING level to minimize clutter. Adjust `LOG_LEVEL` for your needs:
```bash
LOG_LEVEL=DEBUG    # Verbose (all logs)
LOG_LEVEL=INFO     # Normal (default)
LOG_LEVEL=WARNING  # Warnings and errors only
```

### Health Checks

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "kb-rest-api",
  "version": "1.0.0",
  "cloud_provider": "azure"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

Copyright © 2026 Coforge Limited. All rights reserved.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Contact the team at [kartik.jain@coforge.com](mailto:kartik.jain@coforge.com)

## 🗺️ Roadmap

- [ ] GraphQL API support
- [ ] Streaming responses for long queries
- [ ] Advanced caching strategies
- [ ] Multi-language support
- [ ] Batch operations API
- [ ] Webhooks for indexing events

---

**Built with ❤️ by the forgeX Team**

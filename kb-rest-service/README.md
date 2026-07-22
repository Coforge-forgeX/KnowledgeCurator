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
- ✅ **Structured Logging** - JSON-formatted logs with correlation ID tracking
- ✅ **Azure Functions** - Serverless deployment with Azure Functions v1 programming model

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
│   ├── functions/                # Azure Functions (API endpoints)
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
└── function_app.py              # Azure Functions app entry point
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- MongoDB instance
- Redis server
- Neo4j database
- Azure Storage Account (Blob + Queue)
- Azure OpenAI or compatible LLM endpoint
- Ollama (for embeddings)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kb-rest-service
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration
   ```

5. **Initialize databases**
   ```bash
   # Run database migrations (if applicable)
   python scripts/init_db.py
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

#### Development Server
```bash
func start
```

#### With Hot Reload
```bash
func start --python
```

The service will be available at `http://localhost:7071/api/`

## 📚 API Documentation

### Authentication

All API endpoints require JWT authentication via Bearer token:

```bash
Authorization: Bearer <your-jwt-token>
```

### Key Endpoints

#### Query Knowledge Base
```http
POST /api/kb-query
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "What is LightRAG?",
  "mode": "hybrid",
  "workspace_id": 1
}
```

**Response:**
```json
{
  "success": true,
  "answer": "LightRAG is...",
  "sources": [...],
  "retrieved_chunks": [...],
  "timestamp": "2026-07-21T12:00:00Z"
}
```

#### Index Document
```http
POST /api/kb-index
Content-Type: application/json
Authorization: Bearer <token>

{
  "text": "Document content here...",
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
  "message": "Document queued for indexing",
  "queue_message_id": "abc123",
  "timestamp": "2026-07-21T12:00:00Z"
}
```

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

### Azure Functions Deployment

1. **Build and package**
   ```bash
   func azure functionapp publish <function-app-name>
   ```

2. **Configure environment variables in Azure**
   ```bash
   az functionapp config appsettings set \
     --name <function-app-name> \
     --resource-group <resource-group> \
     --settings @.env
   ```

### Docker Deployment (Alternative)

```bash
# Build image
docker build -t kb-rest-service:latest .

# Run container
docker run -p 8080:80 \
  --env-file .env \
  kb-rest-service:latest
```

## 🔐 Security

- **JWT Authentication**: All endpoints protected with JWT tokens
- **Token Revocation**: Redis-based token revocation support
- **Input Validation**: Pydantic models for request validation
- **CORS Configuration**: Configurable CORS policies
- **Security Headers**: Automatic security headers on all responses
- **Rate Limiting**: Configurable rate limiting per endpoint

## 📊 Monitoring

### Logging

Structured JSON logs with correlation IDs:

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

### Health Checks

```http
GET /api/health
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

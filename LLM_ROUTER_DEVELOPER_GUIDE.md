# LLM Router — Developer Integration Guide

## What is the LLM Router?

The LLM Router lets admins configure LLM providers (Azure OpenAI, Quasar) per workspace-agent from the frontend. Once configured, credentials are stored in MongoDB and any agent backend can initialize the configured LLM and use it — no hardcoded env vars needed.

**Flow:**
```
Admin configures LLM (frontend) → Credentials stored in MongoDB → User toggles provider → Backend loads from DB → Developer calls generate_text_async()
```

---

## How It Works (High Level)

1. **Admin configures** an LLM provider (api_key, endpoint, model) for a workspace via `admin_configure_llm_provider` tool. Credentials get validated and stored in `llm_configs.workspace_configs` in MongoDB.
2. **Admin assigns** the provider to specific agents (agent_ids).
3. **User switches** the active provider for an agent using `switch_llm_provider` (simple toggle, no credentials needed).
4. **Backend reads** the active provider from MongoDB, builds a `ConfigurableAIManager`, and generates text.
5. **Cache is cleared** automatically whenever configuration changes, so the next call picks up new settings.

---

## Developer Integration (Using the Configured LLM in Your Agent)

### Step 1: Install `common_adapters` >= 2.0.3

Your agent's `pyproject.toml` or `requirements.txt` should include `common-adapters>=2.0.3`.

### Step 2: Import and Use (2 lines of code)

```python
from common_adapters.configurableAI import get_configured_llm_manager, invalidate_llm_cache
```

That's it. No kbcurator dependency needed.

### Step 3: Get the Configured LLM Manager

```python
# Returns a cached ConfigurableAIManager pre-loaded from MongoDB
manager = get_configured_llm_manager(workspace_id=782, agent_id=5)
```

### Step 4: Generate Text

```python
# Async usage (recommended in FastAPI/MCP handlers)
async def handle_query(workspace_id: int, agent_id: int, user_prompt: str) -> str:
    manager = get_configured_llm_manager(workspace_id, agent_id)
    response = await manager.generate_text_async(user_prompt)
    return response

# Sync usage (if you're not in an async context)
def handle_query_sync(workspace_id: int, agent_id: int, user_prompt: str) -> str:
    manager = get_configured_llm_manager(workspace_id, agent_id)
    response = manager.generate_text(user_prompt)
    return response
```

### Step 5: Cache Invalidation (only if you switch providers programmatically)

```python
# Cache is auto-cleared when switch_llm_provider / admin_configure_llm_provider is called.
# Only call this manually if you need to force-refresh:
invalidate_llm_cache(workspace_id=782, agent_id=5)
```

---

## Complete Example: Replacing AzureCustomLLM

**Before** (hardcoded env vars):
```python
from devagent.llm.azurecustomllm import AzureCustomLLM

llm = AzureCustomLLM()
response = llm.invoke(sys_prompt="You are helpful.", input="What is Python?", history=[])
```

**After** (using LLM Router):
```python
from common_adapters.configurableAI import get_configured_llm_manager

def get_llm_response(workspace_id: int, agent_id: int, prompt: str) -> str:
    manager = get_configured_llm_manager(workspace_id, agent_id)
    return manager.generate_text(prompt)

# Or if you need system prompt + history (construct the prompt yourself):
def get_llm_response_with_context(workspace_id, agent_id, sys_prompt, user_input, history):
    manager = get_configured_llm_manager(workspace_id, agent_id)
    
    # Build a combined prompt (ConfigurableAIManager sends as single user message)
    full_prompt = f"System: {sys_prompt}\n\n"
    for msg in history:
        full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    full_prompt += f"User: {user_input}"
    
    return manager.generate_text(full_prompt)
```

---

## MongoDB Schema (What Gets Stored)

Collection: `llm_configs.workspace_configs`

```json
{
  "workspace_id": 782,
  "provider_credentials": {
    "azure": {
      "api_key": "sk-...",
      "endpoint": "https://my-resource.openai.azure.com/",
      "model": "gpt-4",
      "deployment_name": "gpt-4",
      "api_version": "2024-12-01-preview",
      "is_active": true,
      "created_at": "2025-01-15T10:00:00Z",
      "created_by": 1
    },
    "quasar": {
      "api_key": "qk-...",
      "endpoint": "https://quasar.example.com/v1/chat/completions",
      "model": "claude-sonnet-4",
      "is_active": true
    }
  },
  "agent_configs": {
    "5": {
      "configured_providers": ["azure", "quasar"],
      "current_provider": "azure"
    },
    "12": {
      "configured_providers": ["azure"],
      "current_provider": "azure"
    }
  }
}
```

---

## API Reference (MCP Tools)

### For Admins

| Tool | Purpose |
|------|---------|
| `admin_configure_llm_provider` | Configure/update LLM credentials for a workspace and assign to agents |
| `admin_list_llm_providers` | List all configured providers and their agent assignments |
| `admin_remove_llm_provider` | Deactivate a non-Azure provider |

### For Users

| Tool | Purpose |
|------|---------|
| `list_available_llm_providers` | See which LLMs are available for your agent |
| `switch_llm_provider` | Toggle the active LLM for an agent (no credentials needed) |
| `test_llm_generation` | Smoke-test the currently active LLM |

---

## Frontend Integration Flow

```
1. Admin opens LLM settings page
2. Admin fills: provider, api_key, endpoint, model, agent_ids
3. Frontend calls → admin_configure_llm_provider (credentials validated automatically)
4. User wants to switch LLM → Frontend calls switch_llm_provider(provider, workspace_id, agent_id)
5. Next query from that agent uses the new LLM (cache auto-cleared)
```

---

## Key Points

- **Credentials are never in env vars** — they live in MongoDB per workspace.
- **Cache is auto-cleared** — when `admin_configure_llm_provider` or `switch_llm_provider` is called, `clear_ai_manager_cache()` runs automatically.
- **Fallback logic** — if an agent has no dedicated config, it inherits the workspace default.
- **Validation on configure** — credentials are tested before being stored (unless `skip_validation=True`).
- **Azure cannot be removed** — it's the system default. Other providers (quasar) can be removed.
- **Supported providers** — `azure`, `quasar` (extend `SUPPORTED_PROVIDERS` in `llm_router_config_store.py` to add more).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No LLM configured" error | Admin needs to run `admin_configure_llm_provider` for your workspace/agent |
| Old LLM still being used after switch | Call `clear_ai_manager_cache(workspace_id, agent_id)` manually |
| Credential validation fails | Check api_key, endpoint, and model are correct. Use `skip_validation=True` to bypass |
| Agent not using router yet | Replace `AzureCustomLLM()` with `build_llm_manager(workspace_id, agent_id)` |

---

## File Reference

| File | Purpose |
|------|---------|
| `common_adapters/configurableAI/manager.py` | `ConfigurableAIManager` — the core class that wraps multiple providers |
| `common_adapters/configurableAI/providers.py` | Provider implementations (Azure, Quasar, OpenAI, GCP) |
| `kbcurator/tools/llm_router_tool.py` | MCP tools exposed to frontend (admin + user tools) |
| `kbcurator/services/llm_router_config_store.py` | MongoDB CRUD for workspace configs |
| `kbcurator/services/agent_llm_configuration_service.py` | Agent-level config service |
| `kbcurator/services/workspace_provider_credentials_service.py` | Credential service |

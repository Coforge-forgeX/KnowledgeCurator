# LLM Router — Developer Integration Guide

## What is the LLM Router?

The LLM Router lets admins configure LLM providers (Azure OpenAI, Quasar) per workspace/agent from the frontend. Credentials are stored in MongoDB. Any agent backend can load the configured LLM and use it — no hardcoded env vars needed.

**Flow:**
```
Admin configures (frontend) → MongoDB stores config → Developer calls get_configured_llm_manager() → generate_text_async()
```

---

## Developer Integration (3 Steps)

### Step 1: Install `common_adapters`

```bash
pip install -e /path/to/forgexpackages
```

### Step 2: Import

```python
from common_adapters.configurableAI import get_configured_llm_manager, invalidate_llm_cache
```

### Step 3: Use

```python
async def handle_query(workspace_id: int, agent_id: int, prompt: str) -> str:
    manager = get_configured_llm_manager(workspace_id, agent_id)
    response = await manager.generate_text_async(prompt)
    return response
```

That's it. The manager is pre-configured with the correct provider/model from MongoDB.

---

## How It Works

```python
manager = get_configured_llm_manager(workspace_id=892, agent_id=1)
```

This:
1. Reads `agent_configs.1` from MongoDB for workspace 892
2. Gets `current_provider` (e.g., `"quasar"`) and `current_model` (e.g., `"claude-sonnet-4"`)
3. Loads credentials from `provider_credentials.quasar`
4. Builds a `ConfigurableAIManager` with the correct endpoint, api_key, model
5. Caches it in memory (subsequent calls return cached instance)

---

## Data Model (Per Agent)

```json
{
  "configured_providers": ["azure", "quasar"],
  "configured_models": {
    "azure": ["gpt-4.1"],
    "quasar": ["claude-sonnet-4", "gpt-5-2-chat"]
  },
  "current_provider": "quasar",
  "current_model": "claude-sonnet-4"
}
```

| Field | Purpose |
|-------|---------|
| `configured_providers` | Which providers this agent has access to |
| `configured_models` | Per-provider list of models available to this agent |
| `current_provider` | The provider used at runtime (for LLM calls) |
| `current_model` | The model used at runtime |

---

## Cache Management

The manager is cached per workspace+agent. When config changes happen (via admin tools), the cache is auto-invalidated.

If you need to manually invalidate (e.g., after direct DB updates):

```python
from common_adapters.configurableAI import invalidate_llm_cache

# Invalidate for a specific agent
invalidate_llm_cache(workspace_id=892, agent_id=1)

# Invalidate all for a workspace
invalidate_llm_cache(workspace_id=892)

# Invalidate everything
invalidate_llm_cache()
```

---

## Switching Providers at Runtime

Users can switch providers via the `switch_llm_provider` MCP tool:

```python
# This is called by the MCP tool, not directly by developers
llm_router_config_store.switch_provider(
    workspace_id=892,
    provider="azure",
    agent_id=1,
    model="gpt-4.1",
    user_id=247,
)
```

After switching, the next call to `get_configured_llm_manager()` returns the new provider.

---

## Config Store API Reference

Located in: `common_adapters.configurableAI.llm_router_config_store`

### Provider Credentials

| Method | Description |
|--------|-------------|
| `get_provider_credentials(workspace_id, provider_name)` | Get credentials for a provider |
| `list_workspace_providers(workspace_id)` | List all active providers |
| `upsert_provider_credentials(...)` | Create/update provider credentials |
| `deactivate_provider_credentials(workspace_id, provider_name)` | Soft-delete a provider |
| `build_config_dict(workspace_id, provider_name, model_override)` | Build runtime config dict |
| `set_model_assignments(workspace_id, provider_name, model_name, agent_ids)` | Set agent→model mapping for UI |
| `remove_model_from_provider(workspace_id, provider_name, model_name)` | Remove a model |

### Agent Configuration

| Method | Description |
|--------|-------------|
| `get_configuration(workspace_id, agent_id)` | Get agent config |
| `get_effective_configuration(workspace_id, agent_id)` | Agent config with workspace fallback |
| `create_or_update_configuration(...)` | Create/update agent config |
| `switch_provider(workspace_id, provider, agent_id, model)` | Switch active provider+model |
| `add_provider(workspace_id, provider, agent_id, model)` | Add a provider to an agent |
| `add_model_to_agent(workspace_id, provider, model, agent_id)` | Add a model to agent's configured_models |
| `bulk_create_agent_configurations(workspace_id, agent_ids, ...)` | Bulk-create on workspace setup |
| `get_workspace_configurations(workspace_id)` | Get all agent configs for a workspace |

---

## Environment Variables Required

```env
MONGODB_DATABASE_URI=mongodb+srv://...   # Required for config store
```

For auto-provisioning on first workspace creation:
```env
AZURE_OPENAI_LLM_MODEL_LLM_MODEL=gpt-4.1
AZURE_OPENAI_LLM_MODEL_API_BASE=https://forgexaiservice.openai.azure.com/
AZURE_OPENAI_LLM_MODEL_API_KEY=...
AZURE_OPENAI_LLM_MODEL_API_VERSION=2024-12-01-preview
```

---

## Important Notes

1. **No kbcurator dependency needed** — `common_adapters` is standalone.
2. **Thread-safe** — MongoDB client uses a singleton with thread lock.
3. **Lazy initialization** — DB connection is only made on first use.
4. **Credential inheritance** — If an agent has no config, falls back to `__workspace_default__`.
5. **Azure always available** — Every workspace auto-provisions azure on creation.

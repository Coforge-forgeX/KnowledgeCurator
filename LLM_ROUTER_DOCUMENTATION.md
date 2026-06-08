# LLM Router Tool Documentation

## Overview

The LLM Router provides admin-managed LLM provider configuration per workspace. Provider credentials and per-agent model selections are stored in MongoDB (`llm_configs.workspace_configs`). Each agent can have multiple providers, each provider can have multiple models, and one provider+model is selected as "current" for runtime use.

## Key Features

- **MongoDB-backed Config**: Credentials, model lists, and agent assignments stored per workspace.
- **Multi-Provider / Multi-Model**: Each agent can have multiple providers configured, each with multiple models.
- **Per-Agent Runtime Selection**: Each agent has a `current_provider` + `current_model` that determines which LLM answers queries.
- **Model Assignments**: Independent many-to-many tracking of which agents are assigned to which models (for UI display).
- **Default Azure Protection**: Azure is the system default — all agents must have it; it cannot be removed.
- **Credential Validation**: Credentials are tested before storage.
- **In-memory Cache**: `ConfigurableAIManager` instances are cached and invalidated on config changes.

## Supported Providers

`azure`, `quasar`

## MongoDB Document Schema

Database: `llm_configs`, Collection: `workspace_configs`

```json
{
  "workspace_id": 892,
  "provider_credentials": {
    "azure": {
      "api_key": "...",
      "endpoint": "https://forgexaiservice.openai.azure.com/",
      "model": "gpt-4.1",
      "api_version": "2024-12-01-preview",
      "deployment_name": "gpt-4.1",
      "available_models": [
        {"model_name": "gpt-4.1", "deployment_name": "gpt-4.1"}
      ],
      "model_assignments": {
        "gpt-4.1": [1, 2, 3, 5, 6]
      },
      "is_active": true
    },
    "quasar": {
      "api_key": "...",
      "endpoint": "https://quasarmarket.coforge.com/...",
      "model": "claude-sonnet-4",
      "available_models": [
        {"model_name": "claude-sonnet-4", "deployment_name": "claude-sonnet-4"},
        {"model_name": "gpt-5-2-chat", "deployment_name": "claude-sonnet-4"}
      ],
      "model_assignments": {
        "claude-sonnet-4": [1, 2],
        "gpt-5-2-chat": [1, 2, 5]
      },
      "is_active": true
    }
  },
  "agent_configs": {
    "__workspace_default__": {
      "configured_providers": ["azure"],
      "configured_models": {"azure": ["gpt-4.1"]},
      "current_provider": "azure",
      "current_model": "gpt-4.1",
      "is_active": true
    },
    "1": {
      "configured_providers": ["azure", "quasar"],
      "configured_models": {
        "azure": ["gpt-4.1"],
        "quasar": ["claude-sonnet-4", "gpt-5-2-chat"]
      },
      "current_provider": "quasar",
      "current_model": "claude-sonnet-4",
      "is_active": true
    }
  }
}
```

### Key Fields

| Field | Location | Purpose |
|-------|----------|---------|
| `provider_credentials.{provider}` | Top-level | Shared credentials for a provider (api_key, endpoint) |
| `available_models` | Provider credentials | All models registered for this provider |
| `model_assignments` | Provider credentials | Which agents are assigned to each model (UI display) |
| `agent_configs.{agent_id}` | Per-agent | Agent's provider/model configuration |
| `configured_providers` | Agent config | List of providers available to this agent |
| `configured_models` | Agent config | Dict: provider → list of models configured for this agent |
| `current_provider` | Agent config | The active provider used at runtime |
| `current_model` | Agent config | The active model used at runtime |

## Available Tools

### Admin-Only Tools

#### 1. `admin_configure_llm_provider`

Create or update a provider configuration and assign models to agents.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `provider` | str | Yes | `'azure'` or `'quasar'` |
| `workspace_id` | int | Yes | Target workspace |
| `model` | str | New: Yes | Model name |
| `endpoint` | str | New: Yes | API endpoint URL |
| `api_key` | str | New: Yes | API key (optional on update) |
| `api_version` | str | No | Azure API version |
| `deployment_name` | str | No | Azure deployment name |
| `agent_ids` | List[int] | No | Agents to assign this model to |
| `set_as_current` | bool | No | Set as active provider for listed agents |
| `skip_validation` | bool | No | Skip credential testing |

**Behavior:**
- New provider: Creates credentials, adds model to `available_models`
- Existing provider: Merges with existing config (only provided fields update)
- `agent_ids`: Saves to `model_assignments` and adds model to each agent's `configured_models`
- Azure provider: Cannot modify agent assignments (all agents must have it)

#### 2. `admin_list_llm_providers`

List all configured providers with per-model rows.

**Parameters:** `workspace_id` (int)

**Returns:** One entry per model with assigned agents:
```json
{
  "configured_providers": [
    {
      "provider": "azure",
      "model": "gpt-4.1",
      "endpoint": "...",
      "api_version": "...",
      "agents_enabled": [{"agent_id": 1, "is_current": true}],
      "available_models": [...]
    },
    {
      "provider": "quasar",
      "model": "claude-sonnet-4",
      "agents_enabled": [{"agent_id": 1, "is_current": true}]
    }
  ]
}
```

#### 3. `admin_remove_llm_provider`

Remove a specific model or deactivate an entire provider.

**Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `workspace_id` | int | Yes | Target workspace |
| `provider` | str | Yes | Provider name |
| `model` | str | No | Specific model to remove (omit = remove entire provider) |

**Constraints:**
- Cannot fully remove `azure` (system default)
- Cannot remove the last provider

### User Tools

#### 4. `switch_llm_provider`

Switch the active provider+model for an agent.

**Parameters:** `workspace_id`, `provider`, `agent_id`, `model` (optional)

**Effect:** Updates `current_provider` and `current_model` in agent_configs.

#### 5. `list_available_llm_providers`

List providers available to an agent with their configured models.

#### 6. `test_llm_generation`

Smoke-test the active LLM with a simple prompt.

---

## Technical Deep Dive — Code Flow & Internals

This section explains how the system works end-to-end for technical leads reviewing the architecture.

### System Components

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Forge-X-Web)                                                       │
│  src/components/UserManagement/LLMConfigManager.jsx  — Table UI               │
│  src/components/UserManagement/LLMConfigDialog.jsx   — Add/Edit modal         │
│  src/config/workspace/workspaceListMcp.jsx           — API layer (MCP calls)  │
└──────────────────┬───────────────────────────────────────────────────────────┘
                   │ MCP Tool Calls (via SSE transport)
                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  BACKEND — KnowledgeCurator (MCP Server)                                      │
│  src/kbcurator/tools/llm_router_tool.py              — Tool handlers          │
│  src/kbcurator/services/                                                      │
│    workspace_provider_credentials_service.py         — Credential facade      │
│    agent_llm_configuration_service.py                — Agent config facade    │
└──────────────────┬───────────────────────────────────────────────────────────┘
                   │ Direct method calls
                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  SHARED LIBRARY — CommonAdapters (forgexpackages)                             │
│  src/common_adapters/configurableAI/                                          │
│    llm_router_config_store.py    — MongoDB CRUD (single source of truth)      │
│    llm_router.py                 — Runtime manager factory + cache             │
│    configurable_ai_manager.py    — Provider abstraction (Azure/OpenAI SDK)     │
└──────────────────┬───────────────────────────────────────────────────────────┘
                   │ PyMongo
                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  MongoDB Atlas — Database: llm_configs, Collection: workspace_configs         │
│  One document per workspace (keyed by workspace_id unique index)              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Flow 1: Admin Adds a Model (e.g., adding `claude-sonnet-4` for KB Curator & PO Agent)

```
Frontend                    Backend (llm_router_tool.py)         Config Store (MongoDB)
────────                    ───────────────────────────          ──────────────────────
1. User fills dialog:       
   provider: "quasar"       
   model: "claude-sonnet-4" 
   agent_ids: [1, 2]        
   endpoint, api_key         
                            
2. callMcpTool(             
   "admin_configure_llm_    
    provider", payload)     
         │                  
         ▼                  
                            3. Check is_admin(user_id, workspace_id)
                            4. Determine: is_new_provider? (check existing creds)
                            5. Merge config: only override provided fields
                            6. Validate credentials (test LLM call)
                            7. upsert_provider_credentials()
                            │                                    ┌─ $set provider_credentials.quasar
                            │                                    │  (api_key, endpoint, model,
                            │                                    │   available_models appended)
                            ▼                                    └──────────────────────────────
                            8. set_model_assignments(             ┌─ Read full model_assignments dict
                               provider="quasar",                │  Add key: "claude-sonnet-4": [1,2]
                               model="claude-sonnet-4",          │  $set entire dict back
                               agent_ids=[1, 2])                 │  (avoids dot-notation issues)
                            │                                    └──────────────────────────────
                            ▼
                            9. For each agent_id in [1, 2]:
                               add_model_to_agent(               ┌─ Read agent_configs.1
                                 provider="quasar",              │  Append "quasar" to configured_providers
                                 model="claude-sonnet-4",        │  Append "claude-sonnet-4" to
                                 agent_id=1,                     │    configured_models.quasar
                                 set_as_current=True)            │  Set current_provider="quasar"
                                                                 │  Set current_model="claude-sonnet-4"
                                                                 │  $set agent_configs.1 = payload
                                                                 └──────────────────────────────
                            10. invalidate_llm_cache()
                            11. Return success response
         │
         ▼
3. fetchLLM() → callMcpTool("admin_list_llm_providers")
4. Re-render table with updated data
```

### Flow 2: Admin Lists Configurations (Table Load)

```
Frontend                    Backend                              Config Store
────────                    ───────                              ────────────
1. getLLMConfigurationList  
   ({workspace_id: 892})    
         │                  
         ▼                  
                            2. list_workspace_providers()         → Read provider_credentials
                            3. get_workspace_configurations()     → Read agent_configs
                            
                            4. For each provider:
                               For each model in available_models:
                                 a. Check model_assignments[model] → agent_ids
                                 b. If model_assignments exists for this model:
                                      Use it directly (new behavior)
                                 c. Else (legacy fallback):
                                      Scan agent_configs for agents with 
                                      this model in configured_models[provider]
                                 d. Build row: {provider, model, endpoint,
                                              agents_enabled, ...}
                            
                            5. Return: configured_providers[] (one row per model)
         │
         ▼
6. Render table: each row = one model
   Columns: Provider | Model | Endpoint | API Version | Agents | Actions
```

### Flow 3: Agent Answers a User Query (Runtime)

```
Any Agent Backend           common_adapters/llm_router.py       Config Store
─────────────────           ─────────────────────────────       ────────────
1. manager = get_configured_llm_manager(
     workspace_id=892, agent_id=1)
         │
         ▼
                            2. Check in-memory cache (_manager_cache)
                               Key: "892:1"
                               If cached → return immediately (O(1))
                            
                            3. Cache miss:
                               a. get_effective_configuration(892, 1)
                                  → Read agent_configs.1 from MongoDB
                                  → Fallback to __workspace_default__ if missing
                               
                               b. Extract:
                                  current_provider = "quasar"
                                  current_model = "claude-sonnet-4"
                                  configured_providers = ["azure", "quasar"]
                               
                               c. For each configured provider:
                                  build_config_dict(892, provider, model_override)
                                  → Reads provider_credentials.{provider}
                                  → Resolves deployment_name from available_models
                                  → Returns: {api_key, endpoint, model, ...}
                               
                               d. manager.configure_provider(provider, config)
                                  → Initializes OpenAI/Azure SDK client
                               
                               e. manager.set_current_provider("quasar")
                               
                               f. Cache: _manager_cache["892:1"] = manager
         │
         ▼
2. response = await manager.generate_text_async(prompt)
   → Uses quasar/claude-sonnet-4 SDK client
   → Returns LLM response text
```

### Flow 4: Workspace Creation (Default Config Bootstrap)

```
user_management_system.py              Config Store
─────────────────────────              ────────────
1. create_workspace()
   → PostgreSQL: insert workspace row
   → Commit transaction

2. Seed Azure credentials:
   → Read env: AZURE_OPENAI_LLM_MODEL_*
   → upsert_provider_credentials(                  $set provider_credentials.azure = {
       provider="azure",                              api_key, endpoint, model: "gpt-4.1",
       model="gpt-4.1", ...)                          available_models: [{...}],
                                                      model_assignments: {} }

3. Create workspace default:
   → create_or_update_configuration(               $set agent_configs.__workspace_default__ = {
       agent_id=None,                                 configured_providers: ["azure"],
       configured_providers=["azure"],                configured_models: {azure: ["gpt-4.1"]},
       current_provider="azure",                      current_provider: "azure",
       current_model="gpt-4.1")                       current_model: "gpt-4.1" }

4. Bulk-create agent configs:
   → bulk_create_agent_configurations(             For each agent_id in [1,2,3,5,6]:
       agent_ids=[1,2,3,5,6],                       $set agent_configs.{id} = {
       configured_providers=["azure"],                configured_providers: ["azure"],
       current_provider="azure")                      configured_models: {azure: ["gpt-4.1"]},
                                                      current_provider: "azure",
                                                      current_model: "gpt-4.1" }

5. Set model_assignments:
   → set_model_assignments(                        Read model_assignments dict
       provider="azure",                           Set: {"gpt-4.1": [1,2,3,5,6]}
       model="gpt-4.1",                            $set provider_credentials.azure.model_assignments
       agent_ids=[1,2,3,5,6])
```

### Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Single MongoDB document per workspace** | Avoids joins; all config for a workspace is in one atomic read. Indexed by `workspace_id` (unique). |
| **`model_assignments` in provider_credentials** | Decouples UI display (which agents show under a model) from runtime config. Allows many-to-many without overwriting. |
| **`configured_models` in agent_configs** | Tracks which models an agent has access to (for switching). Separate from what's currently active. |
| **`current_provider` + `current_model`** | Single point of truth for runtime — no ambiguity about which LLM an agent uses. |
| **Read-modify-write for model_assignments** | MongoDB dot notation breaks on model names with dots (e.g., `gpt-4.1`). Full dict read/write avoids this. |
| **Azure is immutable default** | Guarantees every agent always has a fallback LLM. Prevents accidental lockout. |
| **In-memory cache with explicit invalidation** | Avoids MongoDB round-trip on every LLM call. Cache key: `{workspace_id}:{agent_id}`. |
| **Credential validation before storage** | Prevents storing broken configs that would cause runtime failures. |
| **Service layer (thin facades)** | `workspace_provider_credentials_service` and `agent_llm_configuration_service` decouple KnowledgeCurator from `common_adapters` internals. |

### Potential Optimization Areas

| Area | Current State | Potential Improvement |
|------|--------------|----------------------|
| **MongoDB reads in list** | `admin_list` reads full doc + iterates all agents | Could use aggregation pipeline for projection |
| **Credential validation** | Makes a real LLM call on every save | Could cache validated credentials for N minutes |
| **Cache granularity** | One cache entry per workspace:agent | Consider LRU eviction for large deployments |
| **Bulk agent updates** | Iterates and writes individually | Could batch into single `$set` with multiple agent keys |
| **model_assignments write** | Read-modify-write (2 DB ops) | Could use MongoDB array operators if dot issue is solved differently |
| **Available models list** | Only grows (never cleaned up on model removal from all agents) | `remove_model_from_provider` handles this, but orphan entries possible |

### File Map

| File | Role |
|------|------|
| `forgexpackages/src/common_adapters/configurableAI/llm_router_config_store.py` | Core MongoDB CRUD — all data operations |
| `forgexpackages/src/common_adapters/configurableAI/llm_router.py` | Runtime: `get_configured_llm_manager()` + cache |
| `forgexpackages/src/common_adapters/configurableAI/configurable_ai_manager.py` | Provider abstraction (OpenAI SDK wrapper) |
| `KnowledgeCurator/src/kbcurator/tools/llm_router_tool.py` | MCP tool handlers (admin + user tools) |
| `KnowledgeCurator/src/kbcurator/tools/user_management_system.py` | Workspace creation (seeds default config) |
| `KnowledgeCurator/src/kbcurator/services/workspace_provider_credentials_service.py` | Thin facade over config store (credentials) |
| `KnowledgeCurator/src/kbcurator/services/agent_llm_configuration_service.py` | Thin facade over config store (agent configs) |
| `Forge-X-Web/src/components/UserManagement/LLMConfigManager.jsx` | Frontend table component |
| `Forge-X-Web/src/components/UserManagement/LLMConfigDialog.jsx` | Frontend add/edit modal |
| `Forge-X-Web/src/config/workspace/workspaceListMcp.jsx` | Frontend API layer |

## Architecture

```
Frontend (LLM Config UI)
    │
    ▼
admin_configure_llm_provider / admin_list_llm_providers
    │
    ▼
┌─────────────────────────────────────────────────┐
│  llm_router_config_store (common_adapters)      │
│  - provider_credentials CRUD                     │
│  - agent_configs CRUD                            │
│  - model_assignments management                  │
└─────────────────────────────────────────────────┘
    │
    ▼
MongoDB: llm_configs.workspace_configs
    │
    ▼
┌─────────────────────────────────────────────────┐
│  get_configured_llm_manager() (runtime)          │
│  - Reads current_provider + current_model        │
│  - Builds ConfigurableAIManager                  │
│  - Cached in-memory per workspace/agent          │
└─────────────────────────────────────────────────┘
    │
    ▼
Agent calls manager.generate_text_async(prompt)
```

## Workspace Creation Flow

When a workspace is created:
1. Azure credentials seeded from env vars (`AZURE_OPENAI_LLM_MODEL_*`)
2. Workspace default config created (`current_provider='azure'`, `current_model='gpt-4.1'`)
3. All selected agents get configs via `bulk_create_agent_configurations`
4. `model_assignments['gpt-4.1'] = [all agent_ids]` set on azure credentials

## Important Constraints

1. **Azure is always present** — every agent must have azure configured; users cannot remove agents from it.
2. **One model active at runtime** — `current_provider` + `current_model` determines what the agent uses.
3. **Many models configurable** — `configured_models` allows multiple models per provider per agent.
4. **Model assignments are independent** — assigning agents to model A does NOT affect model B.
5. **Dots in model names** — handled via read-modify-write (not MongoDB dot notation).

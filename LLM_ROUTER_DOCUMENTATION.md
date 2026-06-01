# LLM Router Tool Documentation

## Overview

The LLM Router Tool provides a unified, admin-managed interface for configuring and switching between LLM providers (Azure OpenAI, Quasar) with full persistence in MongoDB config documents. Provider credentials and provider-selection settings are stored per workspace in `llm_configs.workspace_configs` and survive service restarts.

## Key Features

- **MongoDB-backed Config**: API keys, endpoints, model names, and agent provider selection are stored in `llm_configs.workspace_configs`.
- **Role-based Access Control**: Admin-only tools for credential management; authenticated-user tools for switching and querying.
- **Workspace/Agent Isolation**: Each workspace stores one credential set per provider. Each agent independently tracks which providers are enabled and which is currently active.
- **Credential Inheritance**: If an agent has no specific configuration, it falls back to the workspace-level default.
- **In-memory Manager Cache**: `ConfigurableAIManager` instances are cached per workspace/agent and invalidated whenever credentials change.
- **Credential Validation**: LLM credentials are tested before storage to ensure they work correctly.
- **Unified Management**: Single tool (`admin_configure_llm_provider`) handles both creating new and updating existing provider configurations.
- **Smart Operation Detection**: Automatically detects whether to create new or update existing provider based on current state.
- **Flexible Updates**: Can update configuration, manage agents, or do both operations in a single call.
- **Agent List Replacement**: When agent_ids is provided for existing providers, it replaces the entire agent list automatically.

## Supported Providers

`azure`, `quasar`

## Architecture

```
┌─────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────────────────────┐
│   MCP Client        │    │   LLM Router Tool        │    │   MongoDB (llm_configs)               │
│                     │───▶│                          │───▶│                                        │
│ - Tool Calls        │    │ - Role check (admin?)    │    │ workspace_configs                      │
│ - JWT Auth Header   │    │ - Credentials service    │    │  - provider_credentials.{provider}     │
│ - Response Handling │    │ - Agent config service   │    │  - agent_configs.{agent_id/default}    │
└─────────────────────┘    │ - ConfigurableAIMgr      │    │                                        │
                           └──────────────────────────┘    └────────────────────────────────────────┘
```

## Available Tools

All tools require a valid JWT token (`Authorization: Bearer <token>` header). Admin-only tools additionally enforce that the caller holds role **Forge-X Admin** (role_id=0) or **Workspace Admin** (role_id=3).

---

### Admin-Only Tools

These tools store or remove LLM provider credentials. Callers without an admin role receive `{"success": false, "error": "Forbidden: ..."}`.

#### 1. `admin_configure_llm_provider`

**Unified tool** to configure or update LLM provider for a workspace. **Automatically detects** whether to create new or update existing provider configuration. **Validates credentials before storing** by testing a simple generation request.

**Operation Modes:**
- **NEW Provider**: When provider doesn't exist - requires `api_key`, `endpoint`, and `model`
- **EXISTING Provider**: When provider exists - only updates provided fields

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `provider` | string | Yes | `"azure"` or `"quasar"` |
| `workspace_id` | int | Yes | Workspace to configure |
| `agent_ids` | list[int] | No | Agent IDs to enable/add for this provider (optional) |
| `api_key` | string | Conditional | API key (required for new providers, optional for updates) |
| `endpoint` | string | Conditional | API endpoint URL (required for new providers, optional for updates) |
| `model` | string | Conditional | Model / deployment name (required for new providers, optional for updates) |
| `api_version` | string | No | Azure only — e.g. `"2024-12-01-preview"` |
| `deployment_name` | string | No | Azure only — defaults to `model` if omitted |
| `set_as_current` | bool | No | If `true`, set this provider as active for all listed agents (default: `false`) |
| `skip_validation` | bool | No | If `true`, skip credential validation (use with caution, default: `false`) |

**Important**: When `agent_ids` is provided for existing providers, it **replaces** the entire agent list. Agents not in the list will have this provider removed from their configuration.

**Example 1 - Create new provider:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_configure_llm_provider",
    "arguments": {
      "provider": "azure",
      "workspace_id": 782,
      "api_key": "sk-...",
      "endpoint": "https://my-resource.openai.azure.com/",
      "model": "gpt-4",
      "api_version": "2024-12-01-preview",
      "agent_ids": [1, 5, 12],
      "set_as_current": true
    }
  }
}
```

**Example 2 - Update existing provider configuration:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_configure_llm_provider",
    "arguments": {
      "provider": "azure",
      "workspace_id": 782,
      "model": "gpt-4o",
      "api_key": "sk-new-key..."
    }
  }
}
```

**Example 3 - Add agents to existing provider:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_configure_llm_provider",
    "arguments": {
      "provider": "azure",
      "workspace_id": 782,
      "agent_ids": [7, 8, 9],
      "set_as_current": true
    }
  }
}
```

**Example 4 - Update config and add agents in one call:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_configure_llm_provider",
    "arguments": {
      "provider": "azure",
      "workspace_id": 782,
      "model": "gpt-4o",
      "agent_ids": [7, 8, 9],
      "set_as_current": true
    }
  }
}
```

**Example 5 - Replace agent list (keep only specified agents):**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_configure_llm_provider",
    "arguments": {
      "provider": "quasar",
      "workspace_id": 835,
      "agent_ids": [3],
      "set_as_current": true
    }
  }
}
```

**Success Response (New Provider):**
```json
{
  "success": true,
  "message": "Provider 'azure' created for workspace 782. credentials validated successfully. enabled for 3/3 agent(s).",
  "provider": "azure",
  "workspace_id": 782,
  "operation_mode": "create",
  "operations_performed": ["credentials_created", "agents_managed"],
  "config_changes": ["api_key", "endpoint", "model", "api_version"],
  "enabled_agent_ids": [1, 5, 12],
  "skipped_agent_ids": [],
  "set_as_current": true,
  "credentials_validated": true
}
```

**Success Response (Update Provider):**
```json
{
  "success": true,
  "message": "Provider 'azure' updated (fields: model, api_key). credentials validated successfully. processed 2 agent(s), 1 already configured/current.",
  "provider": "azure",
  "workspace_id": 782,
  "operation_mode": "update",
  "operations_performed": ["credentials_updated", "agents_managed"],
  "config_changes": ["model", "api_key"],
  "enabled_agent_ids": [7, 8],
  "skipped_agent_ids": [9],
  "removed_agent_ids": [],
  "set_as_current": true,
  "credentials_validated": true,
  "replace_agents": false
}
```

**Success Response (Replace Agents):**
```json
{
  "success": true,
  "message": "Provider 'quasar' updated (fields: endpoint, model, api_version). credentials validated successfully. processed 1 agent(s), removed 1 agent(s).",
  "provider": "quasar",
  "workspace_id": 835,
  "operation_mode": "update",
  "operations_performed": ["credentials_updated", "agents_managed", "agents_removed"],
  "config_changes": ["endpoint", "model", "api_version"],
  "enabled_agent_ids": [3],
  "skipped_agent_ids": [],
  "removed_agent_ids": [1],
  "set_as_current": true,
  "credentials_validated": true
}
```

**Error Responses:**
```json
{ "success": false, "error": "Forbidden: only Workspace Admins or Platform Admins can configure LLM providers." }
{ "success": false, "error": "Unsupported provider 'openai'. Supported: ['azure', 'quasar']" }
{ "success": false, "error": "For new provider configuration, api_key, endpoint, and model are all required. Provider 'azure' does not exist in workspace 782." }
{ "success": false, "error": "LLM credential validation failed: Invalid API key. Please check your API key, endpoint, model name, and ensure the service is accessible." }
```

---

#### 2. `admin_list_llm_providers`

List all LLM providers configured for a workspace, along with which agents each provider is enabled for.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `workspace_id` | int | Yes | Workspace to inspect |

**Example:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_list_llm_providers",
    "arguments": { "workspace_id": 782 }
  }
}
```

**Success Response:**
```json
{
  "success": true,
  "workspace_id": 782,
  "configured_providers": [
    {
      "provider": "azure",
      "endpoint": "https://my-resource.openai.azure.com/",
      "model": "gpt-4.1",
      "api_version": "2024-12-01-preview",
      "configured_at": "2026-05-27T10:00:00+00:00",
      "configured_by": 42,
      "agents_enabled": [
        { "agent_id": 1, "is_current": true },
        { "agent_id": 5, "is_current": true }
      ]
    },
    {
      "provider": "quasar",
      "endpoint": "https://quasarmarket.coforge.com/qag/llmrouter-api/v2/chat/completions",
      "model": "claude-sonnet-4",
      "api_version": null,
      "configured_at": "2026-05-28T08:30:00+00:00",
      "configured_by": 42,
      "agents_enabled": [
        { "agent_id": 1, "is_current": false }
      ]
    }
  ],
  "supported_providers": ["azure", "quasar"]
}
```

---



#### 2. `admin_remove_llm_provider`

Soft-deactivate an LLM provider from a workspace. The credential record is marked `is_active = false` and the in-memory cache is cleared.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `workspace_id` | int | Yes | Workspace to modify |
| `provider` | string | Yes | Provider name to remove |

**Example:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "admin_remove_llm_provider",
    "arguments": { "workspace_id": 782, "provider": "quasar" }
  }
}
```

**Success Response:**
```json
{
  "success": true,
  "message": "Provider 'quasar' has been deactivated for workspace 782.",
  "provider": "quasar",
  "workspace_id": 782
}
```

**Error Responses:**
```json
{ "success": false, "error": "Forbidden: only Workspace Admins or Platform Admins can remove LLM providers." }
{ "success": false, "error": "Provider 'quasar' was not found or is already inactive." }
```

---

### Authenticated-User Tools

Any user with a valid JWT can call these tools. They read from MongoDB config documents but do not modify credentials.

#### 3. `list_available_llm_providers`

List all LLM providers that an admin has configured for a specific workspace-agent, along with which provider is currently active. Call this first to discover available options before calling `switch_llm_provider`.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `workspace_id` | int | Yes | Workspace ID |
| `agent_id` | int | Yes | Agent ID |

**Example:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "list_available_llm_providers",
    "arguments": { "workspace_id": 782, "agent_id": 1 }
  }
}
```

**Success Response (multiple providers configured):**
```json
{
  "success": true,
  "workspace_id": 782,
  "agent_id": 1,
  "configured_providers": [
    {
      "provider": "azure",
      "model": "gpt-4.1",
      "endpoint_host": "my-resource.openai.azure.com",
      "is_current": true
    },
    {
      "provider": "quasar",
      "model": "claude-sonnet-4",
      "endpoint_host": "quasarmarket.coforge.com",
      "is_current": false
    }
  ],
  "current_provider": "azure",
  "can_switch": true,
  "switch_hint": "Use switch_llm_provider with provider=<name> to change the active LLM."
}
```

**Success Response (no providers configured yet):**
```json
{
  "success": true,
  "workspace_id": 782,
  "agent_id": 1,
  "configured_providers": [],
  "current_provider": null,
  "message": "No LLM providers have been configured for this workspace-agent yet. Contact an admin."
}
```

> **Note:** `endpoint_host` is only the hostname portion of the endpoint URL (no scheme, path, or credentials). API keys are never returned to regular users.

---

#### 4. `switch_llm_provider`

Toggle the active LLM provider for an agent. The provider must have been admin-configured for the workspace **and** enabled for the specific agent. This call only updates `current_provider` — it never stores credentials.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `provider` | string | Yes | Provider to switch to (`"azure"` or `"quasar"`) |
| `workspace_id` | int | Yes | Workspace ID |
| `agent_id` | int | Yes | Agent ID |

**Example:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "switch_llm_provider",
    "arguments": { "provider": "quasar", "workspace_id": 782, "agent_id": 1 }
  }
}
```

**Success Response:**
```json
{
  "success": true,
  "message": "Switched to provider 'quasar'.",
  "provider": "quasar",
  "workspace_id": 782,
  "agent_id": 1,
  "configured_providers": ["azure", "quasar"]
}
```

**Error Responses:**
```json
{
  "success": false,
  "error": "Provider 'quasar' has not been configured for workspace 782. An admin must configure it first via admin_configure_llm_provider."
}
{
  "success": false,
  "error": "Provider 'quasar' is not enabled for agent 1 in workspace 782. An admin must enable it first."
}
```

---

#### 5. `query_llm_router_status`

Return the current LLM router state for a workspace/agent, including which providers have credentials and which is active.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `workspace_id` | int | Yes | Workspace ID |
| `agent_id` | int | No | Agent ID — omit for workspace-level default |

**Example:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "query_llm_router_status",
    "arguments": { "workspace_id": 782, "agent_id": 1 }
  }
}
```

**Success Response:**
```json
{
  "success": true,
  "workspace_id": 782,
  "agent_id": 1,
  "current_provider": "azure",
  "configured_providers": [
    { "provider": "azure", "credentials_present": true },
    { "provider": "quasar", "credentials_present": true }
  ],
  "supported_providers": ["azure", "quasar"]
}
```

> **Note:** `configured_providers` is a list of objects (`{provider, credentials_present}`), not plain strings. `credentials_present` confirms that admin has stored credentials in `llm_configs.workspace_configs.provider_credentials`.

---

#### 6. `test_llm_generation`

Smoke-test the currently active provider for an agent by sending a prompt and returning the response.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | No | Text prompt (default: `"Hello, how are you?"`) |
| `workspace_id` | int | No | Workspace ID (default: `0`) |
| `agent_id` | int | No | Agent ID |

**Example:**
```json
{
  "method": "tools/call",
  "params": {
    "name": "test_llm_generation",
    "arguments": {
      "prompt": "What is artificial intelligence?",
      "workspace_id": 782,
      "agent_id": 1
    }
  }
}
```

**Success Response:**
```json
{
  "success": true,
  "provider_used": "azure",
  "workspace_id": 782,
  "agent_id": 1,
  "prompt": "What is artificial intelligence?",
  "response": "Artificial intelligence (AI) refers to...",
  "response_length": 245
}
```

**Error Response (no provider configured):**
```json
{
  "success": false,
  "error": "No provider is currently configured. Ask an admin to run admin_configure_llm_provider."
}
```

> **Note:** This tool always uses the `current_provider` from MongoDB config. There is no `provider` override parameter — use `switch_llm_provider` first if you want a different provider.

---

## MongoDB Schema

### Collection: `llm_configs.workspace_configs`

One document per workspace (`workspace_id` is unique). Provider credentials and agent-level configuration are stored together.

```json
{
  "workspace_id": 782,
  "provider_credentials": {
    "azure": {
      "api_key": "<secret>",
      "endpoint": "https://my-resource.openai.azure.com/",
      "model": "gpt-4.1",
      "api_version": "2024-12-01-preview",
      "deployment_name": "gpt-4.1",
      "extra_config": {},
      "is_active": true,
      "created_at": "2026-05-29T08:00:00Z",
      "updated_at": "2026-05-29T08:00:00Z",
      "created_by": 247,
      "updated_by": 247
    }
  },
  "agent_configs": {
    "__workspace_default__": {
      "configured_providers": ["azure"],
      "current_provider": "azure",
      "is_active": true,
      "created_at": "2026-05-29T08:00:00Z",
      "updated_at": "2026-05-29T08:00:00Z",
      "created_by": 247,
      "updated_by": 247
    },
    "12": {
      "configured_providers": ["azure", "quasar"],
      "current_provider": "quasar",
      "is_active": true,
      "created_at": "2026-05-29T08:00:00Z",
      "updated_at": "2026-05-29T08:10:00Z",
      "created_by": 247,
      "updated_by": 247
    }
  },
  "created_at": "2026-05-29T08:00:00Z",
  "updated_at": "2026-05-29T08:10:00Z"
}
```

Indexes created by the service:
- Unique index on `workspace_id`
- Non-unique index on `updated_at`

---

## Configuration Hierarchy

- **Agent-specific** (`agent_configs["<agent_id>"]`): Takes precedence. Tracks providers and current selection for that specific agent.
- **Workspace default** (`agent_configs["__workspace_default__"]`): Fallback when an agent has no specific config.

**Resolution logic** (`get_effective_configuration`):
1. Look up agent-specific config key (`"<agent_id>"`) → use if found
2. Look up workspace-default key (`"__workspace_default__"`) → use if found
3. Neither found → no configuration (tools return appropriate errors)

---

## Typical Workflows

### Initial Setup Workflow
```
Admin                                          User
  │                                             │
  ├─ admin_configure_llm_provider               │
  │   provider=azure, api_key=...,              │
  │   agent_ids=[1,5], set_as_current=true      │
  │   → Credentials validated automatically     │
  │                                             │
  ├─ admin_configure_llm_provider               │
  │   provider=quasar, api_key=...,             │
  │   agent_ids=[1], set_as_current=false       │
  │   → Credentials validated automatically     │
  │                                             │
  │                          ┌──────────────────┤
  │                          │ list_available_llm_providers
  │                          │   workspace_id=782, agent_id=1
  │                          │   → [azure (current), quasar]
  │                          │
  │                          │ switch_llm_provider
  │                          │   provider=quasar, workspace_id=782, agent_id=1
  │                          │
  │                          │ test_llm_generation
  │                          │   prompt="Hello!", workspace_id=782, agent_id=1
  │                          └──────────────────┤
```

### Update and Expand Workflow
```
Admin                                          User
  │                                             │
  ├─ admin_configure_llm_provider               │
  │   provider=azure, workspace_id=782,         │
  │   model=gpt-4o, api_key=new_key,            │
  │   agent_ids=[7,8,9], set_as_current=true    │
  │   → Detects existing provider automatically │
  │   → Updates config + adds agents in one call│
  │   → Credentials validated automatically     │
  │                                             │
  │                          ┌──────────────────┤
  │                          │ list_available_llm_providers
  │                          │   workspace_id=782, agent_id=7
  │                          │   → [azure (current)]
  │                          │
  │                          │ test_llm_generation
  │                          │   workspace_id=782, agent_id=7
  │                          └──────────────────┤
```

---

## Role Reference

| Role | role_id | Can call admin tools? | Can call user tools? |
|---|---|---|---|
| Forge-X Admin | 0 | Yes | Yes |
| Workspace Admin | 3 | Yes | Yes |
| All other roles | any | No (Forbidden) | Yes |

---

## Developer Integration

### Using `_build_manager_from_db` pattern (inside KnowledgeCurator)

The internal `_build_manager_from_db` helper constructs a `ConfigurableAIManager` from MongoDB-backed workspace credentials. This is the correct pattern for any code inside KnowledgeCurator that needs to generate text:

```python
from kbcurator.tools.llm_router_tool import _build_manager_from_db

async def my_service_call(workspace_id: int, agent_id: int, prompt: str) -> str:
    manager = _build_manager_from_db(workspace_id, agent_id)
    current = manager.get_current_provider()
    if not current:
        raise RuntimeError("No LLM provider configured for this agent.")
    return await manager.generate_text_async(prompt)
```

### Using `ConfigurableAIManager` directly (other agents / scripts)

For agents outside KnowledgeCurator (e.g. DevAgent, ProductOwner) that manage their own credentials via env vars, use `ConfigurableAIManager` directly:

```python
from common_adapters.configurableAI import ConfigurableAIManager

manager = ConfigurableAIManager()

# Configure from explicit dict
manager.configure_provider("azure", {
    "provider_name": "azure",
    "api_key": "sk-...",
    "endpoint": "https://my-resource.openai.azure.com/",
    "model": "gpt-4.1",
    "deployment_name": "gpt-4.1",
    "api_version": "2024-12-01-preview",
})
manager.set_current_provider("azure")

response = await manager.generate_text_async("Hello!")
```

Or use the `get_ai_manager` convenience function for quick env-var-based setup (does **not** hit MongoDB config documents):

```python
from common_adapters.configurableAI import get_ai_manager

# Reads AZURE_OPENAI_LLM_MODEL_* env vars
manager = get_ai_manager(provider_name="azure", auto_configure=True)
response = await manager.generate_text_async("Hello!")
```

> **`get_ai_manager` signature:** `get_ai_manager(provider_name="azure", auto_configure=True)` — it does **not** accept `workspace_id` or `agent_id`. Use `_build_manager_from_db` for MongoDB-backed, workspace-scoped generation.

### Clearing the cache after admin changes

```python
from common_adapters.configurableAI import clear_ai_manager_cache

# Clear cache for a specific workspace/agent
clear_ai_manager_cache(workspace_id=782, agent_id=1)

# Clear all cached managers
clear_ai_manager_cache()
```

The admin tools (`admin_configure_llm_provider`, `admin_remove_llm_provider`, `switch_llm_provider`) call `clear_ai_manager_cache` automatically.

---

## Service Layer Reference

### `WorkspaceProviderCredentialsService`

| Method | Description |
|---|---|
| `get_provider_credentials(workspace_id, provider_name)` | Returns credential dict or `None` |
| `list_workspace_providers(workspace_id)` | Returns all active credential records |
| `build_config_dict(workspace_id, provider_name)` | Returns dict ready for `configure_provider()`, or `None` |
| `upsert_provider_credentials(...)` | Create or update credentials |
| `deactivate_provider_credentials(workspace_id, provider_name, user_id)` | Soft-delete (set `is_active=false`) |

### `AgentLLMConfigurationService`

| Method | Description |
|---|---|
| `get_configuration(workspace_id, agent_id)` | Get agent-specific or workspace-default config |
| `get_effective_configuration(workspace_id, agent_id)` | Agent config with workspace fallback |
| `add_provider(workspace_id, provider, agent_id, set_as_current, user_id)` | Append provider to agent's list |
| `switch_provider(workspace_id, provider, agent_id, user_id)` | Set `current_provider` |
| `create_or_update_configuration(...)` | Upsert full configuration row |
| `delete_configuration(workspace_id, agent_id, user_id)` | Soft-delete agent config |
| `get_workspace_configurations(workspace_id)` | All configs for a workspace |
| `bulk_create_agent_configurations(workspace_id, agent_ids, ...)` | Bulk-create on workspace setup |

---

## Error Handling Reference

| Scenario | Tool | Response |
|---|---|---|
| Caller not admin | Any admin tool | `{"success": false, "error": "Forbidden: ..."}` |
| Unsupported provider | `admin_configure_llm_provider` | `{"success": false, "error": "Unsupported provider '...'"}` |
| Missing credentials | `admin_configure_llm_provider` | `{"success": false, "error": "api_key, endpoint and model are all required."}` |
| Credential validation failed | `admin_configure_llm_provider` | `{"success": false, "error": "LLM credential validation failed. Configuration aborted.", "validation_error": "...", "validation_details": "..."}` |
| Missing required fields for new provider | `admin_configure_llm_provider` | `{"success": false, "error": "For new provider configuration, api_key, endpoint, and model are all required. Provider '...' does not exist in workspace ...."}` |
| Validation failed | `admin_configure_llm_provider` | `{"success": false, "error": "LLM credential validation failed: Invalid API key. Please check your API key, endpoint, model name, and ensure the service is accessible."}` |
| No config for workspace-agent | `list_available_llm_providers` | `{"success": true, "configured_providers": [], "message": "No LLM providers have been configured..."}` |
| Provider not in workspace | `switch_llm_provider` | `{"success": false, "error": "Provider '...' has not been configured..."}` |
| Provider not enabled for agent | `switch_llm_provider` | `{"success": false, "error": "Provider '...' is not enabled for agent..."}` |
| No active provider | `test_llm_generation` | `{"success": false, "error": "No provider is currently configured..."}` |
| Provider not found / already inactive | `admin_remove_llm_provider` | `{"success": false, "error": "Provider '...' was not found or is already inactive."}` |
| MongoDB / network error | Any tool | `{"success": false, "error": "<exception message>"}` |

---

## Troubleshooting

**"No provider is currently configured"**
- An admin must call `admin_configure_llm_provider` with `set_as_current: true`, or the user must call `switch_llm_provider` after admin setup.

**`configured_providers` is empty in `query_llm_router_status`**
- No agent config row exists yet. Admin must run `admin_configure_llm_provider` first.

**`credentials_present: false` for a provider in `query_llm_router_status`**
- The agent config row lists the provider but admin credentials were removed (`admin_remove_llm_provider`). Re-configure via `admin_configure_llm_provider`.

**Provider switch succeeds but generation still uses old provider**
- The in-memory cache may not have been cleared. Call `clear_ai_manager_cache(workspace_id, agent_id)` or restart the service.

**"LLM credential validation failed"**
- The provided API key, endpoint, or model name is incorrect or the service is unreachable. Verify credentials and network connectivity.
- Use `skip_validation: true` to bypass validation if you're certain the credentials are correct but validation is failing due to network issues.

**"list index out of range" error in `list_available_llm_providers`**
- This can occur if endpoint URL parsing fails. The system now handles this gracefully and will use the full endpoint if parsing fails.

---

## Workspace Setup Integration

Workspace creation now always seeds a workspace-level default provider selection (`azure`) and then bulk-creates per-agent configs for selected agents:

```python
from kbcurator.services.agent_llm_configuration_service import agent_llm_config_service

agent_llm_config_service.bulk_create_agent_configurations(
    workspace_id=782,
    agent_ids=[1, 5, 12],
    configured_providers=["azure"],
    current_provider="azure",
    user_id=admin_user_id,
)
```

Note: provider selection defaults are created automatically, but credentials must still be stored separately via `admin_configure_llm_provider` (or directly via the service) before generation can succeed.

---

**Version**: 4.0.0
**Last Updated**: 2026-06-01
**Credential Source**: `llm_configs.workspace_configs` collection (MongoDB config documents)
**Security Model**: JWT required for all tools; admin role_id (0 or 3) required for credential management
**Major Changes**: Unified `admin_configure_llm_provider` tool with smart operation detection for both create and update operations

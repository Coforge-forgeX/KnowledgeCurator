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

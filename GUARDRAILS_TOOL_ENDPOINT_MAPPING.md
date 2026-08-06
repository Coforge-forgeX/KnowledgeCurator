# Guardrails Tool-to-Endpoint Mapping

Last updated: 2026-08-04

This document captures the backend tool-to-endpoint mapping, input contracts, and real request/response samples.

## 1) Get guardrails configuration
- Tool: `get_guardrail_config`
- Method: `GET`
- Endpoint: `/guardrails/configuration`
- Inputs:
  - `workspace_id` (required)
- Body: none
- Output (structuredContent):
```json
{
  "workspaceId": "1006",
  "sections": [
    {
      "id": "security",
      "title": "SECURITY",
      "description": "Detects attempts to override system instructions",
      "items": [
        {
          "id": "<uuid>",
          "guardrail_name": "PROMPT_INJECTION",
          "score": 0.8,
          "is_active": true,
          "actions": "B"
        }
      ]
    }
  ]
}
```

## 2) Update guardrails configuration (batch)
- Tool: `batch_update_guardrail_config`
- Method: `POST`
- Endpoint: `/guardrails/configuration/batch`
- Inputs:
  - `workspace_id` (required)
  - `updates` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Body:
```json
{
  "updates": [
    {
      "id": "<uuid>",
      "field": "score | is_active | actions | ...",
      "value": "<any>"
    }
  ]
}
```
- Output (structuredContent): passthrough from backend (not remapped by tool)
```json
{
  "status": "success",
  "message": "Guardrail configuration updated successfully",
  "updated_count": 1
}
```

## 3) Get guardrail logs
- Tool: `get_guardrail_logs`
- Method: `GET`
- Endpoint: `/dashboard/guardrail-logs`
- Inputs:
  - `workspace_id` (required)
  - `start_date` (required, `YYYY-MM-DD`)
  - `end_date` (required, `YYYY-MM-DD`)
  - `user_email` (required)
  - `limit` (optional, default `20`)
  - `offset` (optional, default `0`)
  - `user_id` (optional)
- Query params: `start_date`, `end_date`, `limit`, `offset`
- Output (structuredContent):
```json
{
  "workspaceId": "1006",
  "metadata": {
    "start_date": "2026-08-01",
    "end_date": "2026-08-04",
    "limit": 20,
    "offset": 0,
    "total": 120
  },
  "data": [
    {
      "id": "<log-id>",
      "timestamp": "2026-08-04T07:30:00Z",
      "guardrail_name": "PII",
      "action": "B",
      "result": "blocked"
    }
  ]
}
```

## 4) Get PII entities
- Tool: `get_pii_entities`
- Method: `GET`
- Endpoint: `/guardrails/pii/entities`
- Inputs:
  - `workspace_id` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Output (structuredContent):
```json
{
  "workspaceId": "1006",
  "records": [
    {
      "id": "<entity-id>",
      "entity_name": "EMAIL_ADDRESS",
      "is_active": true,
      "linked": false
    }
  ]
}
```

## 5) Update PII entities (batch)
- Tool: `batch_update_pii_entities`
- Method: `POST`
- Endpoint: `/guardrails/pii/entities/batch`
- Inputs:
  - `workspace_id` (required)
  - `updates` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Body:
```json
{
  "updates": [
    {
      "entity_id": "<uuid>",
      "is_active": true,
      "linked": false
    }
  ]
}
```
- Output (structuredContent): passthrough from backend (not remapped by tool)
```json
{
  "status": "success",
  "message": "PII entities updated successfully",
  "updated_count": 1
}
```

## 6) Create competitor
- Tool: `create_competitor`
- Method: `POST`
- Endpoint: `/guardrails/competitors`
- Inputs:
  - `workspace_id` (required)
  - `competitor_name` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Body:
```json
{
  "competitor_name": "TCS"
}
```
- Output (structuredContent): passthrough from backend (not remapped by tool)
```json
{
  "status": "success",
  "message": "Competitor created successfully",
  "id": 21
}
```

## 7) Get competitors
- Tool: `get_competitors`
- Method: `GET`
- Endpoint: `/guardrails/competitors`
- Inputs:
  - `workspace_id` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Output (structuredContent):
```json
{
  "workspaceId": "1006",
  "terms": [
    {
      "id": 21,
      "competitor_name": "TCS",
      "is_active": true
    }
  ]
}
```

## 8) Delete competitor
- Tool: `delete_competitor`
- Method: `DELETE`
- Endpoint: `/guardrails/competitors/{competitor_id}`
- Inputs:
  - `workspace_id` (required)
  - `competitor_id` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Output (structuredContent): passthrough from backend (not remapped by tool)
```json
{
  "status": "success",
  "message": "Competitor deleted successfully",
  "updated_count": 1
}
```

## 9) Get regex patterns
- Tool: `get_regex_patterns`
- Method: `GET`
- Endpoint: `/guardrails/regex-patterns`
- Real sample request:
```json
{
  "method": "tools/call",
  "params": {
    "name": "get_regex_patterns",
    "arguments": {
      "workspace_id": "1006",
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"workspaceId\":\"1006\",\"patterns\":[{\"id\":6,\"name\":\"Valid Name1\",\"pattern\":\"^[A-Z]\",\"action\":\"B\",\"is_active\":true,\"created_at\":\"2026-08-04T07:21:27.050211\"}]}"
    }
  ],
  "structuredContent": {
    "workspaceId": "1006",
    "patterns": [
      {
        "id": 6,
        "name": "Valid Name1",
        "pattern": "^[A-Z]",
        "action": "B",
        "is_active": true,
        "created_at": "2026-08-04T07:21:27.050211"
      }
    ]
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

## 10) Update regex pattern status
- Tool: `update_regex_pattern_status`
- Method: `PUT`
- Endpoint: `/guardrails/regex-patterns/{pattern_id}/status`
- Real sample request:
```json
{
  "method": "tools/call",
  "params": {
    "name": "update_regex_pattern_status",
    "arguments": {
      "workspace_id": "1006",
      "pattern_id": 6,
      "is_active": false,
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\":\"success\",\"message\":\"Regex pattern 6 status updated successfully\",\"updated_count\":1}"
    }
  ],
  "structuredContent": {
    "status": "success",
    "message": "Regex pattern 6 status updated successfully",
    "updated_count": 1
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

## 11) Update regex pattern
- Tool: `update_regex_pattern`
- Method: `PUT`
- Endpoint: `/guardrails/regex-patterns/{pattern_id}`
- Canonical UI request (recommended):
```json
{
  "method": "tools/call",
  "params": {
    "name": "update_regex_pattern",
    "arguments": {
      "workspace_id": "1006",
      "updates": [
        {
          "pattern_id": 6,
          "config_dict": {
            "id": 6,
            "name": "Valid Name3",
            "pattern": "^[A-Z]",
            "action": "B",
            "is_active": true
          }
        }
      ],
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Single-pattern convenience (same contract):
  - Send one item in `updates`, or pass `updates` as a single object.
```json
{
  "method": "tools/call",
  "params": {
    "name": "update_regex_pattern",
    "arguments": {
      "workspaceId": "1006",
      "updates": {
        "pattern_id": 6,
        "config_dict": {
          "id": 6,
          "name": "Valid Name3",
          "pattern": "^[A-Z]",
          "action": "B",
          "is_active": true
        }
      },
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\":\"success\",\"message\":\"Regex pattern 6 updated successfully\",\"updated_count\":1}"
    }
  ],
  "structuredContent": {
    "status": "success",
    "message": "Regex pattern 6 updated successfully",
    "updated_count": 1
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

## 12) Delete regex pattern
- Tool: `delete_regex_pattern`
- Method: `DELETE`
- Endpoint: `/guardrails/regex-patterns/{pattern_id}`
- Real sample request:
```json
{
  "method": "tools/call",
  "params": {
    "name": "delete_regex_pattern",
    "arguments": {
      "workspace_id": "1006",
      "pattern_id": 6,
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\":\"success\",\"message\":\"Regex pattern 6 deleted successfully\",\"updated_count\":1}"
    }
  ],
  "structuredContent": {
    "status": "success",
    "message": "Regex pattern 6 deleted successfully",
    "updated_count": 1
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

## 13) Guardrails overview
- Tool: `get_dashboard_overview`
- Method: `GET`
- Endpoint: `/dashboard/overview`
- Inputs:
  - `workspace_id` (required)
  - `start_date` (required, `YYYY-MM-DD`)
  - `end_date` (required, `YYYY-MM-DD`)
  - `user_email` (required)
  - `user_id` (optional)
- Query params: `start_date`, `end_date`
- Output (structuredContent): passthrough from backend (not remapped by tool)
```json
{
  "overview": {
    "total_requests": 1200,
    "blocked_requests": 33,
    "warned_requests": 47
  },
  "trend": [
    {
      "date": "2026-08-04",
      "total": 120,
      "blocked": 3,
      "warned": 4
    }
  ]
}
```

## 14) List API keys for app/workspace
- Tool: `list_api_keys`
- Method: `GET`
- Endpoint: `/api/v1/api-keys/`
- Inputs:
  - `workspace_id` (required)
  - `user_email` (required)
  - `user_id` (optional)
- Query params sent by backend:
  - `user_id=<user_email>`
- Output (structuredContent): passthrough from backend (not remapped by tool)
```json
{
  "response": [
    {
      "id": "<key-id>",
      "name": "default-key",
      "prefix": "sk-xxxx",
      "is_active": true,
      "created_at": "2026-08-04T07:00:00Z"
    }
  ]
}
```

## Additional Tool Samples (Actual Data)

### Create regex pattern
- Tool: `create_regex_pattern`
- Method: `POST`
- Endpoint: `/guardrails/regex-patterns`
- Real sample request:
```json
{
  "method": "tools/call",
  "params": {
    "name": "create_regex_pattern",
    "arguments": {
      "workspace_id": "1006",
      "config_dict": {
        "name": "Valid Name3",
        "pattern": "^[A-Z]",
        "action": "B",
        "is_active": true
      },
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"status\":\"success\",\"message\":\"Regex pattern added successfully with ID 7\",\"updated_count\":1}"
    }
  ],
  "structuredContent": {
    "status": "success",
    "message": "Regex pattern added successfully with ID 7",
    "updated_count": 1
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

### Get AI gateway system config
- Tool: `get_ai_gateway_system_config`
- Method: `GET`
- Endpoint: `/ai-gateway/system-config`
- Real sample request:
```json
{
  "method": "tools/call",
  "params": {
    "name": "get_ai_gateway_system_config",
    "arguments": {
      "workspace_id": "1006",
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"id\":\"7f755de3-2075-423c-b3e4-002a3fe3099d\",\"guardrail_model\":\"gpt-4-1\",\"is_guardrails_enabled\":true,\"admin_emails\":\"test@test.com\",\"on_prem_llm_model\":\"lamma-4-scout\",\"is_guardrail_notification_enabled\":true,\"input_guardrail_execution_mode\":\"async\",\"output_guardrail_execution_mode\":\"sync\",\"warning_message\":\"warning message 2\",\"block_message\":\"block message\",\"app_id\":\"8861e655-23f5-4382-b782-db94845514b3\"}"
    }
  ],
  "structuredContent": {
    "id": "7f755de3-2075-423c-b3e4-002a3fe3099d",
    "guardrail_model": "gpt-4-1",
    "is_guardrails_enabled": true,
    "admin_emails": "test@test.com",
    "on_prem_llm_model": "lamma-4-scout",
    "is_guardrail_notification_enabled": true,
    "input_guardrail_execution_mode": "async",
    "output_guardrail_execution_mode": "sync",
    "warning_message": "warning message 2",
    "block_message": "block message",
    "app_id": "8861e655-23f5-4382-b782-db94845514b3"
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

### Update AI gateway system config
- Tool: `update_ai_gateway_system_config`
- Method: `PUT`
- Endpoint: `/ai-gateway/system-config`
- Real sample request:
```json
{
  "method": "tools/call",
  "params": {
    "name": "update_ai_gateway_system_config",
    "arguments": {
      "workspace_id": "1006",
      "config_dict": {
        "id": "7f755de3-2075-423c-b3e4-002a3fe3099d",
        "guardrail_model": "gpt-4-1",
        "is_guardrails_enabled": true,
        "admin_emails": "test@test.com",
        "on_prem_llm_model": "lamma-4-scout",
        "is_guardrail_notification_enabled": true,
        "input_guardrail_execution_mode": "async",
        "output_guardrail_execution_mode": "sync",
        "warning_message": "warning message 2",
        "block_message": "block message",
        "app_id": "8861e655-23f5-4382-b782-db94845514b3"
      },
      "user_email": null,
      "user_id": null
    }
  }
}
```
- Real sample response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"id\":\"7f755de3-2075-423c-b3e4-002a3fe3099d\",\"guardrail_model\":\"gpt-4-1\",\"is_guardrails_enabled\":true,\"admin_emails\":\"test@test.com\",\"on_prem_llm_model\":\"lamma-4-scout\",\"is_guardrail_notification_enabled\":true,\"input_guardrail_execution_mode\":\"async\",\"output_guardrail_execution_mode\":\"sync\",\"warning_message\":\"warning message 2\",\"block_message\":\"block message\",\"app_id\":\"8861e655-23f5-4382-b782-db94845514b3\"}"
    }
  ],
  "structuredContent": {
    "id": "7f755de3-2075-423c-b3e4-002a3fe3099d",
    "guardrail_model": "gpt-4-1",
    "is_guardrails_enabled": true,
    "admin_emails": "test@test.com",
    "on_prem_llm_model": "lamma-4-scout",
    "is_guardrail_notification_enabled": true,
    "input_guardrail_execution_mode": "async",
    "output_guardrail_execution_mode": "sync",
    "warning_message": "warning message 2",
    "block_message": "block message",
    "app_id": "8861e655-23f5-4382-b782-db94845514b3"
  },
  "isError": false,
  "_meta": {
    "fastmcp": {
      "tags": []
    }
  }
}
```

## Duplicate endpoint samples shared
These sample files point to the same endpoint behavior:
- `GuardrailsConfiguration.txt` and `GetGuardrailsConfigurationForApp.txt`
- `UpdateGuardrailsConfiguration.txt` and `UpdateGuardrailsForApp.txt`

## UI payload contract (recommended)
Use this common shape for calling backend tools from UI:

```json
{
  "workspace_id": "<workspace-id>",
  "user_email": "user@company.com",
  "...tool_specific_fields": "..."
}
```

## Compatibility note
If UI still sends legacy `user_id`, backend can accept it, but endpoint identity is currently derived from `user_email`.

## Common error output
All methods above return this error shape when workspace configuration is missing:

```json
{
  "error": "Workspace configuration not found"
}
```

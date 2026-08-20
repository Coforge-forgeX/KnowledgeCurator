"""
Extract Keywords API Endpoint

Extracts relevant node labels from a user query given candidate node labels as context.
Optimized, production-ready implementation following standard coding practices.
"""
import json
import re
import time
from typing import Any, Dict, List, Optional

from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.exceptions import ValidationException
from src.core.logging import get_logger
from shared.lightrag import build_azure_openai_chat_completion_func

from .payloads import ExtractKeywordsRequest, ExtractKeywordsResponse

try:
    from common_adapters.configurableAI.llm_router_config_store import (
        llm_router_config_store,
    )
except Exception:  # pragma: no cover - optional dependency fallback
    llm_router_config_store = None

logger = get_logger(__name__)


def _build_llm_func(workspace_id: int, agent_id: Optional[int]):
    """
    Build LLM completion callable using workspace/agent routing or default Azure OpenAI settings.
    """
    eff_workspace_id = workspace_id
    eff_agent_id = agent_id or 1


    if llm_router_config_store is not None:
        try:
            effective = llm_router_config_store.get_effective_configuration(eff_workspace_id, eff_agent_id)
            current_provider = (effective or {}).get("current_provider", "").strip().lower()
            current_model = (effective or {}).get("current_model")
            if current_provider:
                provider_config = llm_router_config_store.build_config_dict(
                    eff_workspace_id,
                    current_provider,
                    model_override=current_model,
                )
                if isinstance(provider_config, dict) and (provider_config.get("provider_name") or "").strip().lower() == "azure":
                    api_key = provider_config.get("api_key")
                    api_base = provider_config.get("endpoint")
                    api_version = provider_config.get("api_version") or settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION
                    deployment = provider_config.get("deployment_name") or provider_config.get("model")
                    if all([api_key, api_base, deployment]):
                        logger.info(
                            "Using common_adapters LLM config for keyword extraction",
                            workspace_id=eff_workspace_id,
                            agent_id=eff_agent_id,
                            provider=current_provider,
                        )
                        return build_azure_openai_chat_completion_func(
                            api_key=str(api_key or ""),
                            api_base=str(api_base or ""),
                            api_version=str(api_version or ""),
                            deployment=str(deployment or ""),
                        )
        except Exception as route_err:
            logger.warning(
                "Failed to resolve common_adapters LLM config for keyword extraction",
                workspace_id=eff_workspace_id,
                error=route_err,
            )

    # Fallback to default Azure OpenAI LLM settings
    api_key = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_KEY
    api_base = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_BASE or getattr(settings.lightrag, "AZURE_OPENAI_LLM_MODEL_ENDPOINT", None)
    api_version = settings.lightrag.AZURE_OPENAI_LLM_MODEL_API_VERSION
    deployment = settings.lightrag.AZURE_OPENAI_LLM_MODEL_LLM_MODEL or getattr(settings.lightrag, "AZURE_OPENAI_LLM_MODEL_NAME", None)

    if all([api_key, api_base, deployment]):
        return build_azure_openai_chat_completion_func(
            api_key=str(api_key or ""),
            api_base=str(api_base or ""),
            api_version=str(api_version or ""),
            deployment=str(deployment or ""),
        )

    raise ValidationException(message="No LLM configuration available for keyword extraction")


def _parse_llm_response(response_text: str) -> List[str]:
    """
    Parse keywords from LLM JSON or text response.
    """
    text = (response_text or "").strip()

    # Clean markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try parsing as JSON object or list
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, dict):
            for key in ("keywords", "relevant_keywords", "node_labels", "matched_labels"):
                val = data.get(key)
                if isinstance(val, list):
                    return [str(item).strip() for item in val if str(item).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON array pattern using regex
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return [str(item).strip() for item in arr if str(item).strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: line by line or comma-separated list
    lines = [line.strip().lstrip("-*•0123456789. ") for line in text.splitlines() if line.strip()]
    return [line for line in lines if line]


def _filter_and_dedupe_keywords(
    raw_keywords: List[str],
    candidate_labels: List[str],
) -> List[str]:
    """
    Match and deduplicate extracted keywords against candidate node labels.
    """
    if not candidate_labels:
        # If no candidates provided, return unique extracted keywords as-is
        seen: set[str] = set()
        result: List[str] = []
        for kw in raw_keywords:
            norm = kw.lower()
            if norm not in seen:
                seen.add(norm)
                result.append(kw)
        return result

    # Build canonical lookup map for candidate node labels
    label_map: Dict[str, str] = {lbl.strip().lower(): lbl.strip() for lbl in candidate_labels if lbl.strip()}
    matched_results: List[str] = []
    matched_seen: set[str] = set()

    for kw in raw_keywords:
        kw_norm = kw.strip().lower()
        # Direct exact match
        if kw_norm in label_map and kw_norm not in matched_seen:
            matched_seen.add(kw_norm)
            matched_results.append(label_map[kw_norm])
            continue

        # Substring / partial match against candidates
        for cand_norm, cand_orig in label_map.items():
            if cand_norm not in matched_seen:
                if kw_norm == cand_norm or kw_norm in cand_norm or cand_norm in kw_norm:
                    matched_seen.add(cand_norm)
                    matched_results.append(cand_orig)
                    break

    return matched_results


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Extract relevant node labels from a user query given candidate node labels as context.

    POST /api/v2/kb/extract-keywords
    """
    correlation_id = context.correlation_id
    start_time = time.time()

    try:
        raw_payload, error_response = parse_request(req, ExtractKeywordsRequest)
        if error_response or not isinstance(raw_payload, ExtractKeywordsRequest):
            return error_response or create_error_response("Invalid request payload", status_code=400)

        payload: ExtractKeywordsRequest = raw_payload
        user_query = payload.user_query
        node_labels = [str(lbl).strip() for lbl in (payload.node_labels or []) if str(lbl).strip()]
        workspace_id = payload.workspace_id
        agent_id = payload.agent_id

        logger.info(
            "Extract keywords request received",
            correlation_id=correlation_id,
            query_length=len(user_query),
            candidate_count=len(node_labels),
            workspace_id=workspace_id,
        )

        # Build system prompt context
        node_labels_str = ", ".join(f'"{lbl}"' for lbl in node_labels) if node_labels else "None"
        system_prompt = (
            f"You are a Knowledge Graph entity and keyword extraction specialist.\n"
            f"Given the following node labels available in the knowledge graph: [{node_labels_str}].\n"
            f"Extract the node labels from the user query that match or are closely related to these candidate labels.\n"
            f"Return ONLY a JSON array of strings containing the relevant matched node labels.\n"
            f"Response Format Example: [\"Label 1\", \"Label 2\"]"
        )

        llm_func = _build_llm_func(workspace_id=workspace_id, agent_id=agent_id)

        response_text = await llm_func(
            prompt=user_query,
            system_prompt=system_prompt,
            temperature=0.1,
            top_p=1,
            n=1,
        )

        raw_extracted = _parse_llm_response(response_text)
        final_keywords = _filter_and_dedupe_keywords(raw_extracted, node_labels)

        # Direct exact fallback if LLM produced empty list but user query explicitly mentions a candidate label
        if not final_keywords and node_labels:
            query_lower = user_query.lower()
            for candidate in node_labels:
                if candidate.lower() in query_lower and candidate not in final_keywords:
                    final_keywords.append(candidate)

        response_data = ExtractKeywordsResponse(
            keywords=final_keywords,
            user_query=user_query,
            node_labels=node_labels,
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            "Extract keywords completed successfully",
            correlation_id=correlation_id,
            keywords_found=len(final_keywords),
            elapsed_ms=elapsed_ms,
        )

        return create_success_response(
            message="Keywords extracted successfully",
            data=response_data.dict(),
            status_code=200,
            correlation_id=correlation_id,
        )

    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(
            "Extract keywords failed",
            error=e,
            correlation_id=correlation_id,
            exc_info=True,
        )
        return create_internal_error_response(
            message="Failed to extract keywords from query",
            error=e,
            error_code="EXTRACT_KEYWORDS_ERROR",
            correlation_id=correlation_id,
        )
 
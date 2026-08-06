"""Shared builders for LightRAG LLM and embedding callables."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import aiohttp
import numpy as np
from lightrag.llm.ollama import ollama_embed


class RateLimitError(Exception):
    """Raised when Azure OpenAI rate limit is exceeded."""
    pass


async def _post_json(
    endpoint: str,
    headers: dict,
    payload: dict,
    timeout_seconds: Optional[float] = None,
) -> dict:
    """Post JSON payload and return JSON response.

    Args:
        endpoint: API endpoint URL
        headers: HTTP headers
        payload: JSON payload
        timeout_seconds: Request timeout

    Returns:
        JSON response

    Raises:
        RateLimitError: When rate limit is exceeded
        ValueError: On other request failures
    """
    timeout = aiohttp.ClientTimeout(total=timeout_seconds) if timeout_seconds else None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                return await response.json()

            response_text = await response.text()

            # Check for rate limit errors
            is_rate_limit = (
                response.status == 429 or
                "rate_limit_exceeded" in response_text.lower() or
                "too_many_requests" in response_text.lower()
            )

            if is_rate_limit:
                raise RateLimitError(f"Rate limit exceeded: {response_text}")

            raise ValueError(f"Request failed: {response_text}")


def build_azure_openai_chat_completion_func(
    *,
    api_key: str,
    api_base: str,
    api_version: str,
    deployment: str,
) -> Callable[..., Any]:
    """Build Azure OpenAI chat completion function compatible with LightRAG.

    Args:
        api_key: Azure OpenAI API key
        api_base: Azure OpenAI API base URL
        api_version: Azure OpenAI API version
        deployment: Azure OpenAI deployment name

    Returns:
        LLM function compatible with LightRAG
    """

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List] = None,
        **kwargs,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        endpoint = (
            f"{api_base}openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
            "top_p": kwargs.get("top_p", 1),
            "n": kwargs.get("n", 1),
        }

        result = await _post_json(endpoint=endpoint, headers=headers, payload=payload)
        return result["choices"][0]["message"]["content"]

    return llm_model_func


def build_azure_openai_embedding_func(
    *,
    api_key: str,
    api_base: str,
    api_version: str,
    deployment: str,
    dimensions: Optional[int] = None,
) -> Callable[[list[str]], Any]:
    """Build Azure OpenAI embedding function compatible with LightRAG.

    Args:
        api_key: Azure OpenAI API key
        api_base: Azure OpenAI API base URL
        api_version: Azure OpenAI API version
        deployment: Azure OpenAI deployment name
        dimensions: Embedding dimensions (optional)

    Returns:
        Embedding function compatible with LightRAG
    """

    async def embedding_func(texts: list[str]):
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        endpoint = (
            f"{api_base}openai/deployments/{deployment}/embeddings"
            f"?api-version={api_version}"
        )

        payload = {"input": texts}
        if dimensions is not None:
            payload["dimensions"] = dimensions

        result = await _post_json(endpoint=endpoint, headers=headers, payload=payload)
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)

    return embedding_func


def build_ollama_embedding_func(*, host: str, embed_model: str) -> Callable[[list[str]], Any]:
    """Build Ollama embedding function compatible with LightRAG."""

    async def embedding_func(texts: list[str]):
        return await ollama_embed(texts, embed_model=embed_model, host=host)

    return embedding_func

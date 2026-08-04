"""AWS Lambda HTTP entrypoint using provider-agnostic handlers."""

import asyncio
from typing import Any, Dict

from src.adapters.cloud_function_adapter import (
    abstract_response_to_http_tuple,
    dispatch_request,
    from_lambda_event,
)


async def _handle(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    abstract_req, abstract_ctx = from_lambda_event(event, context)
    abstract_resp = await dispatch_request(abstract_req, abstract_ctx)
    body, status_code, headers, _ = abstract_response_to_http_tuple(abstract_resp)

    return {
        "statusCode": status_code,
        "headers": headers,
        "body": body,
        "isBase64Encoded": False,
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    return asyncio.run(_handle(event, context))

"""GCP Cloud Functions HTTP entrypoint using provider-agnostic handlers."""

import asyncio
from typing import Any

from src.adapters.cloud_function_adapter import (
    abstract_response_to_http_tuple,
    dispatch_request,
    from_gcp_request,
)


async def _handle(request: Any):
    abstract_req, abstract_ctx = from_gcp_request(request)
    abstract_resp = await dispatch_request(abstract_req, abstract_ctx)
    body, status_code, headers, mimetype = abstract_response_to_http_tuple(abstract_resp)
    return body, status_code, headers


def entrypoint(request: Any):
    # GCP Functions accepts Flask-style return tuples: (body, status, headers)
    return asyncio.run(_handle(request))

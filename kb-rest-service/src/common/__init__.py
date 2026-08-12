"""Shared utilities for kb-rest-service"""
from .messages import ErrorMessages, InfoMessages, SuccessMessages
from .payloads import (
    BasePayload,
    BulkDocumentIndexPayload,
    DocumentDeletePayload,
    DocumentIndexPayload,
    NonEmptyStr,
    OptionalNonEmptyStr,
    PaginationPayload,
    QueryRequestPayload,
    VALID_QUERY_MODES,
    parse_request,
)
from .response_utils import (
    ITEMS_KEY,
    build_error_body,
    build_success_body,
    create_batch_response,
    create_error_response,
    create_exception_response,
    create_internal_error_response,
    create_paginated_response,
    create_query_response,
    create_success_response,
)

__all__ = [
    # Messages
    "ErrorMessages",
    "SuccessMessages",
    "InfoMessages",
    # Payloads
    "BasePayload",
    "NonEmptyStr",
    "OptionalNonEmptyStr",
    "PaginationPayload",
    "QueryRequestPayload",
    "DocumentIndexPayload",
    "BulkDocumentIndexPayload",
    "DocumentDeletePayload",
    "VALID_QUERY_MODES",
    "parse_request",
    # Response Utils
    "ITEMS_KEY",
    "build_success_body",
    "build_error_body",
    "create_success_response",
    "create_error_response",
    "create_exception_response",
    "create_internal_error_response",
    "create_paginated_response",
    "create_query_response",
    "create_batch_response",
]

"""Utility functions for creating standardized API responses"""
import json
from datetime import datetime
from typing import Any, Dict, Optional

import azure.functions as func


def create_success_response(
    message: str,
    data: Optional[Any] = None,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
) -> func.HttpResponse:
    """
    Create a standardized success response.

    Args:
        message: Success message
        data: Optional response data
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking

    Returns:
        func.HttpResponse: Formatted success response
    """
    response_body = {
        "success": True,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if data is not None:
        response_body["data"] = data

    if correlation_id:
        response_body["correlation_id"] = correlation_id

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return func.HttpResponse(
        body=json.dumps(response_body),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_error_response(
    message: str,
    error_code: str = "ERROR",
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400,
    correlation_id: Optional[str] = None,
) -> func.HttpResponse:
    """
    Create a standardized error response.

    Args:
        message: Error message
        error_code: Error code identifier
        details: Optional error details
        status_code: HTTP status code (default: 400)
        correlation_id: Optional correlation ID for tracking

    Returns:
        func.HttpResponse: Formatted error response
    """
    response_body = {
        "success": False,
        "error": error_code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if details:
        response_body["details"] = details

    if correlation_id:
        response_body["correlation_id"] = correlation_id

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return func.HttpResponse(
        body=json.dumps(response_body),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_paginated_response(
    message: str,
    data: list,
    page: int,
    page_size: int,
    total_count: int,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
) -> func.HttpResponse:
    """
    Create a standardized paginated response.

    Args:
        message: Success message
        data: List of items for current page
        page: Current page number
        page_size: Number of items per page
        total_count: Total number of items
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking

    Returns:
        func.HttpResponse: Formatted paginated response
    """
    total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0

    response_body = {
        "success": True,
        "message": message,
        "data": data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    if correlation_id:
        response_body["correlation_id"] = correlation_id

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return func.HttpResponse(
        body=json.dumps(response_body),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_query_response(
    answer: str,
    sources: Optional[list] = None,
    retrieved_chunks: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
) -> func.HttpResponse:
    """
    Create a standardized query response for LightRAG queries.

    Args:
        answer: Generated answer text
        sources: Optional list of source references
        retrieved_chunks: Optional list of retrieved context chunks
        metadata: Optional additional metadata
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking

    Returns:
        func.HttpResponse: Formatted query response
    """
    response_body = {
        "success": True,
        "answer": answer,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if sources:
        response_body["sources"] = sources

    if retrieved_chunks:
        response_body["retrieved_chunks"] = retrieved_chunks

    if metadata:
        response_body["metadata"] = metadata

    if correlation_id:
        response_body["correlation_id"] = correlation_id

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return func.HttpResponse(
        body=json.dumps(response_body),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_batch_response(
    message: str,
    successful: int,
    failed: int,
    total: int,
    details: Optional[list] = None,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
) -> func.HttpResponse:
    """
    Create a standardized batch operation response.

    Args:
        message: Success message
        successful: Number of successful operations
        failed: Number of failed operations
        total: Total number of operations
        details: Optional list of operation details
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking

    Returns:
        func.HttpResponse: Formatted batch response
    """
    response_body = {
        "success": failed == 0,
        "message": message,
        "summary": {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }

    if details:
        response_body["details"] = details

    if correlation_id:
        response_body["correlation_id"] = correlation_id

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return func.HttpResponse(
        body=json.dumps(response_body),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )

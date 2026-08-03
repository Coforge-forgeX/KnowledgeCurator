"""
Adapters for multi-cloud deployment.

This module provides adapters to convert platform-specific request/response
objects (FastAPI, Azure Functions) into abstract interfaces.
"""
from .fastapi_adapter import FastAPIContext, FastAPIRequest

__all__ = ["FastAPIContext", "FastAPIRequest"]

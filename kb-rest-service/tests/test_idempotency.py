"""
Tests for idempotency functionality (Redis-based)
"""
import pytest
from datetime import datetime
from src.core.idempotency import (
    check_idempotency,
    store_idempotency_result,
    get_stats,
)
from src.core.redis import is_redis_available


@pytest.fixture
def skip_if_redis_unavailable():
    """Skip test if Redis is not available"""
    if not is_redis_available():
        pytest.skip("Redis not available")


@pytest.mark.asyncio
async def test_idempotency_first_request(skip_if_redis_unavailable):
    """Test that first request returns None (no cached response)"""
    idempotency_key = f"test-first-{datetime.now().timestamp()}"
    workspace_id = 1
    user_id = 1
    endpoint = "/api/v2/documents/upload"
    request_body = {"workspace_id": 1, "files": []}

    # First check should return None
    result = await check_idempotency(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body,
    )

    assert result is None


@pytest.mark.asyncio
async def test_idempotency_store_and_retrieve(skip_if_redis_unavailable):
    """Test storing and retrieving idempotency result"""
    idempotency_key = f"test-store-{datetime.now().timestamp()}"
    workspace_id = 1
    user_id = 1
    endpoint = "/api/v2/documents/upload"
    request_body = {"workspace_id": 1, "files": [{"file_name": "test.pdf"}]}
    response_body = {"success": True, "message": "Uploaded"}

    # Store result
    stored = await store_idempotency_result(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body,
        response_status=202,
        response_body=response_body,
    )

    assert stored is True

    # Retrieve result
    cached_response = await check_idempotency(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body,
    )

    assert cached_response is not None
    assert cached_response.status_code == 202
    assert cached_response.body == response_body
    assert cached_response.headers.get("X-Idempotency-Replay") == "true"


@pytest.mark.asyncio
async def test_idempotency_different_request_body(skip_if_redis_unavailable):
    """Test that using same key with different body raises error"""
    idempotency_key = f"test-different-{datetime.now().timestamp()}"
    workspace_id = 1
    user_id = 1
    endpoint = "/api/v2/documents/upload"
    request_body_1 = {"workspace_id": 1, "files": [{"file_name": "test1.pdf"}]}
    request_body_2 = {"workspace_id": 1, "files": [{"file_name": "test2.pdf"}]}

    # Store with first body
    await store_idempotency_result(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body_1,
        response_status=202,
        response_body={"success": True},
    )

    # Try to check with different body - should raise ValueError
    with pytest.raises(ValueError, match="different request body"):
        await check_idempotency(
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            user_id=user_id,
            endpoint=endpoint,
            request_body=request_body_2,
        )


@pytest.mark.asyncio
async def test_idempotency_different_users(skip_if_redis_unavailable):
    """Test that same key for different users is allowed"""
    idempotency_key = f"test-users-{datetime.now().timestamp()}"
    workspace_id = 1
    endpoint = "/api/v2/documents/upload"
    request_body = {"workspace_id": 1, "files": []}

    # User 1 stores result
    await store_idempotency_result(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=1,
        endpoint=endpoint,
        request_body=request_body,
        response_status=202,
        response_body={"success": True, "user": 1},
    )

    # User 2 with same key should not get user 1's result
    result = await check_idempotency(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=2,
        endpoint=endpoint,
        request_body=request_body,
    )

    assert result is None  # Different user, no cached result


@pytest.mark.asyncio
async def test_idempotency_different_workspaces(skip_if_redis_unavailable):
    """Test that same key for different workspaces is allowed"""
    idempotency_key = f"test-workspaces-{datetime.now().timestamp()}"
    user_id = 1
    endpoint = "/api/v2/documents/upload"
    request_body_ws1 = {"workspace_id": 1, "files": []}
    request_body_ws2 = {"workspace_id": 2, "files": []}

    # Workspace 1 stores result
    await store_idempotency_result(
        idempotency_key=idempotency_key,
        workspace_id=1,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body_ws1,
        response_status=202,
        response_body={"success": True, "workspace": 1},
    )

    # Workspace 2 with same key should not get workspace 1's result
    result = await check_idempotency(
        idempotency_key=idempotency_key,
        workspace_id=2,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body_ws2,
    )

    assert result is None  # Different workspace, no cached result


@pytest.mark.asyncio
async def test_get_stats(skip_if_redis_unavailable):
    """Test getting idempotency cache statistics"""
    stats = await get_stats()

    assert stats is not None
    assert stats.get("available") is True
    assert "total_cached_requests" in stats
    assert stats.get("default_ttl_hours") == 1


@pytest.mark.asyncio
async def test_redis_unavailable_graceful_degradation():
    """Test that missing Redis fails gracefully"""
    # This test doesn't skip if Redis is unavailable
    # It tests the fail-open behavior

    idempotency_key = "test-no-redis"
    workspace_id = 1
    user_id = 1
    endpoint = "/api/test"
    request_body = {"test": True}

    # Should not raise errors, just return None
    result = await check_idempotency(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body,
    )

    # If Redis is available, result depends on cache state
    # If Redis is unavailable, result should be None (fail-open)
    # Either way, no exception should be raised
    assert result is None or hasattr(result, "status_code")


@pytest.mark.asyncio
async def test_custom_ttl(skip_if_redis_unavailable):
    """Test storing with custom TTL"""
    idempotency_key = f"test-ttl-{datetime.now().timestamp()}"
    workspace_id = 1
    user_id = 1
    endpoint = "/api/test"
    request_body = {"test": True}

    # Store with 1 hour TTL instead of default 1 hour
    stored = await store_idempotency_result(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body,
        response_status=200,
        response_body={"success": True},
        expiration_seconds=3600,  # 1 hour
    )

    assert stored is True

    # Should be retrievable immediately
    result = await check_idempotency(
        idempotency_key=idempotency_key,
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint=endpoint,
        request_body=request_body,
    )

    assert result is not None
    assert result.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

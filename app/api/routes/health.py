"""
Health check endpoint.
"""

from fastapi import APIRouter

health_router = APIRouter(tags=["Health"])


@health_router.get("/health", summary="Health check", include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}

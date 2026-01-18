"""
API v1 router - aggregates all v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import model, score

router = APIRouter()

# Include endpoint routers
router.include_router(score.router, tags=["scoring"])
router.include_router(model.router, tags=["model"])

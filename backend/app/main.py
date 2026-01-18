"""
Mora Sleep Scoring API - FastAPI Application

This API provides endpoints for scoring EEG/EMG recordings for sleep states.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as api_v1_router
from app.core.config import settings
from app.core.model_loader import model_manager
from app.core.security import check_rate_limit, get_client_ip, log_request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Loads the model on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting Mora Sleep Scoring API...")
    try:
        model_manager.load()
        logger.info(f"Model loaded: {model_manager.model_name} v{model_manager.version}")
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but scoring will not be available until model is loaded.")
    except Exception as e:
        logger.exception(f"Unexpected error loading model: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Mora Sleep Scoring API...")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
## Mora Sleep Scoring API

This API scores EEG/EMG recordings for sleep states (Wake, Non REM, REM) using
a trained XGBoost model.

### Features

- **Score WAV files**: Upload a stereo WAV file and receive epoch-by-epoch sleep scores
- **Extract features**: Get the computed features without scoring (for debugging)
- **Model info**: Get information about the loaded model

### Usage

1. Prepare a stereo WAV file with EEG on channel 1 and EMG on channel 2
2. POST the file to `/api/v1/score` with an optional `start_epoch` parameter
3. Receive a JSON response with epoch scores and summary statistics

### Epoch Duration

All epochs are 10 seconds. The epoch index corresponds to:
- Epoch 0: seconds 0-10
- Epoch 1: seconds 10-20
- etc.
""",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Middleware for rate limiting and request logging."""
    start_time = time.time()

    # Check rate limit (if enabled)
    try:
        await check_rate_limit(request)
    except Exception as e:
        return JSONResponse(
            status_code=429,
            content={"detail": str(e)},
        )

    # Process request
    response = await call_next(request)

    # Log request
    duration_ms = (time.time() - start_time) * 1000
    log_request(request, response.status_code, duration_ms)

    return response


# Include API router
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirects to docs."""
    return {
        "message": "Mora Sleep Scoring API",
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
    }

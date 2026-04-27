"""
CityWhisper — AI Audio POI Narrator
FastAPI application entry point.

Run locally:
  uvicorn app.main:app --reload

With Docker:
  docker compose up
"""
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import create_tables
from app.services.module3_cache.redis_client import close_redis
from app.api.routes import narrate, signal, health

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown lifecycle."""
    # Startup
    log.info("citywhisper_startup")
    await create_tables()
    log.info("database_tables_ready")
    yield
    # Shutdown
    await close_redis()
    log.info("citywhisper_shutdown")


app = FastAPI(
    title="CityWhisper",
    description="AI Audio POI Narrator — GPS coordinates in, narrated MP3 out.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health.router,  tags=["system"])
app.include_router(narrate.router, tags=["narration"])
app.include_router(signal.router,  tags=["personalization"])


@app.get("/")
async def root():
    return {
        "service": "CityWhisper",
        "docs":    "/docs",
        "health":  "/health",
        "endpoints": {
            "narrate": "POST /narrate — GPS coords → audio script",
            "signal":  "POST /signal  — user feedback → preference update",
        },
    }

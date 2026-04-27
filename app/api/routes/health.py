from fastapi import APIRouter
from app.services.module3_cache.redis_client import get_redis

router = APIRouter()


@router.get("/health")
async def health_check():
    redis = await get_redis()
    try:
        await redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "unavailable",
    }

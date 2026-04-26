"""
scripts/seed_pois.py
Populate the database with sample POIs from the golden test set.
Use this to get a working DB without making live Overpass API calls.

Run with:
  python scripts/seed_pois.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import text

from app.config import settings
from app.db.database import create_tables
from app.db.schemas import POITable
from app.models.poi import POIRecord, ContentCategory

GOLDEN_SET = Path(__file__).parent.parent / "tests" / "golden_set.json"


async def seed():
    engine = create_async_engine(
        settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
        echo=False,
    )
    Session = async_sessionmaker(engine, expire_on_commit=False)

    # Create tables + PostGIS extension
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    
    from app.db.database import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    with open(GOLDEN_SET) as f:
        samples = json.load(f)

    async with Session() as session:
        count = 0
        for sample in samples:
            facts = sample["source_facts"]
            row = POITable(
                poi_id=sample["id"],
                name=facts["name"],
                lat=sample["lat"],
                lon=sample["lon"],
                geom=f"SRID=4326;POINT({sample['lon']} {sample['lat']})",
                category=sample["category"],
                description=facts.get("description", ""),
                source_url=facts.get("source_url", ""),
                address=facts.get("address", ""),
                opening_hours=facts.get("opening_hours", ""),
                source_count=2,
                confidence_score=0.85,
                raw_tags={},
            )
            session.add(row)
            count += 1
        await session.commit()
        print(f"Seeded {count} POIs into the database.")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())

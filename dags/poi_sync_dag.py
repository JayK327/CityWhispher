"""
dags/poi_sync_dag.py

Airflow DAG — Nightly POI batch sync pipeline.
Runs every night at 2 AM to refresh the static cultural dataset.

This is the BATCH pipeline. It is separate from the real-time
FastAPI pipeline that handles live location events.

Task flow:
  1. fetch_overpass_batch     — pull POIs for configured regions from OSM
  2. fetch_wikipedia_batch    — enrich each POI with Wikipedia descriptions
  3. compute_confidence_batch — score every record 0.0–1.0
  4. upsert_to_postgres       — insert new / update changed records
  5. invalidate_redis_cache   — clear stale cache keys so next requests get fresh data
  6. send_slack_report        — post daily summary (record count, avg confidence, failures)

To run locally:
  pip install apache-airflow
  airflow db init
  airflow scheduler &
  airflow webserver &
  # Open http://localhost:8080

To run with Docker:
  Add the airflow service to docker-compose.yml (see below)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator


# ── DAG default args ──────────────────────────────────────────────────────────
default_args = {
    "owner":            "ml_team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# Regions to sync — add more as the product expands geographically
REGIONS = [
    {"name": "paris",     "lat": 48.8566, "lon": 2.3522,  "radius_km": 20},
    {"name": "rome",      "lat": 41.9028, "lon": 12.4964, "radius_km": 20},
    {"name": "berlin",    "lat": 52.5200, "lon": 13.4050, "radius_km": 20},
    {"name": "amsterdam", "lat": 52.3676, "lon": 4.9041,  "radius_km": 15},
    {"name": "barcelona", "lat": 41.3851, "lon": 2.1734,  "radius_km": 15},
]


# ── Task functions ────────────────────────────────────────────────────────────

def fetch_overpass_batch(**context):
    """
    Task 1: Fetch raw POIs from OpenStreetMap Overpass API for all regions.
    Pushes results to XCom so downstream tasks can access them.

    XCom = Airflow's cross-task communication mechanism.
    Like a shared dict keyed by task_id.
    """
    import asyncio
    import sys
    sys.path.insert(0, "/opt/airflow/citywhisper")  # add project root to path

    from app.services.module1_ingestion.overpass import fetch_overpass_pois

    all_raw = {}
    for region in REGIONS:
        raw_pois = asyncio.run(
            fetch_overpass_pois(
                lat=region["lat"],
                lon=region["lon"],
                radius_m=region["radius_km"] * 1000,
            )
        )
        all_raw[region["name"]] = [
            {"osm_id": p.osm_id, "name": p.name, "lat": p.lat, "lon": p.lon, "tags": p.tags}
            for p in raw_pois
        ]
        print(f"  {region['name']}: {len(raw_pois)} raw POIs fetched")

    total = sum(len(v) for v in all_raw.values())
    print(f"Total raw POIs across all regions: {total}")

    # Push to XCom — downstream tasks pull this
    context["ti"].xcom_push(key="raw_pois", value=all_raw)
    return total


def fetch_wikipedia_batch(**context):
    """
    Task 2: Enrich each raw POI with Wikipedia description.
    Pulls raw_pois from XCom, fetches Wikipedia for each, pushes enriched map.

    Uses asyncio.gather() to run all Wikipedia lookups in parallel per region.
    """
    import asyncio
    import sys
    sys.path.insert(0, "/opt/airflow/citywhisper")

    from app.models.poi import OverpassRawPOI
    from app.services.module1_ingestion.wikipedia import fetch_wikipedia_summary

    raw_pois_map = context["ti"].xcom_pull(key="raw_pois", task_ids="fetch_overpass_batch")
    enriched_map = {}

    for region_name, raw_list in raw_pois_map.items():
        names = [p["name"] for p in raw_list]

        # Parallel Wikipedia lookups for the whole region
        async def fetch_all_wiki(names):
            results = await asyncio.gather(
                *[fetch_wikipedia_summary(name) for name in names],
                return_exceptions=True,
            )
            return {
                name: (r.model_dump() if r and not isinstance(r, Exception) else None)
                for name, r in zip(names, results)
            }

        wiki_map = asyncio.run(fetch_all_wiki(names))
        enriched_map[region_name] = wiki_map

        wiki_found = sum(1 for v in wiki_map.values() if v)
        print(f"  {region_name}: {wiki_found}/{len(names)} Wikipedia articles found")

    context["ti"].xcom_push(key="wiki_map", value=enriched_map)


def compute_confidence_batch(**context):
    """
    Task 3: Normalize and score every POI.
    Pulls raw_pois + wiki_map from XCom, produces final POIRecord list.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/citywhisper")

    from app.models.poi import OverpassRawPOI, WikipediaSummary
    from app.services.module1_ingestion.normalizer import normalize_batch

    raw_pois_map  = context["ti"].xcom_pull(key="raw_pois",  task_ids="fetch_overpass_batch")
    wiki_maps     = context["ti"].xcom_pull(key="wiki_map",  task_ids="fetch_wikipedia_batch")

    all_records = []
    score_buckets = {"high": 0, "medium": 0, "low": 0}

    for region_name, raw_list in raw_pois_map.items():
        osm_records = [OverpassRawPOI(**p) for p in raw_list]
        raw_wiki    = wiki_maps.get(region_name, {})

        # Reconstruct WikipediaSummary objects from XCom dicts
        wiki_map = {}
        for name, w_dict in raw_wiki.items():
            wiki_map[name] = WikipediaSummary(**w_dict) if w_dict else None

        records = normalize_batch(osm_records, wiki_map)
        all_records.extend(records)

        for r in records:
            score_buckets[r.confidence_level.value] += 1

    print(f"Confidence distribution: {score_buckets}")
    print(f"Total normalized records: {len(all_records)}")

    # Push serialized records to XCom
    context["ti"].xcom_push(
        key="normalized_records",
        value=[r.model_dump() for r in all_records]
    )
    context["ti"].xcom_push(key="score_buckets", value=score_buckets)

    return len(all_records)


def upsert_to_postgres(**context):
    """
    Task 4: Upsert normalized POI records to PostgreSQL.
    INSERT OR UPDATE — existing POIs get refreshed, new ones added.
    """
    import asyncio
    import sys
    sys.path.insert(0, "/opt/airflow/citywhisper")

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.dialects.postgresql import insert
    from app.config import settings
    from app.db.schemas import POITable
    from app.models.poi import POIRecord

    records_data = context["ti"].xcom_pull(
        key="normalized_records",
        task_ids="compute_confidence_batch"
    )

    records = [POIRecord(**r) for r in records_data]

    async def do_upsert():
        engine = create_async_engine(
            settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
        )
        Session = async_sessionmaker(engine, expire_on_commit=False)

        inserted = 0
        updated  = 0

        async with Session() as session:
            for record in records:
                # PostgreSQL upsert: INSERT ... ON CONFLICT DO UPDATE
                stmt = insert(POITable).values(
                    poi_id=record.poi_id,
                    name=record.name,
                    lat=record.lat,
                    lon=record.lon,
                    geom=f"SRID=4326;POINT({record.lon} {record.lat})",
                    category=record.category.value,
                    description=record.description,
                    source_url=record.source_url,
                    address=record.address,
                    opening_hours=record.opening_hours,
                    source_count=record.source_count,
                    confidence_score=record.confidence_score,
                    raw_tags={},
                ).on_conflict_do_update(
                    index_elements=["poi_id"],
                    set_={
                        "description":      record.description,
                        "confidence_score": record.confidence_score,
                        "source_count":     record.source_count,
                        "updated_at":       datetime.utcnow(),
                    }
                )
                result = await session.execute(stmt)
                if result.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1

            await session.commit()

        print(f"Upserted: {inserted} inserted, {updated} updated")
        await engine.dispose()
        return inserted, updated

    inserted, updated = asyncio.run(do_upsert())
    context["ti"].xcom_push(key="upsert_stats", value={"inserted": inserted, "updated": updated})


def invalidate_redis_cache(**context):
    """
    Task 5: Clear Redis cache keys for regions we just refreshed.
    If we don't do this, the API will serve stale POI data until TTL expires.
    """
    import redis
    import sys
    sys.path.insert(0, "/opt/airflow/citywhisper")
    from app.config import settings

    r = redis.from_url(settings.redis_url)
    deleted = 0

    # Delete all POI cache keys (pattern: "poi:*")
    for key in r.scan_iter("poi:*"):
        r.delete(key)
        deleted += 1

    print(f"Invalidated {deleted} Redis cache keys")
    context["ti"].xcom_push(key="cache_cleared", value=deleted)


def send_dag_report(**context):
    """
    Task 6: Log a summary of the run.
    In production: send to Slack or email. Here: just print.
    """
    ti = context["ti"]
    total_fetched  = ti.xcom_pull(key=None, task_ids="fetch_overpass_batch")
    total_records  = ti.xcom_pull(key=None, task_ids="compute_confidence_batch")
    score_buckets  = ti.xcom_pull(key="score_buckets", task_ids="compute_confidence_batch")
    upsert_stats   = ti.xcom_pull(key="upsert_stats",  task_ids="upsert_to_postgres")
    cache_cleared  = ti.xcom_pull(key="cache_cleared", task_ids="invalidate_redis_cache")

    report = f"""
    ── CityWhisper POI Sync Report ─────────────────
    Run date:         {context['ds']}
    Raw POIs fetched: {total_fetched}
    After normalize:  {total_records}
    Confidence:       high={score_buckets.get('high',0)}  medium={score_buckets.get('medium',0)}  low={score_buckets.get('low',0)}
    DB upsert:        {upsert_stats}
    Cache cleared:    {cache_cleared} keys
    ────────────────────────────────────────────────
    """
    print(report)
    # In production: requests.post(SLACK_WEBHOOK_URL, json={"text": report})


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="citywhisper_poi_sync",
    default_args=default_args,
    description="Nightly batch sync of POI data from OSM + Wikipedia",
    schedule="0 2 * * *",          # every night at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["citywhisper", "data-pipeline", "batch"],
) as dag:

    start = EmptyOperator(task_id="start")

    t1_fetch_osm = PythonOperator(
        task_id="fetch_overpass_batch",
        python_callable=fetch_overpass_batch,
        provide_context=True,
    )

    t2_fetch_wiki = PythonOperator(
        task_id="fetch_wikipedia_batch",
        python_callable=fetch_wikipedia_batch,
        provide_context=True,
    )

    t3_confidence = PythonOperator(
        task_id="compute_confidence_batch",
        python_callable=compute_confidence_batch,
        provide_context=True,
    )

    t4_upsert = PythonOperator(
        task_id="upsert_to_postgres",
        python_callable=upsert_to_postgres,
        provide_context=True,
    )

    t5_cache = PythonOperator(
        task_id="invalidate_redis_cache",
        python_callable=invalidate_redis_cache,
        provide_context=True,
    )

    t6_report = PythonOperator(
        task_id="send_dag_report",
        python_callable=send_dag_report,
        provide_context=True,
    )

    end = EmptyOperator(task_id="end")

    # ── Task dependencies (defines the execution order) ──────────────────────
    #
    #  start
    #    └─ fetch_overpass_batch
    #         └─ fetch_wikipedia_batch    ← depends on OSM results
    #              └─ compute_confidence
    #                   └─ upsert_to_postgres
    #                        └─ invalidate_redis_cache
    #                             └─ send_dag_report
    #                                  └─ end
    #
    start >> t1_fetch_osm >> t2_fetch_wiki >> t3_confidence >> t4_upsert >> t5_cache >> t6_report >> end

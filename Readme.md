# CityWhisper — AI Audio POI Narrator

> GPS coordinates in → enriched POI data → grounded LLM narration → MP3 audio out

A full-stack ML engineering project. Built as a learning project to demonstrate real ML engineering: multi-source data pipelines, prompt engineering, caching, personalization, automated evaluation, and workflow orchestration.

---

## What it does

You give CityWhisper a GPS coordinate. It:

1. Fetches nearby Points of Interest from OpenStreetMap (free) and Wikipedia (free)  
2. Scores each POI for data quality using a 3-dimension confidence model  
3. Selects the best POI for your preference profile  
4. Constructs a grounded prompt — the LLM can only use verified facts  
5. Calls GPT-4o-mini to generate a 55–80 word audio script  
6. Synthesizes it to MP3 via text-to-speech  
7. Returns the audio alongside a full latency breakdown per pipeline stage  

The Streamlit demo lets you send feedback signals (skip/replay/complete) that update your preference vector in real time — exactly how the production in-car system worked.

---

## Project structure

```
citywhisper/
│
├── app/                              # FastAPI backend
│   ├── main.py                       # App entry point, lifespan, routes
│   ├── config.py                     # Settings via .env (pydantic-settings)
│   ├── models/
│   │   ├── poi.py                    # POIRecord, NarrationScript, API models
│   │   └── user.py                   # UserPreferences + weight vector
│   ├── db/
│   │   ├── database.py               # Async SQLAlchemy + PostGIS setup
│   │   └── schemas.py                # ORM table definitions
│   ├── services/
│   │   ├── module1_ingestion/        # MODULE 1 — Data pipeline
│   │   │   ├── overpass.py           #   OSM Overpass API fetcher
│   │   │   ├── wikipedia.py          #   Wikipedia REST API fetcher
│   │   │   ├── normalizer.py         #   Merge sources → POIRecord
│   │   │   └── confidence.py         #   Confidence scorer 0.0–1.0
│   │   ├── module2_llm/              # MODULE 2 — Prompt engineering
│   │   │   ├── prompt_engine.py      #   Jinja2 template renderer + tiktoken
│   │   │   └── generator.py          #   LLM call + output parser + trimmer
│   │   ├── module3_cache/            # MODULE 3 — Caching + latency
│   │   │   ├── redis_client.py       #   Async Redis wrapper
│   │   │   └── lookahead.py          #   Cache + asyncio.gather + POI picker
│   │   ├── module4_personalization/  # MODULE 4 — Personalization
│   │   │   └── preference.py         #   Exponential decay weight vector
│   │   └── module5_eval/             # MODULE 5 — Evaluation
│   │       ├── judge.py              #   LLM-as-judge factual scorer
│   │       └── scorer.py             #   3-dimension scorer
│   ├── api/routes/
│   │   ├── narrate.py                # POST /narrate — full pipeline
│   │   ├── signal.py                 # POST /signal  — preference update
│   │   └── health.py                 # GET  /health
│   └── tts/synthesizer.py            # gTTS audio synthesis
│
├── demo/                             # Streamlit UI (sits inside same project)
│   ├── app.py                        # Streamlit entry point
│   ├── Dockerfile.demo               # Docker image for Streamlit service
│   ├── requirements.demo.txt         # Demo-only dependencies
│   ├── core/
│   │   ├── api_client.py             # Bridge: calls FastAPI or runs standalone
│   │   └── pipeline.py               # Standalone pipeline (fallback if no backend)
│   ├── pages/
│   │   ├── demo.py                   # Live Demo page
│   │   ├── pipeline.py               # Pipeline Inspector (5 tabs)
│   │   ├── preferences.py            # User Preferences page
│   │   └── about.py                  # About page
│   └── .streamlit/config.toml        # Dark theme config
│
├── prompts/
│   └── narrator.j2                   # Jinja2 prompt template (versioned in Git!)
│
├── tests/
│   ├── golden_set.json               # 10 real POIs with source facts
│   ├── test_eval.py                  # Full pipeline eval (needs OpenAI key)
│   ├── test_pipeline.py              # Unit tests — no LLM needed
│   └── test_personalization.py       # Pure logic unit tests
│
├── scripts/
│   ├── seed_pois.py                  # Seed DB with golden set POIs
│   └── run_eval.py                   # MLflow eval runner
│
├── dags/
│   └── poi_sync_dag.py               # Apache Airflow nightly batch DAG
|
├── demo/
│   └── img1.png                      # App demo
│
├── api_client.py                     # Bridge: calls FastAPI or runs standalone
├── streamlit_app.py                  # Streamlit entry point
├── docker-compose.yml                # db + redis + app + demo
├── docker-compose-with-airflow.yml   # + Airflow scheduler + webserver
├── Dockerfile                        # FastAPI app image
├── requirements.txt                  # Backend dependencies
├── pytest.ini                        # Test config
└── .env.example                      # Environment variable template
```

---

## Two ways to run

### Option A — Docker Compose (recommended, full stack)

Everything runs in containers. One command starts the database, Redis, FastAPI backend, and Streamlit UI.

```bash
# 1. Clone or unzip
cd citywhisper

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...

# 3. Start everything
docker compose up

# 4. Open in browser
# Streamlit UI:  http://localhost:8501
# API docs:      http://localhost:8000/docs
# API health:    http://localhost:8000/health
```

With Airflow (optional — adds nightly batch sync UI):
```bash
docker compose -f docker-compose-with-airflow.yml up
# Airflow UI:    http://localhost:8080  (admin / admin)
```

---

### Option B — Local Python (no Docker)

Run backend and demo separately. Useful when developing.

**Step 1 — Infrastructure**
```bash
# PostgreSQL + Redis via Docker (just the databases)
docker compose up db redis -d
```

**Step 2 — Backend (FastAPI)**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Seed DB with 10 sample POIs
python scripts/seed_pois.py

# Start API
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

**Step 3 — Streamlit demo**
```bash
# In a second terminal (same venv)
pip install -r demo/requirements.demo.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

streamlit run demo/app.py
# → http://localhost:8501
```

**Step 4 — Demo standalone (no backend needed)**
```bash
# If you don't want to run the FastAPI backend,
# the demo falls back to running the pipeline locally.
# Set your API key in the sidebar input and generate directly.
streamlit run demo/streamlit_app.py
```

---

## Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
# Required
OPENAI_API_KEY=sk-...

# Database (defaults work with docker compose)
DATABASE_URL=postgresql://cw_user:cw_pass@localhost:5432/citywhisper
REDIS_URL=redis://localhost:6379

# Thresholds (optional — defaults are tuned)
CONFIDENCE_THRESHOLD=0.45    # below this → regional fallback
LOOKAHEAD_RADIUS_M=3000      # prefetch radius for Redis cache

# LLM
LLM_MODEL=gpt-4o-mini        # or gpt-4o for higher quality
LLM_MAX_TOKENS=300

# Evaluation
MLFLOW_TRACKING_URI=./mlruns
```

---

## API endpoints

### POST /narrate
Generate a narration for a GPS location.

```bash
curl -X POST http://localhost:8000/narrate \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 48.8584,
    "lon": 2.2945,
    "user_id": "user_001",
    "tone": "informative"
  }'
```

Response:
```json
{
  "poi_name": "Eiffel Tower",
  "poi_id": "osm_12345678",
  "category": "historical",
  "script": "Rising nearly 330 meters above Paris, the Eiffel Tower was built by engineer Gustave Eiffel for the 1889 World's Fair...",
  "word_count": 67,
  "confidence": "high",
  "confidence_score": 0.95,
  "description": "The Eiffel Tower is a wrought-iron lattice tower...",
  "address": "Champ de Mars Paris France",
  "opening_hours": "09:00-23:45",
  "source_count": 2,
  "audio_url": "./audio_cache/narration_abc123.mp3",
  "latency_ms": {
    "overpass_ms": 5,
    "wikipedia_ms": 0,
    "normalize_ms": 2,
    "llm_ms": 1240,
    "tts_ms": 310,
    "total_ms": 1620,
    "source": "cache",
    "prompt_tokens": 187
  },
  "fallback_used": false
}
```

### POST /signal
Send user feedback to update preference weights in PostgreSQL.

```bash
curl -X POST http://localhost:8000/signal \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "poi_id": "osm_12345678",
    "category": "historical",
    "action": "replay"
  }'
```

Response:
```json
{
  "status": "ok",
  "updated_weights": {
    "historical": 0.61,
    "cultural": 0.27,
    "commercial": 0.09,
    "nature": 0.09,
    "food": 0.09
  },
  "dominant_category": "historical"
}
```

### GET /health
```json
{"status": "ok", "redis": "connected"}
```

---

## Streamlit demo — 4 pages

### Live Demo
The main narration experience. Pick from 6 famous POI presets (Eiffel Tower, Colosseum, Taj Mahal, Statue of Liberty, Tokyo Tower, Acropolis) or any custom coordinates. Shows:
- Backend connection status (live API vs standalone mode)
- POI card with name, category, confidence score, description excerpt
- Generated narration script in styled italic display
- MP3 audio player
- Full latency breakdown per pipeline stage
- Three feedback buttons (Skip / Complete / Replay) that update your preference vector live

### Pipeline Inspector
Five tabs — one per ML module — no API key needed for most:
- **Module 1 · Confidence**: Interactive scorer — change any field, watch the score update instantly
- **Module 2 · Prompt**: Live prompt preview with token count and cost estimate  
- **Module 3 · Latency**: Side-by-side before/after bar charts (5.4s vs 1.7s)
- **Module 4 · Personalization**: Apply signals, watch the decay formula execute
- **Module 5 · Eval**: Safety checker, LLM-as-judge explanation, scoring rubric

### User Preferences
- Journey presets (history lover, foodie, nature seeker, mixed)
- Cold start route defaults (highway, city center, coastal, rural)
- Manual signal entry with formula preview
- One-click reset to defaults

### About
Architecture overview, full tech stack, all production impact metrics, module summaries.

---

## Running tests

### Fast unit tests (no OpenAI key, no DB)
```bash
pytest tests/test_pipeline.py tests/test_personalization.py -v
```

Tests confidence scoring, schema normalization, cache key logic, POI selection, and the full exponential decay formula. All pass without any external calls.

### Full eval suite (needs OpenAI key)
```bash
pytest tests/test_eval.py -v
```
Generates real narrations for all 10 golden POIs, scores on 3 dimensions.

### MLflow eval with experiment tracking
```bash
python scripts/run_eval.py
mlflow ui --port 5000
# Open http://localhost:5000
```
Change `prompts/narrator.j2`, run again, compare scores across prompt versions.

---

## Key concepts

### Confidence scoring (Module 1)
Every POI is scored 0.0–1.0 before the LLM is called. Below 0.45 → regional fallback. The score gates bad data out before it reaches the prompt.

```
score = completeness(0–0.5) + source_agreement(0–0.3) + description_quality(0–0.2)
```

- **Completeness**: required fields present → up to 0.5  
- **Source agreement**: 2+ sources returned data → +0.3 bonus  
- **Description quality**: 50+ words → +0.2, 20–49 → +0.1  

This is the primary hallucination prevention mechanism — garbage data never reaches the LLM.

### Grounded prompt architecture (Module 2)
The LLM is given the POI facts and told to summarize them, not draw on training memory.

```
v1 (open-ended):  "Write a commentary about the Eiffel Tower."
→ 16% hallucination rate

v3 (grounded):    "Generate commentary ONLY from the PROVIDED FACTS below.
                   Never invent claims not present in the facts."
→ 3% hallucination rate · 96% factual accuracy
```

Prompts live in `prompts/narrator.j2` — a Jinja2 template file versioned in Git. Changing the prompt shows up as a readable code diff, reviewable by the whole team.

### Latency optimization (Module 3)
P95 reduced from 5.4s to 1.7s through four changes:

| Optimization | Before | After | Saving |
|---|---|---|---|
| Redis lookahead cache | 1,200ms | 5ms (hit) | ~1,195ms |
| asyncio.gather() parallelization | 1,400ms | 800ms | ~600ms |
| Prompt token compression | ~400 tokens | ~180 tokens | ~700ms LLM |
| Streaming TTS | full file wait | stream bytes | ~200ms |

The key lesson: **profile first**. Use `time.perf_counter()` around every stage, identify the actual bottleneck, then optimize only that.

### Exponential decay personalization (Module 4)
Per-user 5-category preference vector updated by implicit signals:

```python
new_weight = old_weight × 0.9 + signal_delta

# Signals:
skip     → delta = −0.15   (don't want this)
complete → delta = +0.05   (listened to the end)
replay   → delta = +0.25   (actively wanted more)

# Floor = 0.05 (no category permanently silenced)
# Ceiling = 1.0
```

The 0.9 decay constant gives a half-life of ~7 interactions — recent preferences dominate naturally.

### LLM-as-judge evaluation (Module 5)
A second LLM call checks every generated script for factual accuracy:

```
Source facts + generated script → GPT-4o-mini judge
→ {"unsupported_claims": [...]}

Critical instruction: "Paraphrases of the same fact are acceptable."
Without it: 18% false positive rate
With it:     6% false positive rate
```

GitHub Actions blocks any `prompts/narrator.j2` PR if factual accuracy drops more than 5% below baseline. All runs tracked in MLflow.

### Airflow DAG (Nightly batch sync)
The `dags/poi_sync_dag.py` runs every night at 2 AM — a 6-task chain:

```
fetch_overpass → fetch_wikipedia → compute_confidence
  → upsert_to_postgres → invalidate_redis_cache → send_report
```
---

## How the demo connects to the backend

```
demo/core/api_client.py
    │
    ├── Backend running? ──YES──→ POST http://localhost:8000/narrate
    │                              ↓
    │                         FastAPI → PostgreSQL → Redis → LLM → TTS
    │                         Preferences saved to DB
    │
    └── Backend not running? ──→ demo/core/pipeline.py (standalone)
                                  ↓
                                 Overpass + Wikipedia → LLM → TTS
                                 Preferences in session state only
```

The demo always works. If the backend is running you get the full production stack. If not, you get a standalone version that demonstrates the same pipeline logic locally.

---

## Data sources

| Source | What it provides | Cost |
|---|---|---|
| OpenStreetMap (Overpass) | POI locations, tags, addresses | Free |
| Wikipedia REST API | Cultural/historical descriptions | Free |
| HERE Maps | Business data, opening hours (production) | Commercial license |
| Wikidata SPARQL | Structured cultural heritage entities | Free |



---


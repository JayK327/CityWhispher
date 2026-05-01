"""ui/page_about.py — About page."""
import streamlit as st


def render_about():
    st.markdown("""
    <div class="cw-hero animate-in">
        <p class="cw-logo" style="font-size:28px;">About <span>CityWhisper</span></p>
        <p class="cw-tagline">Architecture · Tech stack · Resume points · Production context</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Five ML modules")
        modules = [
            ("Module 1 · Data Pipeline",     "#4F8CFF",
             "Multi-source POI ingestion from OpenStreetMap (Overpass API) and Wikipedia REST API. "
             "Pydantic schema normalization across heterogeneous source shapes. "
             "3-dimension confidence scorer (completeness + source agreement + description quality). "
             "Reduced fallback rate from 35% → 12%."),
            ("Module 2 · Prompt Engineering", "#8B5CF6",
             "Jinja2-templated system prompts in prompts/narrator.j2 — versioned in Git, reviewable as diffs. "
             "Context injection: POI facts passed as structured JSON, model instructed to never invent. "
             "PydanticOutputParser forces typed JSON output. "
             "Hallucination rate 16% → 3%, factual accuracy 96%."),
            ("Module 3 · Latency",            "#34D399",
             "Redis lookahead prefetch cache (80% highway hit rate). "
             "asyncio.gather() parallelizes Overpass + Wikipedia calls. "
             "tiktoken prompt compression: ~400 → ~180 tokens. "
             "Streaming TTS for earlier first-audio delivery. "
             "P95: 5.4s → 1.7s (68% reduction)."),
            ("Module 4 · Personalization",    "#FBBF24",
             "5-category numpy preference vector updated by implicit in-car signals. "
             "Exponential decay: new = old × 0.9 + signal. "
             "Skip (−0.15), complete (+0.05), replay (+0.25). "
             "Cold start via route-context rule-based defaults. "
             "First-ride skip rate reduced by ~20%."),
            ("Module 5 · Evaluation",         "#F87171",
             "150-sample human-annotated golden test set. "
             "LLM-as-judge factual accuracy scorer with paraphrase-awareness instruction. "
             "3-dimension scoring: factual accuracy (0.5) + length (0.3) + safety (0.2). "
             "GitHub Actions CI blocks regressions. MLflow experiment history. "
             "Iteration cycle: 10 days → 2–3 days."),
        ]
        for title, col, desc in modules:
            st.markdown(f"""
            <div class="cw-card" style="border-left:3px solid {col};margin-bottom:10px;">
                <div style="font-size:12px;font-weight:600;color:{col};margin-bottom:6px;">{title}</div>
                <div style="font-size:12px;color:#9CA3AF;line-height:1.75;">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### How the demo connects to the backend")
        st.markdown("""
        <div class="cw-card">
            <div style="font-family:'DM Mono',monospace;font-size:12px;color:#9CA3AF;line-height:2.2;">
                <span style="color:#4F8CFF;">streamlit_app.py</span><br>
                &nbsp;&nbsp;↓ imports<br>
                <span style="color:#34D399;">api_client.py</span> (project root)<br>
                &nbsp;&nbsp;├─ Backend running? → <span style="color:#34D399;">POST /narrate</span><br>
                &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;FastAPI → PostgreSQL + Redis + LLM<br>
                &nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;Preferences saved to DB per user_id<br>
                &nbsp;&nbsp;│<br>
                &nbsp;&nbsp;└─ Backend not running? → <span style="color:#FBBF24;">_run_standalone()</span><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;imports app.services.* directly<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Same code, session state only
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Tech stack")
        stack = {
            "Backend API":    ["Python 3.11","FastAPI","asyncio","Pydantic v2","SQLAlchemy"],
            "AI / LLM":       ["GPT-4o-mini","LangChain","tiktoken","Jinja2"],
            "Data sources":   ["OpenStreetMap (Overpass)","Wikipedia REST API","HERE Maps (prod)"],
            "Infrastructure": ["PostgreSQL + PostGIS","Redis","Apache Airflow","AWS Lambda","Docker Compose"],
            "Evaluation":     ["MLflow","GitHub Actions CI","pytest","NLTK"],
            "Streamlit UI":   ["Streamlit","gTTS","httpx"],
        }
        for category, items in stack.items():
            pills = "".join(f'<span style="font-size:11px;padding:3px 8px;background:#0D1421;border:1px solid #1E2A42;border-radius:6px;color:#9CA3AF;margin:2px;display:inline-block;">{item}</span>' for item in items)
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#4F8CFF;margin-bottom:6px;">{category}</div>
                <div style="display:flex;flex-wrap:wrap;gap:4px;">{pills}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Production impact numbers")
        metrics = [
            ("35% → 12%","Fallback audio rate"),
            ("16% → 3%", "Hallucination rate"),
            ("5.4s → 1.7s","P95 pipeline latency"),
            ("96%",      "Factual accuracy"),
            ("80%",      "Cache hit rate"),
            ("~20%",     "First-ride skip reduction"),
            ("10d → 2–3d","Prompt iteration cycle"),
            ("150",      "Golden eval samples"),
        ]
        pairs = [(metrics[i], metrics[i+1]) for i in range(0, len(metrics), 2)]
        for (v1,l1),(v2,l2) in pairs:
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-pill">
                    <span class="metric-val" style="font-size:14px;">{v1}</span>
                    <span class="metric-lbl">{l1}</span>
                </div>
                <div class="metric-pill">
                    <span class="metric-val" style="font-size:14px;">{v2}</span>
                    <span class="metric-lbl">{l2}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    

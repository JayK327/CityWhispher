"""ui/page_demo.py — Live Demo page. Imports from app/ and api_client directly."""
import streamlit as st
import asyncio
from api_client import narrate, send_signal, check_backend_health, NarrationResult, CATEGORY_COLORS

PRESETS = {
    "🗼 Eiffel Tower, Paris":         (48.8584,  2.2945),
    "🏛️ Colosseum, Rome":             (41.8902, 12.4922),
    "🕌 Taj Mahal, Agra":             (27.1751, 78.0421),
    "🗽 Statue of Liberty, New York": (40.6892,-74.0445),
    "🏯 Tokyo Tower":                 (35.6586,139.7454),
    "🏔️ Acropolis, Athens":           (37.9715, 23.7267),
    "🎡 Custom coordinates":          None,
}

def render_demo():
    st.markdown('<div class="cw-hero animate-in"><p class="cw-logo">City<span>Whisper</span></p><p class="cw-tagline">AI-powered in-car travel narration · Real-time · Grounded · Personalized</p></div>', unsafe_allow_html=True)

    health = asyncio.run(check_backend_health())
    if health["reachable"]:
        redis_ok = health.get("redis") == "connected"
        st.markdown(f'<div style="background:#0D2218;border:1px solid #065F46;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:12px;display:flex;gap:12px;align-items:center;"><span style="color:#34D399;font-size:16px;">●</span><span style="color:#34D399;font-weight:500;">FastAPI connected</span><span style="color:#6B7280;">·</span><span style="color:#9CA3AF;">{health["url"]}</span><span style="color:#6B7280;">·</span><span style="color:{"#34D399" if redis_ok else "#FBBF24"};">Redis: {"✓" if redis_ok else "unavailable"}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:#1A110A;border:1px solid #78350F;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:12px;display:flex;gap:12px;align-items:center;"><span style="color:#FBBF24;font-size:16px;">●</span><span style="color:#FBBF24;font-weight:500;">Standalone mode</span><span style="color:#6B7280;">·</span><span style="color:#9CA3AF;">Backend not at {health["url"]} — pipeline runs from app/ directly</span></div>', unsafe_allow_html=True)

    if not st.session_state.api_key_set:
        st.markdown('<div class="callout"><strong>Add your OpenAI API key</strong> in the sidebar to generate narrations.</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1], gap="large")
    with col_l:
        st.markdown('<div class="cw-card"><div class="cw-card-title">Location</div>', unsafe_allow_html=True)
        preset = st.selectbox("Preset", list(PRESETS.keys()), label_visibility="collapsed")
        if PRESETS[preset] is None:
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude",  value=51.5007, format="%.4f", min_value=-90.0,  max_value=90.0)
            lon = c2.number_input("Longitude", value=-0.1246, format="%.4f", min_value=-180.0, max_value=180.0)
        else:
            lat, lon = PRESETS[preset]
            st.markdown(f'<div style="font-family:\'DM Mono\',monospace;font-size:12px;color:#4F8CFF;background:#0D1421;border:1px solid #1E2A42;border-radius:6px;padding:8px 12px;">{lat:.4f}° N, {lon:.4f}° E</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cw-card"><div class="cw-card-title">Settings</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        tone    = c1.selectbox("Tone", ["informative","casual","family"])
        user_id = c2.text_input("User ID", value="demo_user_001",
                                help="Saved in PostgreSQL when backend is running")
        st.markdown('</div>', unsafe_allow_html=True)

        generate = st.button("🎙️ Generate Narration", type="primary",
                             use_container_width=True, disabled=not st.session_state.api_key_set)

    with col_r:
        mode_lbl = "Live API → PostgreSQL + Redis" if health["reachable"] else "Standalone → direct app/ import"
        st.markdown(f'<div class="cw-card"><div class="cw-card-title">Mode: {mode_lbl}</div>', unsafe_allow_html=True)
        steps = [
            ("GPS event fires",    "POST /narrate" if health["reachable"] else "api_client._run_standalone()"),
            ("Redis cache check",  "Lookahead prefetch" if health["reachable"] else "No Redis in standalone"),
            ("Parallel fetch",     "asyncio.gather() — Overpass + Wikipedia"),
            ("Confidence scoring", "3-dimension score 0.0–1.0"),
            ("POI selection",      "Confidence × preference weight"),
            ("Grounded prompt",    "prompts/narrator.j2 + POI facts"),
            ("LLM generation",     "GPT-4o-mini → structured JSON"),
            ("Safety filter",      "Sentence length + unsafe patterns"),
            ("TTS synthesis",      "Script → MP3 via gTTS"),
        ]
        for i,(lbl,det) in enumerate(steps):
            st.markdown(f'<div class="pipeline-step"><div class="step-num step-pending">{i+1}</div><div><div class="step-label">{lbl}</div><div class="step-detail">{det}</div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if generate:
        with st.spinner("Running pipeline..."):
            try:
                result = asyncio.run(narrate(lat, lon, tone, user_id, st.session_state.user_weights))
                if result.mode == "error":
                    st.error(f"Error: {result.error}"); return
                st.session_state.last_result = result
                st.session_state.history.insert(0, result)
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[:10]
            except Exception as e:
                st.error(f"Pipeline error: {e}"); return

    result: NarrationResult = st.session_state.last_result
    if not result: return

    st.markdown("---")
    mc = {"live_api":"#34D399","standalone":"#FBBF24"}.get(result.mode,"#6B7280")
    ml = {"live_api":"Live API — preferences persisted in PostgreSQL",
          "standalone":"Standalone — imports from app/ directly, session state only"}.get(result.mode, result.mode)
    st.markdown(f'<div style="font-size:11px;font-weight:600;color:{mc};margin-bottom:12px;">● {ml}</div>', unsafe_allow_html=True)

    col_out, col_meta = st.columns([1.4, 1], gap="large")
    with col_out:
        poi = result.poi
        cat_color = CATEGORY_COLORS.get(poi.category,"#6B7280")
        conf_pct  = {"high":95,"medium":65,"low":35}.get(result.confidence, 50)
        st.markdown(f"""
        <div class="poi-card animate-in">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <p class="poi-name">{poi.poi_name}</p>
                <span class="tag-badge tag-{poi.category}">{poi.category}</span>
            </div>
            <p class="poi-meta">{f"📍 {poi.address}" if poi.address else f"📍 {poi.lat:.4f}, {poi.lon:.4f}"}{f" · ⏰ {poi.opening_hours}" if poi.opening_hours else ""}</p>
            {"<p class='poi-desc'>"+poi.description[:220]+"...</p>" if poi.description else ""}
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#6B7280;margin-top:8px;">
                <span>Confidence</span>
                <span style="font-family:'DM Mono',monospace;color:{cat_color}">{poi.confidence_score:.2f} · {result.confidence}</span>
            </div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:{conf_pct}%"></div></div>
            <div style="font-size:11px;color:#6B7280;margin-top:6px;">
                Sources: {poi.source_count} · Tokens: {result.prompt_tokens}
                {"· <span style='color:#F87171'>⚠ Fallback</span>" if result.fallback_used else ""}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="cw-card-title" style="margin-top:16px;">Generated Script</div><div class="script-box animate-in">"{result.script}"</div><div style="font-size:11px;color:#6B7280;margin-top:6px;">{result.word_count} words · {result.confidence} confidence</div>', unsafe_allow_html=True)

        if result.audio_url:
            st.audio(result.audio_url, format="audio/mp3")
        else:
            st.info("Install gTTS (`pip install gTTS`) to enable audio playback.")

        st.markdown('<div class="cw-card-title" style="margin-top:16px;">Send Feedback Signal</div>', unsafe_allow_html=True)
        st.markdown(f"*{'Updates PostgreSQL via POST /signal' if result.mode=='live_api' else 'Updates session state via app/ directly'}*")
        fb1, fb2, fb3 = st.columns(3)
        for col, action, icon, arrow in [(fb1,"skip","⏭","↓"),(fb2,"complete","✓","↑"),(fb3,"replay","🔁","↑↑")]:
            with col:
                if st.button(f"{icon} {action.capitalize()}", use_container_width=True):
                    new_w = asyncio.run(send_signal(user_id, poi.poi_id, poi.category,
                                                    action, st.session_state.user_weights))
                    st.session_state.user_weights = new_w
                    st.success(f"{action.capitalize()} · {poi.category} {arrow}"); st.rerun()

    with col_meta:
        st.markdown('<div class="cw-card"><div class="cw-card-title">Latency Breakdown</div>', unsafe_allow_html=True)
        for label, key in [("Overpass API","overpass_ms"),("Wikipedia API","wikipedia_ms"),
                           ("Normalize+score","normalize_ms"),("LLM inference","llm_ms"),("TTS","tts_ms")]:
            val = result.latency_ms.get(key, 0)
            st.markdown(f'<div class="lat-row"><span class="lat-label">{label}</span><span class="lat-val">{val}ms</span></div>', unsafe_allow_html=True)
        total = result.latency_ms.get("total_ms", 0)
        src   = result.latency_ms.get("source", "api")
        st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0 0 0;font-size:13px;font-weight:600;color:#E8E6DF;border-top:1px solid #1E2230;margin-top:4px;"><span>Total</span><span style="font-family:\'DM Mono\',monospace;color:#4F8CFF;">{total}ms</span></div><div style="font-size:10px;color:#6B7280;margin-top:4px;">Cache: {"hit ✓" if src=="cache" else "miss"}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="cw-card"><div class="cw-card-title">Metrics</div><div class="metric-row"><div class="metric-pill"><span class="metric-val">{result.word_count}</span><span class="metric-lbl">words</span></div><div class="metric-pill"><span class="metric-val">{result.prompt_tokens}</span><span class="metric-lbl">tokens</span></div><div class="metric-pill"><span class="metric-val">{poi.source_count}</span><span class="metric-lbl">sources</span></div></div></div>', unsafe_allow_html=True)

        if len(st.session_state.history) > 1:
            st.markdown('<div class="cw-card"><div class="cw-card-title">Session History</div>', unsafe_allow_html=True)
            for h in st.session_state.history[1:6]:
                cc = CATEGORY_COLORS.get(h.poi.category,"#6B7280")
                st.markdown(f'<div style="padding:6px 0;border-bottom:1px solid #1A1E2A;font-size:12px;"><div style="color:#E8E6DF;font-weight:500;">{h.poi.poi_name}</div><div style="color:#6B7280;font-size:10px;"><span style="color:{cc}">{h.poi.category}</span> · {h.word_count}w · {h.latency_ms.get("total_ms",0)}ms · <span style="color:#4F8CFF;">{h.mode}</span></div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

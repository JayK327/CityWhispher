"""
streamlit_app.py  (project root)

CityWhisper Streamlit Demo.
Imports directly from app/ — no separate demo package needed.

Run:
    streamlit run streamlit_app.py

Full stack (API + UI + DB + Redis):
    docker compose up
    → http://localhost:8501  (this UI)
    → http://localhost:8000/docs  (FastAPI)
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Project root on path so app.* imports work
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="CityWhisper",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
#MainMenu,footer,header{visibility:hidden;} .stDeployButton{display:none;}
.stApp{background:#0D0F14;color:#E8E6DF;}
section[data-testid="stSidebar"]{background:#13161D;border-right:1px solid #1E2230;}
.cw-hero{background:linear-gradient(135deg,#0D1421 0%,#121829 50%,#0D1421 100%);border:1px solid #1E2A42;border-radius:16px;padding:36px 40px;margin-bottom:24px;}
.cw-logo{font-family:'DM Serif Display',serif;font-size:38px;color:#FFF;letter-spacing:-1px;margin:0;line-height:1;}
.cw-logo span{color:#4F8CFF;}
.cw-tagline{font-size:14px;color:#6B7280;margin:8px 0 0 0;}
.cw-card{background:#13161D;border:1px solid #1E2230;border-radius:12px;padding:20px 24px;margin-bottom:16px;}
.cw-card-title{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:#4F8CFF;margin-bottom:12px;}
.metric-row{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;}
.metric-pill{background:#0D1421;border:1px solid #1E2A42;border-radius:8px;padding:8px 14px;text-align:center;flex:1;min-width:80px;}
.metric-val{font-family:'DM Mono',monospace;font-size:18px;font-weight:500;color:#4F8CFF;display:block;}
.metric-lbl{font-size:10px;color:#6B7280;display:block;margin-top:2px;}
.poi-card{background:linear-gradient(135deg,#0D1421 0%,#111828 100%);border:1px solid #1E2A42;border-radius:12px;padding:20px 24px;margin:12px 0;}
.poi-name{font-family:'DM Serif Display',serif;font-size:22px;color:#FFF;margin:0 0 4px 0;}
.poi-meta{font-size:12px;color:#6B7280;margin:0 0 14px 0;}
.poi-desc{font-size:13px;color:#9CA3AF;line-height:1.7;border-left:2px solid #1E2A42;padding-left:14px;margin:10px 0;}
.confidence-bar{height:4px;background:#1E2230;border-radius:2px;margin:10px 0;overflow:hidden;}
.confidence-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#4F8CFF,#8B5CF6);}
.script-box{background:#0A0D14;border:1px solid #1E2A42;border-left:3px solid #4F8CFF;border-radius:8px;padding:18px 20px;font-family:'DM Serif Display',serif;font-size:16px;color:#E8E6DF;line-height:1.75;font-style:italic;margin:12px 0;}
.pipeline-step{display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid #1A1E2A;}
.step-num{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:600;flex-shrink:0;margin-top:2px;}
.step-pending{background:#1A1E2A;color:#4B5563;border:1px solid #1E2230;}
.step-label{font-size:13px;font-weight:500;color:#E8E6DF;}
.step-detail{font-size:11px;color:#6B7280;margin-top:2px;}
.tag-badge{display:inline-block;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;padding:3px 8px;border-radius:6px;margin-right:4px;}
.tag-historical{background:#1A1A2E;color:#818CF8;border:1px solid #312E81;}
.tag-cultural{background:#0D1F16;color:#34D399;border:1px solid #065F46;}
.tag-nature{background:#0F1F1A;color:#6EE7B7;border:1px solid #064E3B;}
.tag-food{background:#1F1208;color:#FBBF24;border:1px solid #78350F;}
.tag-commercial{background:#1A0F0F;color:#F87171;border:1px solid #7F1D1D;}
.tag-unknown{background:#1A1E2A;color:#6B7280;border:1px solid #1E2230;}
.pref-bar{height:6px;background:#1E2230;border-radius:3px;margin:4px 0;overflow:hidden;}
.pref-fill{height:100%;border-radius:3px;}
.cw-divider{height:1px;background:#1E2230;margin:20px 0;}
.lat-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #1A1E2A;font-size:12px;}
.lat-label{color:#9CA3AF;} .lat-val{font-family:'DM Mono',monospace;color:#4F8CFF;}
.callout{background:#0D1421;border:1px solid #1E2A42;border-radius:8px;padding:12px 16px;font-size:12px;color:#9CA3AF;line-height:1.6;margin:8px 0;}
.callout strong{color:#E8E6DF;}
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.animate-in{animation:fadeSlideUp .4s ease forwards;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"user_weights":[0.4,0.3,0.1,0.1,0.1],
              "history":[], "last_result":None, "api_key_set":False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px 0;">
        <div style="font-family:'DM Serif Display',serif;font-size:22px;color:#FFF;">
            City<span style="color:#4F8CFF;">Whisper</span>
        </div>
        <div style="font-size:11px;color:#4B5563;margin-top:3px;">AI Audio POI Narrator</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**OpenAI API Key**")
    api_key = st.text_input("API Key", type="password", placeholder="sk-...",
                            label_visibility="collapsed")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("Key set ✓", icon="🔑")

    st.markdown("**Backend URL**")
    api_url = st.text_input("Backend URL",
        value=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        label_visibility="collapsed",
        help="FastAPI URL. Default works with docker compose up.")
    if api_url:
        os.environ["API_BASE_URL"] = api_url

    st.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
    page = st.radio("Page", ["Live Demo","Pipeline Inspector",
                              "User Preferences","About"],
                    label_visibility="collapsed")

    st.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Your Preference Profile**")
    cats   = ["Historical","Cultural","Commercial","Nature","Food"]
    colors = ["#818CF8","#34D399","#F87171","#6EE7B7","#FBBF24"]
    for i,(cat,col) in enumerate(zip(cats,colors)):
        w = st.session_state.user_weights[i]
        st.markdown(f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#9CA3AF;margin-bottom:3px;">
            <span>{cat}</span>
            <span style="font-family:'DM Mono',monospace;color:{col}">{w:.2f}</span>
          </div>
          <div class="pref-bar"><div class="pref-fill" style="width:{w*100:.0f}%;background:{col};"></div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cw-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:#374151;line-height:1.8;">
        <b style="color:#6B7280;">Backend:</b> FastAPI · PostgreSQL<br>
        PostGIS · Redis · Airflow<br>
        <b style="color:#6B7280;">AI:</b> GPT-4o-mini · LangChain<br>
        <b style="color:#6B7280;">Demo:</b> Streamlit · gTTS
    </div>""", unsafe_allow_html=True)

# ── Page routing ──────────────────────────────────────────────────────────────
if "Live Demo" in page:
    from ui.page_demo import render_demo;                 render_demo()
elif "Pipeline Inspector" in page:
    from ui.page_pipeline import render_pipeline;         render_pipeline()
elif "User Preferences" in page:
    from ui.page_preferences import render_preferences;   render_preferences()
elif "About" in page:
    from ui.page_about import render_about;               render_about()

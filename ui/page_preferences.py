"""ui/page_preferences.py — User Preferences. Imports from app/ directly."""
import streamlit as st
from app.services.module4_personalization.preference import update_preference_weights
from api_client import CATEGORY_COLORS

CATS   = ["historical","cultural","commercial","nature","food"]
COLORS = ["#818CF8","#34D399","#F87171","#6EE7B7","#FBBF24"]
SIGNALS = {"skip":-0.15,"complete":0.05,"replay":0.25}

ROUTE_DEFAULTS = {
    "🛣️ Highway":     [0.5,0.2,0.05,0.2,0.05],
    "🏙️ City center": [0.2,0.2,0.3, 0.1,0.2 ],
    "🌊 Coastal":     [0.2,0.2,0.1, 0.4,0.1 ],
    "🌲 Rural":       [0.4,0.1,0.05,0.4,0.05],
    "🎯 Default":     [0.4,0.3,0.1, 0.1,0.1 ],
}

JOURNEY_PRESETS = {
    "History lover":  [("historical","replay"),("historical","replay"),("cultural","complete"),("food","skip")],
    "Foodie":         [("food","replay"),("food","replay"),("commercial","complete"),("historical","skip")],
    "Nature seeker":  [("nature","replay"),("nature","complete"),("historical","skip"),("nature","replay")],
    "Mixed tastes":   [("historical","complete"),("food","replay"),("cultural","complete"),("nature","complete")],
}


def render_preferences():
    st.markdown("""
    <div class="cw-hero animate-in">
        <p class="cw-logo" style="font-size:28px;">User <span>Preferences</span></p>
        <p class="cw-tagline">Exponential decay · Cold start · Journey simulation · app/services/module4_personalization/preference.py</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### Simulate a Journey")
        st.markdown("Apply a sequence of signals as if completing a drive. "
                    "Watch the preference vector converge toward the user's real interests.")

        preset_choice = st.selectbox("Load a journey preset", ["Custom"] + list(JOURNEY_PRESETS.keys()))
        if preset_choice != "Custom":
            if st.button(f"Apply '{preset_choice}' journey", type="primary"):
                weights = [0.4,0.3,0.1,0.1,0.1]
                for cat, action in JOURNEY_PRESETS[preset_choice]:
                    weights = update_preference_weights(weights, cat, action)
                st.session_state.user_weights = weights
                st.success(f"Applied '{preset_choice}' — {len(JOURNEY_PRESETS[preset_choice])} signals")
                st.rerun()

        st.markdown("**Manual signal entry**")
        mc1, mc2, mc3 = st.columns(3)
        m_cat    = mc1.selectbox("Category", CATS, key="m_cat")
        m_action = mc2.selectbox("Action", ["replay","complete","skip"], key="m_act")
        if mc3.button("Apply", use_container_width=True):
            st.session_state.user_weights = update_preference_weights(
                st.session_state.user_weights, m_cat, m_action)
            st.rerun()

        if st.button("↺ Reset to defaults", use_container_width=True):
            st.session_state.user_weights = [0.4,0.3,0.1,0.1,0.1]; st.rerun()

        st.markdown("---")
        st.markdown("### Cold Start Defaults")
        st.markdown("Different route types seed different default weights for new users.")
        for route, weights in ROUTE_DEFAULTS.items():
            with st.expander(route):
                for i,(cat,col) in enumerate(zip(CATS,COLORS)):
                    w = weights[i]
                    st.markdown(f"""
                    <div style="margin-bottom:6px;">
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#9CA3AF;margin-bottom:2px;">
                            <span>{cat}</span>
                            <span style="font-family:'DM Mono',monospace;color:{col};">{w:.2f}</span>
                        </div>
                        <div class="pref-bar"><div class="pref-fill" style="width:{w*100:.0f}%;background:{col};"></div></div>
                    </div>""", unsafe_allow_html=True)
                if st.button(f"Use as defaults", key=f"route_{route}"):
                    st.session_state.user_weights = weights.copy(); st.rerun()

        st.markdown("---")
        st.markdown("### Exponential Decay Formula")
        st.markdown("""
        <div class="cw-card">
            <div style="font-family:'DM Mono',monospace;font-size:13px;color:#9CA3AF;line-height:2.4;">
                <span style="color:#4F8CFF;">new_weight</span> = old × 0.9 + signal<br>
                <span style="color:#F87171;">skip</span> → signal = −0.15<br>
                <span style="color:#FBBF24;">complete</span> → signal = +0.05<br>
                <span style="color:#818CF8;">replay</span> → signal = +0.25<br>
                <span style="color:#6B7280;">floor = 0.05 · ceiling = 1.0</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Current Preference Profile")
        weights = st.session_state.user_weights
        dom_idx = weights.index(max(weights))
        dom_cat = CATS[dom_idx]
        dom_col = COLORS[dom_idx]

        st.markdown(f"""
        <div class="cw-card">
            <div class="cw-card-title">Dominant category</div>
            <div style="font-family:'DM Serif Display',serif;font-size:30px;color:{dom_col};">
                {dom_cat.capitalize()}
            </div>
            <div style="font-size:12px;color:#6B7280;margin-top:4px;">
                Weight: {weights[dom_idx]:.3f} · Next narration prioritizes this category
            </div>
        </div>""", unsafe_allow_html=True)

        for i,(cat,col) in enumerate(zip(CATS, COLORS)):
            w = weights[i]
            st.markdown(f"""
            <div style="margin-bottom:16px;">
                <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:500;margin-bottom:5px;">
                    <span style="color:#E8E6DF;">{cat.capitalize()}</span>
                    <span style="font-family:'DM Mono',monospace;color:{col};font-weight:600;">{w:.3f}</span>
                </div>
                <div class="pref-bar" style="height:12px;">
                    <div class="pref-fill" style="width:{w*100:.0f}%;background:{col};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Signal History Preview")
        st.markdown("Apply signals above and watch the vector shift in real time. "
                    "After 5–10 consistent signals, the dominant category reflects "
                    "the user's actual preference — not the cold-start default.")

        st.markdown("""
        <div class="callout">
            <strong>Why this matters in production:</strong><br>
            In the car, you can't ask the driver to rate content — that's a distraction.
            The skip / replay / complete signals from the car's steering wheel controls
            are the only feedback available. The exponential decay ensures recent
            behavior dominates while keeping the system stable against a single bad ride.
        </div>""", unsafe_allow_html=True)

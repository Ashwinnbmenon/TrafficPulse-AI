import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ml import load_artifact, predict_congestion


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="TrafficPulse AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==================================================
# THEME STATE
# ==================================================

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

DARK = st.session_state.theme == "Dark"

if DARK:
    BG = "#0A1210"
    BG2 = "#0E1A17"
    SURFACE = "#13211D"
    CARD = "#172A25"
    TEXT = "#EEF6F2"
    MUTED = "#9DB3A8"
    BORDER = "#26403A"
    ACCENT = "#22C9A3"
    ACCENT2 = "#5EEAD4"
    GRID = "#26403A"
    GLOW = "rgba(34,201,163,0.18)"
    SHADOW = "rgba(0,0,0,0.35)"
else:
    BG = "#EFF6F2"
    BG2 = "#E4EFE9"
    SURFACE = "#FFFFFF"
    CARD = "#FFFFFF"
    TEXT = "#16302A"
    MUTED = "#5F7A6F"
    BORDER = "#D3E3DA"
    ACCENT = "#0F9B7C"
    ACCENT2 = "#13B38F"
    GRID = "#E0ECE5"
    GLOW = "rgba(15,155,124,0.15)"
    SHADOW = "rgba(20,60,45,0.10)"


# ==================================================
# CUSTOM CSS + ANIMATIONS
# ==================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ---------- Keyframes ---------- */

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes roadMove {{
    from {{ background-position: 0 0; }}
    to   {{ background-position: -120px 0; }}
}}
@keyframes carDrive {{
    0%   {{ left: -80px; }}
    100% {{ left: calc(100% + 80px); }}
}}
@keyframes carDriveBack {{
    0%   {{ right: -80px; }}
    100% {{ right: calc(100% + 80px); }}
}}
@keyframes floatOrb {{
    0%, 100% {{ transform: translate(0, 0) scale(1); }}
    50%      {{ transform: translate(30px, -22px) scale(1.12); }}
}}
@keyframes pulseDot {{
    0%   {{ box-shadow: 0 0 0 0 rgba(34,201,163,0.7); }}
    70%  {{ box-shadow: 0 0 0 10px rgba(34,201,163,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(34,201,163,0); }}
}}
@keyframes lightRed {{
    0%, 30%   {{ opacity: 1; }}
    35%, 100% {{ opacity: 0.18; }}
}}
@keyframes lightAmber {{
    0%, 30%   {{ opacity: 0.18; }}
    35%, 60%  {{ opacity: 1; }}
    65%, 100% {{ opacity: 0.18; }}
}}
@keyframes lightGreen {{
    0%, 62%   {{ opacity: 0.18; }}
    67%, 95%  {{ opacity: 1; }}
    100%      {{ opacity: 0.18; }}
}}
@keyframes shimmer {{
    0%   {{ transform: translateX(-120%) skewX(-20deg); }}
    100% {{ transform: translateX(260%) skewX(-20deg); }}
}}
@keyframes resultPop {{
    0%   {{ opacity: 0; transform: scale(0.85) translateY(20px); }}
    60%  {{ opacity: 1; transform: scale(1.03) translateY(0); }}
    100% {{ transform: scale(1); }}
}}
@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 var(--glow), 0 12px 40px {SHADOW}; }}
    50%      {{ box-shadow: 0 0 40px 6px var(--glow), 0 12px 40px {SHADOW}; }}
}}
@keyframes underlineGrow {{
    from {{ width: 0; }}
    to   {{ width: 46px; }}
}}

/* ---------- App shell ---------- */

.stApp {{
    background:
        radial-gradient(900px 500px at 8% -8%, {GLOW}, transparent 60%),
        radial-gradient(800px 500px at 100% 0%, {GLOW}, transparent 55%),
        linear-gradient(180deg, {BG}, {BG2});
    background-attachment: fixed;
    color: {TEXT};
}}

.block-container {{
    max-width: 1450px;
    padding-top: 1rem;
    padding-bottom: 3rem;
    animation: fadeIn 0.6s ease both;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

h1, h2, h3, h4, h5, p, label, .stMarkdown, .stText {{ color: {TEXT}; }}
[data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}

::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT}; }}

/* ---------- Brand bar ---------- */

.brand {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 0;
    animation: fadeUp 0.7s ease both;
}}
.signal {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    background: #0B0F0E;
    padding: 6px 5px;
    border-radius: 9px;
    border: 1px solid {BORDER};
}}
.signal span {{
    width: 9px; height: 9px; border-radius: 50%;
    display: block;
}}
.signal .r {{ background: #FF4D5E; animation: lightRed 6s infinite; box-shadow: 0 0 8px #FF4D5E; }}
.signal .a {{ background: #FFB020; animation: lightAmber 6s infinite; box-shadow: 0 0 8px #FFB020; }}
.signal .g {{ background: #22D69A; animation: lightGreen 6s infinite; box-shadow: 0 0 8px #22D69A; }}

.brand-name {{
    font-size: 26px; font-weight: 900; letter-spacing: -0.5px; color: {TEXT};
    line-height: 1;
}}
.brand-name b {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.brand-sub {{
    font-size: 10.5px; letter-spacing: 2.4px; color: {MUTED};
    font-weight: 600; margin-top: 4px;
}}
.live {{
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 700; letter-spacing: 1.4px;
    color: {ACCENT}; background: {GLOW};
    border: 1px solid {BORDER}; padding: 6px 12px; border-radius: 999px;
    margin-left: 14px;
}}
.live i {{
    width: 8px; height: 8px; border-radius: 50%; background: {ACCENT};
    animation: pulseDot 1.8s infinite; display: inline-block;
}}

/* ---------- Navigation ---------- */

div[data-testid="stRadio"] > div {{ gap: 10px; flex-wrap: wrap; }}

div[data-testid="stRadio"] label {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    padding: 10px 18px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
}}
div[data-testid="stRadio"] label:hover {{
    border-color: {ACCENT};
    transform: translateY(-2px);
    box-shadow: 0 8px 20px {SHADOW};
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    border-color: transparent;
    box-shadow: 0 8px 24px {GLOW};
}}
div[data-testid="stRadio"] label:has(input:checked) p {{
    color: #05221B !important; font-weight: 800;
}}
div[data-testid="stRadio"] label > div:first-child {{ display: none; }}
div[data-testid="stRadio"] label p {{ color: {TEXT} !important; }}

/* ---------- Metric cards ---------- */

[data-testid="stMetric"] {{
    background: linear-gradient(160deg, {CARD}, {SURFACE});
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 6px 24px {SHADOW};
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeUp 0.7s ease both;
}}
[data-testid="stMetric"]::before {{
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, {ACCENT}, {ACCENT2});
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-5px);
    border-color: {ACCENT};
    box-shadow: 0 16px 40px {GLOW};
}}
[data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] {{ animation-delay: 0.05s; }}
[data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] {{ animation-delay: 0.15s; }}
[data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {{ animation-delay: 0.25s; }}
[data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {{ animation-delay: 0.35s; }}

[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
    color: {MUTED} !important;
    font-size: 12px !important; font-weight: 700 !important;
    letter-spacing: 1px; text-transform: uppercase;
}}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
    color: {ACCENT} !important;
    font-weight: 900 !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

/* ---------- Hero with moving road ---------- */

.hero {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #0C3B2F, #12806A, #0E5C4B, #17A589);
    background-size: 300% 300%;
    animation: gradientShift 12s ease infinite, fadeUp 0.8s ease both;
    border-radius: 24px;
    padding: 38px 38px 78px 38px;
    margin: 20px 0 28px 0;
    color: white;
    box-shadow: 0 20px 60px {GLOW};
    border: 1px solid rgba(255,255,255,0.08);
}}
.hero .orb {{
    position: absolute; border-radius: 50%; filter: blur(40px);
    animation: floatOrb 9s ease-in-out infinite;
    pointer-events: none;
}}
.hero .o1 {{ width: 220px; height: 220px; background: rgba(139,227,197,0.35); top: -70px; right: 8%; }}
.hero .o2 {{ width: 160px; height: 160px; background: rgba(255,255,255,0.14); bottom: 20px; right: 30%; animation-delay: -3s; }}
.hero .o3 {{ width: 120px; height: 120px; background: rgba(255,176,32,0.22); top: 20px; right: 2%; animation-delay: -6s; }}

.hero-tag {{
    position: relative;
    color: #A6F2D6; font-size: 11px; letter-spacing: 3px; font-weight: 800;
}}
.hero h1 {{
    position: relative;
    color: #FFFFFF !important;
    font-size: 40px; font-weight: 900; letter-spacing: -1px;
    margin: 8px 0 10px 0; padding: 0;
}}
.hero p {{
    position: relative;
    color: #D8F3E9 !important; font-size: 15px; max-width: 640px; line-height: 1.6;
}}

.road {{
    position: absolute; left: 0; right: 0; bottom: 0; height: 40px;
    background: rgba(5, 18, 14, 0.55);
    border-top: 2px solid rgba(255,255,255,0.12);
}}
.road::before {{
    content: ""; position: absolute; left: 0; right: 0; top: 50%;
    height: 3px; transform: translateY(-50%);
    background: repeating-linear-gradient(90deg, rgba(255,255,255,0.75) 0 36px, transparent 36px 72px);
    background-size: 120px 3px;
    animation: roadMove 1.4s linear infinite;
}}
.car {{
    position: absolute; font-size: 22px; top: 2px;
    animation: carDrive 9s linear infinite;
    filter: drop-shadow(0 3px 4px rgba(0,0,0,0.4));
}}
.car.c2 {{ animation-duration: 13s; animation-delay: -5s; top: 14px; font-size: 20px; }}
.car.c3 {{ animation-duration: 11s; animation-delay: -8s; top: 4px; font-size: 24px; }}

/* ---------- Section headings ---------- */

.section-heading {{
    font-size: 20px; font-weight: 800; color: {TEXT} !important;
    margin: 30px 0 16px 0; position: relative; padding-bottom: 10px;
    animation: fadeUp 0.6s ease both;
}}
.section-heading::after {{
    content: ""; position: absolute; left: 0; bottom: 0; height: 3px;
    border-radius: 3px; background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    animation: underlineGrow 0.9s ease both;
}}

/* ---------- Buttons ---------- */

.stButton > button, .stFormSubmitButton > button {{
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
    color: #05221B !important;
    border: none; border-radius: 12px;
    min-height: 48px; font-weight: 800; letter-spacing: 0.3px;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 8px 24px {GLOW};
}}
.stButton > button::after, .stFormSubmitButton > button::after {{
    content: ""; position: absolute; top: 0; left: 0; width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent);
    animation: shimmer 3s ease-in-out infinite;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 14px 34px {GLOW};
    color: #05221B !important;
}}
.stButton > button:active, .stFormSubmitButton > button:active {{
    transform: translateY(0) scale(0.99);
}}
.stDownloadButton > button {{
    border-radius: 12px; font-weight: 700;
    background: {SURFACE} !important; color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    transition: all 0.25s ease;
}}
.stDownloadButton > button:hover {{
    border-color: {ACCENT} !important; transform: translateY(-2px);
}}

/* ---------- Inputs ---------- */

.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input, .stTextInput input, .stDateInput input {{
    background: {SURFACE} !important;
    color: {TEXT} !important;
    border-radius: 11px;
    border: 1px solid {BORDER} !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.stSelectbox div[data-baseweb="select"] > div:hover,
.stNumberInput input:focus, .stTextInput input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {GLOW};
}}
[data-baseweb="popover"], [data-baseweb="menu"] {{ background: {SURFACE} !important; }}
[data-baseweb="menu"] li, [data-baseweb="menu"] div {{ color: {TEXT} !important; }}
[data-baseweb="select"] span {{ color: {TEXT} !important; }}
[data-testid="stNumberInput"] button {{ background: {SURFACE} !important; color: {TEXT} !important; }}

/* ---------- Form container ---------- */

[data-testid="stForm"] {{
    background: linear-gradient(160deg, {CARD}, {SURFACE});
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 34px {SHADOW};
    animation: fadeUp 0.7s ease both;
}}

/* ---------- Expander ---------- */

[data-testid="stExpander"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    transition: border-color 0.25s ease;
}}
[data-testid="stExpander"]:hover {{ border-color: {ACCENT}; }}
[data-testid="stExpander"] summary {{ color: {TEXT} !important; font-weight: 600; }}

/* ---------- Charts & tables ---------- */

[data-testid="stPlotlyChart"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 8px;
    box-shadow: 0 6px 24px {SHADOW};
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeUp 0.8s ease both;
}}
[data-testid="stPlotlyChart"]:hover {{
    transform: translateY(-4px);
    border-color: {ACCENT};
    box-shadow: 0 16px 40px {GLOW};
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
    animation: fadeUp 0.8s ease both;
}}

/* ---------- Result card ---------- */

.result-card {{
    --glow: {GLOW};
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, {CARD}, {SURFACE});
    border: 1px solid var(--rc, {ACCENT});
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    animation: resultPop 0.7s cubic-bezier(.2,.9,.3,1.2) both, glowPulse 2.8s ease-in-out 0.8s infinite;
}}
.result-label {{
    color: {MUTED}; font-size: 12px; letter-spacing: 3px; font-weight: 700;
}}
.result-value {{
    font-size: 54px; font-weight: 900; margin: 14px 0 6px 0; letter-spacing: -1px;
}}
.result-desc {{ color: {MUTED}; font-size: 14px; }}
.result-model {{
    display: inline-block; margin-top: 16px;
    font-family: 'JetBrains Mono', monospace;
    color: {MUTED}; font-size: 12px;
    border: 1px solid {BORDER}; padding: 5px 12px; border-radius: 999px;
}}

/* ---------- Alerts, misc ---------- */

[data-testid="stAlert"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
    color: {TEXT} !important;
    animation: fadeUp 0.6s ease both;
}}
hr {{ border-color: {BORDER} !important; }}

.footer {{
    text-align: center; color: {MUTED}; font-size: 12px;
    padding: 30px 0; line-height: 1.8;
}}
.footer b {{ color: {TEXT}; font-size: 14px; }}

@media (max-width: 768px) {{
    .hero {{ padding: 26px 22px 70px 22px; }}
    .hero h1 {{ font-size: 28px; }}
    .live {{ display: none; }}
}}
</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

header1, header2 = st.columns([5, 1])

with header1:
    st.markdown("""
    <div class="brand">
        <div class="signal"><span class="r"></span><span class="a"></span><span class="g"></span></div>
        <div>
            <div class="brand-name">TrafficPulse <b>AI</b></div>
            <div class="brand-sub">SMART TRAFFIC INTELLIGENCE</div>
        </div>
        <div class="live"><i></i>LIVE ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)

with header2:
    selected_theme = st.selectbox(
        "Appearance",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
        label_visibility="collapsed",
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()


# ==================================================
# LOAD DATA AND MODEL
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("Banglore_traffic_Dataset.csv")


@st.cache_resource
def get_artifact():
    return load_artifact()


try:
    df = load_data()
    artifact = get_artifact()
except Exception as e:
    st.error(f"Unable to load data or model: {e}")
    st.stop()


# ==================================================
# HELPERS
# ==================================================

PALETTE = ["#22C9A3", "#5EEAD4", "#3B82F6", "#F59E0B", "#EF4444", "#A78BFA", "#F472B6"]


def chart_layout(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter", color=TEXT, size=12),
        margin=dict(l=20, r=20, t=55, b=20),
        title_font=dict(size=16, color=TEXT, family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        coloraxis_showscale=False,
        transition=dict(duration=600, easing="cubic-in-out"),
        hoverlabel=dict(bgcolor=CARD, font=dict(color=TEXT, family="Inter")),
    )
    return fig


def show(fig, height=380):
    st.plotly_chart(
        chart_layout(fig, height),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def hero(tag, title, description):
    st.markdown(f"""
    <div class="hero">
        <div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
        <div class="hero-tag">{tag}</div>
        <h1>{title}</h1>
        <p>{description}</p>
        <div class="road">
            <span class="car">🚗</span>
            <span class="car c2">🚕</span>
            <span class="car c3">🚌</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def heading(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)


# ==================================================
# NAVIGATION
# ==================================================

pages = ["Overview", "Analytics", "Prediction", "Accident Analysis", "Model Performance"]

page = st.radio("Navigation", pages, horizontal=True, label_visibility="collapsed")


# ==================================================
# GLOBAL FILTERS
# ==================================================

if page != "Model Performance":

    with st.expander("⚙️ Dashboard Filters", expanded=False):
        f1, f2 = st.columns(2)

        with f1:
            areas = ["All Areas"] + sorted(df["Area Name"].dropna().astype(str).unique().tolist())
            selected_area = st.selectbox("Select Area", areas)

        with f2:
            weather_options = ["All Weather"] + sorted(
                df["Weather Conditions"].dropna().astype(str).unique().tolist()
            )
            selected_weather = st.selectbox("Select Weather", weather_options)

    filtered_df = df.copy()

    if selected_area != "All Areas":
        filtered_df = filtered_df[filtered_df["Area Name"].astype(str) == selected_area]

    if selected_weather != "All Weather":
        filtered_df = filtered_df[filtered_df["Weather Conditions"].astype(str) == selected_weather]

    if filtered_df.empty:
        st.warning("No records match the selected filters. Try a different combination.")
        st.stop()
else:
    filtered_df = df.copy()


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    hero(
        "SMART CITY INTELLIGENCE",
        "Traffic Overview",
        "A unified view of traffic conditions, congestion patterns and road activity across Bengaluru.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traffic Records", f"{len(filtered_df):,}")
    c2.metric("Avg Traffic Volume", f"{filtered_df['Traffic Volume'].mean():,.0f}")
    c3.metric("Avg Speed (km/h)", f"{filtered_df['Average Speed'].mean():.1f}")
    c4.metric("Incident Reports", f"{filtered_df['Incident Reports'].sum():,.0f}")

    heading("Congestion & Weather")

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered_df, x="Congestion Level", nbins=20,
            title="Congestion Level Distribution",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_traces(marker_line_width=0, opacity=0.92)
        show(fig)

    with right:
        weather = filtered_df.groupby("Weather Conditions", as_index=False)["Traffic Volume"].mean()
        fig = px.bar(
            weather, x="Weather Conditions", y="Traffic Volume",
            title="Average Traffic by Weather",
            color="Traffic Volume", color_continuous_scale="Teal",
        )
        show(fig)

    heading("Recent Traffic Data")

    st.dataframe(filtered_df.head(10), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="traffic_filtered.csv",
        mime="text/csv",
    )


# ==================================================
# PAGE 2: ANALYTICS
# ==================================================

elif page == "Analytics":

    hero(
        "ADVANCED DATA EXPLORATION",
        "Traffic Analytics",
        "Explore area-wise congestion, vehicle volume, road capacity and environmental patterns.",
    )

    left, right = st.columns(2)

    with left:
        area_data = (
            filtered_df.groupby("Area Name", as_index=False)["Traffic Volume"]
            .mean().sort_values("Traffic Volume", ascending=False)
        )
        fig = px.bar(
            area_data, x="Area Name", y="Traffic Volume",
            title="Average Traffic Volume by Area",
            color="Traffic Volume", color_continuous_scale="Teal",
        )
        fig.update_xaxes(tickangle=-35)
        show(fig)

    with right:
        congestion_data = (
            filtered_df.groupby("Area Name", as_index=False)["Congestion Level"]
            .mean().sort_values("Congestion Level", ascending=False)
        )
        fig = px.bar(
            congestion_data, x="Area Name", y="Congestion Level",
            title="Congestion by Area",
            color="Congestion Level", color_continuous_scale="Reds",
        )
        fig.update_xaxes(tickangle=-35)
        show(fig)

    left, right = st.columns(2)

    with left:
        weather_data = filtered_df.groupby("Weather Conditions", as_index=False)["Congestion Level"].mean()
        fig = px.bar(
            weather_data, x="Weather Conditions", y="Congestion Level",
            title="Weather vs Congestion",
            color="Congestion Level", color_continuous_scale="Teal",
        )
        show(fig)

    with right:
        speed_data = filtered_df.groupby("Area Name", as_index=False)["Average Speed"].mean()
        fig = px.bar(
            speed_data, x="Area Name", y="Average Speed",
            title="Average Speed by Area",
            color="Average Speed", color_continuous_scale="Viridis",
        )
        fig.update_xaxes(tickangle=-35)
        show(fig)

    heading("Speed vs Congestion")

    fig = px.scatter(
        filtered_df, x="Average Speed", y="Congestion Level",
        color="Weather Conditions", opacity=0.7,
        title="How speed relates to congestion",
        color_discrete_sequence=PALETTE, hover_data=["Area Name"],
    )
    show(fig, 430)

    heading("Road Capacity Utilization")

    fig = px.histogram(
        filtered_df, x="Road Capacity Utilization", nbins=25,
        title="Distribution of Road Capacity Utilization",
        color_discrete_sequence=[ACCENT],
    )
    show(fig)


# ==================================================
# PAGE 3: PREDICTION
# ==================================================

elif page == "Prediction":

    hero(
        "MACHINE LEARNING ENGINE",
        "Traffic Congestion Prediction",
        "Enter traffic conditions to classify congestion using your trained machine learning model.",
    )

    feature_columns = artifact["feature_columns"]
    numeric_columns = artifact["numeric_columns"]
    categorical_columns = artifact["categorical_columns"]

    SPEED_RANGES = [f"{i}–{i + 10}" for i in range(0, 160, 10)]
    PCT_RANGES = [f"{i}–{i + 10}" for i in range(0, 100, 10)]
    RANGE_FIELDS = {
        "Average Speed": SPEED_RANGES,
        "Public Transport Usage": PCT_RANGES,
        "Road Capacity Utilization": PCT_RANGES,
    }

    input_values = {}

    heading("Traffic Information")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        for i, col in enumerate(feature_columns):

            with (col1 if i % 2 == 0 else col2):

                if col in categorical_columns:
                    encoder = artifact["label_encoders"][col]
                    input_values[col] = st.selectbox(col, list(encoder.classes_))

                elif col in numeric_columns:

                    if col in RANGE_FIELDS:
                        selected_range = st.selectbox(col, RANGE_FIELDS[col])
                        lower, upper = map(int, selected_range.replace("–", "-").split("-"))
                        input_values[col] = (lower + upper) / 2
                    else:
                        input_values[col] = st.number_input(
                            col,
                            value=float(artifact["numeric_fill_values"][col]),
                            format="%.2f",
                        )

        submitted = st.form_submit_button("🚀 Predict Traffic Congestion", use_container_width=True)

    if submitted:

        try:
            with st.spinner("Analyzing traffic patterns..."):
                result = predict_congestion(input_values)

            colors = {"Low": "#16C47F", "Medium": "#F5A524", "High": "#F0445A"}
            levels = {"Low": 1, "Medium": 2, "High": 3}
            descriptions = {
                "Low": "Traffic flow is classified as low congestion. Roads are moving freely.",
                "Medium": "Traffic flow is classified as moderate congestion. Expect some slowdowns.",
                "High": "Traffic flow is classified as high congestion. Consider alternate routes.",
            }

            color = colors.get(result, ACCENT)

            res_col, gauge_col = st.columns([3, 2])

            with res_col:
                st.markdown(f"""
                <div class="result-card" style="--rc:{color};--glow:{color}33">
                    <div class="result-label">PREDICTION RESULT</div>
                    <div class="result-value" style="color:{color}">{str(result).upper()}</div>
                    <div class="result-desc">{descriptions.get(result, "Prediction generated.")}</div>
                    <div class="result-model">Model · {artifact['model_name']}</div>
                </div>
                """, unsafe_allow_html=True)

            with gauge_col:
                gauge = go.Figure(go.Indicator(
                    mode="gauge",
                    value=levels.get(result, 0),
                    gauge=dict(
                        axis=dict(range=[0, 3], tickvals=[0.5, 1.5, 2.5],
                                  ticktext=["Low", "Medium", "High"], tickcolor=MUTED),
                        bar=dict(color=color, thickness=0.35),
                        bgcolor=SURFACE,
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 1], color="rgba(22,196,127,0.25)"),
                            dict(range=[1, 2], color="rgba(245,165,36,0.25)"),
                            dict(range=[2, 3], color="rgba(240,68,90,0.25)"),
                        ],
                    ),
                ))
                gauge.update_layout(
                    height=270,
                    paper_bgcolor=SURFACE,
                    font=dict(family="Inter", color=TEXT),
                    margin=dict(l=30, r=30, t=30, b=10),
                    transition=dict(duration=900, easing="cubic-in-out"),
                )
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ==================================================
# PAGE 4: ACCIDENT ANALYSIS
# ==================================================

elif page == "Accident Analysis":

    hero(
        "ROAD SAFETY INTELLIGENCE",
        "Accident & Incident Analysis",
        "Understand reported incident patterns and their relationship with weather and congestion.",
    )

    st.info("This section analyzes reported incidents. It does not calculate validated accident probabilities.")

    left, right = st.columns(2)

    with left:
        incident_area = (
            filtered_df.groupby("Area Name", as_index=False)["Incident Reports"]
            .sum().sort_values("Incident Reports", ascending=False)
        )
        fig = px.bar(
            incident_area, x="Area Name", y="Incident Reports",
            title="Incident Reports by Area",
            color="Incident Reports", color_continuous_scale="Reds",
        )
        fig.update_xaxes(tickangle=-35)
        show(fig)

    with right:
        incident_weather = filtered_df.groupby("Weather Conditions", as_index=False)["Incident Reports"].sum()
        fig = px.pie(
            incident_weather, names="Weather Conditions", values="Incident Reports",
            hole=0.58, title="Incident Distribution by Weather",
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(
            textposition="inside",
            marker=dict(line=dict(color=SURFACE, width=3)),
            pull=[0.02] * len(incident_weather),
        )
        show(fig)

    heading("Congestion vs Incidents")

    fig = px.scatter(
        filtered_df, x="Congestion Level", y="Incident Reports",
        color="Weather Conditions", opacity=0.75,
        title="Congestion vs Incident Reports",
        color_discrete_sequence=PALETTE, hover_data=["Area Name"],
    )
    show(fig, 440)


# ==================================================
# PAGE 5: MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":

    hero(
        "MODEL INTELLIGENCE",
        "Machine Learning Performance",
        "Review your selected machine learning classifier and its saved evaluation results.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model", artifact["model_name"])
    c2.metric("Test Accuracy", f"{artifact['accuracy'] * 100:.2f}%")
    c3.metric("Input Features", len(artifact["feature_columns"]))

    heading("Accuracy at a Glance")

    acc = artifact["accuracy"] * 100
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=acc,
        number=dict(suffix="%", font=dict(color=ACCENT, size=44)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED),
            bar=dict(color=ACCENT, thickness=0.3),
            bgcolor=SURFACE,
            borderwidth=0,
            steps=[
                dict(range=[0, 60], color="rgba(240,68,90,0.18)"),
                dict(range=[60, 85], color="rgba(245,165,36,0.18)"),
                dict(range=[85, 100], color="rgba(22,196,127,0.18)"),
            ],
        ),
    ))
    gauge.update_layout(
        height=320,
        paper_bgcolor=SURFACE,
        font=dict(family="Inter", color=TEXT),
        margin=dict(l=30, r=30, t=30, b=10),
        transition=dict(duration=900, easing="cubic-in-out"),
    )
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

    heading("Model Details")

    details = pd.DataFrame({
        "Information": ["Selected Model", "Target Variable", "Target Categories", "Number of Input Features"],
        "Value": [
            artifact["model_name"],
            "Congestion_Category",
            "Low, Medium, High",
            len(artifact["feature_columns"]),
        ],
    })
    details["Value"] = details["Value"].astype(str)

    st.dataframe(details, use_container_width=True, hide_index=True)

    heading("Input Features")

    feats = pd.DataFrame({
        "Feature": artifact["feature_columns"],
        "Type": [
            "Categorical" if c in artifact["categorical_columns"] else "Numeric"
            for c in artifact["feature_columns"]
        ],
    })
    st.dataframe(feats, use_container_width=True, hide_index=True)

    st.info(
        "This page displays the saved test accuracy. "
        "The four-model comparison and confusion matrix from your notebook can be added next."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    🚦 <b>TrafficPulse AI</b><br>
    Intelligent Traffic Monitoring &amp; Congestion Analysis<br>
    Machine Learning Project ·
</div>
""", unsafe_allow_html=True)
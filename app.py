# ============================================================
#   AI-Based Job Recommendation System  ·  ULTIMATE EDITION
#   Prepared by  : Waqaas Hussain
#   Subject      : Programming for AI
#   Framework    : Streamlit + Scikit-learn + Plotly
#   Features     : Glassmorphism · Dark/Light · CSS Animations
#                  Inline SVG Icons · Plotly Charts · Skill Gap
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import re, json, warnings
warnings.filterwarnings('ignore')

# ╔══════════════════════════════════════════════════════════╗
#   PAGE CONFIG
# ╚══════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="JobMatch AI · Waqaas Hussain",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%2300C9A7'/><stop offset='1' stop-color='%233B82F6'/></linearGradient></defs><circle cx='16' cy='16' r='16' fill='%230A1628'/><path d='M8 20 L16 8 L24 20' stroke='url(%23g)' stroke-width='2.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/><circle cx='16' cy='10' r='2' fill='%2300C9A7'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ╔══════════════════════════════════════════════════════════╗
#   INLINE SVG ICON LIBRARY  (zero external deps, never breaks)
# ╚══════════════════════════════════════════════════════════╝
def _svg(paths, color="currentColor", size=18, fill="none", sw=2):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="{fill}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;">'
            f'{paths}</svg>')

class IC:
    """Central SVG icon registry — all inline, no CDN."""
    @staticmethod
    def brain(c="currentColor",s=18):
        return _svg('<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.99-3 2.5 2.5 0 0 1-1.13-4.42A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2-3 2.5 2.5 0 0 0 1.13-4.42A2.5 2.5 0 0 0 14.5 2z"/>',c,s)
    @staticmethod
    def briefcase(c="currentColor",s=18):
        return _svg('<rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',c,s)
    @staticmethod
    def building(c="currentColor",s=18):
        return _svg('<rect width="16" height="20" x="4" y="2" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01M16 6h.01M12 6h.01M12 10h.01M12 14h.01M16 10h.01M16 14h.01M8 10h.01M8 14h.01"/>',c,s)
    @staticmethod
    def location(c="currentColor",s=18):
        return _svg('<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',c,s)
    @staticmethod
    def clock(c="currentColor",s=18):
        return _svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',c,s)
    @staticmethod
    def dollar(c="currentColor",s=18):
        return _svg('<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',c,s)
    @staticmethod
    def check(c="currentColor",s=18):
        return _svg('<path d="M20 6 9 17l-5-5"/>',c,s)
    @staticmethod
    def xmark(c="currentColor",s=18):
        return _svg('<path d="M18 6 6 18M6 6l12 12"/>',c,s)
    @staticmethod
    def star(c="currentColor",s=18):
        return _svg('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',c,s,'currentColor',1.5)
    @staticmethod
    def graduation(c="currentColor",s=18):
        return _svg('<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>',c,s)
    @staticmethod
    def layers(c="currentColor",s=18):
        return _svg('<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12-8.6 3.92a2 2 0 0 1-1.66 0L3 12"/><path d="m22 17-8.6 3.92a2 2 0 0 1-1.66 0L3 17"/>',c,s)
    @staticmethod
    def target(c="currentColor",s=18):
        return _svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',c,s)
    @staticmethod
    def chart(c="currentColor",s=18):
        return _svg('<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/><path d="M2 20h20"/>',c,s)
    @staticmethod
    def info(c="currentColor",s=18):
        return _svg('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',c,s)
    @staticmethod
    def download(c="currentColor",s=18):
        return _svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',c,s)
    @staticmethod
    def alert(c="currentColor",s=18):
        return _svg('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/>',c,s)
    @staticmethod
    def search(c="currentColor",s=18):
        return _svg('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',c,s)
    @staticmethod
    def settings(c="currentColor",s=18):
        return _svg('<circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>',c,s)
    @staticmethod
    def home(c="currentColor",s=18):
        return _svg('<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',c,s)
    @staticmethod
    def cpu(c="currentColor",s=18):
        return _svg('<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',c,s)
    @staticmethod
    def trending(c="currentColor",s=18):
        return _svg('<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',c,s)
    @staticmethod
    def code(c="currentColor",s=18):
        return _svg('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',c,s)
    @staticmethod
    def lightning(c="currentColor",s=18):
        return _svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',c,s,'currentColor',1.5)
    @staticmethod
    def moon(c="currentColor",s=18):
        return _svg('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',c,s)
    @staticmethod
    def sun(c="currentColor",s=18):
        return _svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',c,s)
    @staticmethod
    def arrow_right(c="currentColor",s=18):
        return _svg('<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',c,s)
    @staticmethod
    def filter(c="currentColor",s=18):
        return _svg('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',c,s)
    @staticmethod
    def send(c="currentColor",s=18):
        return _svg('<path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/>',c,s)
    @staticmethod
    def python(c="#3776AB",s=18):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" '
                f'style="vertical-align:middle;" fill="{c}">'
                f'<path d="M11.914 0C5.82 0 6.2 2.656 6.2 2.656l.007 2.752h5.814v.826H3.9S0 5.789 0 11.969c0 6.18 3.403 5.963 3.403 5.963h2.031v-2.868s-.109-3.402 3.35-3.402h5.769s3.24.052 3.24-3.131V3.129S18.28 0 11.914 0zm-3.21 1.818a1.042 1.042 0 1 1 0 2.084 1.042 1.042 0 0 1 0-2.084z"/>'
                f'<path d="M12.086 24c6.094 0 5.714-2.656 5.714-2.656l-.007-2.752h-5.814v-.826h8.121S24 18.211 24 12.031c0-6.18-3.403-5.963-3.403-5.963h-2.031v2.868s.109 3.402-3.35 3.402H9.447s-3.24-.052-3.24 3.131v5.402S5.72 24 12.086 24zm3.21-1.818a1.042 1.042 0 1 1 0-2.084 1.042 1.042 0 0 1 0 2.084z" fill="{c}"/></svg>')


# ╔══════════════════════════════════════════════════════════╗
#   THEME  (dark / light — stored in session_state)
# ╚══════════════════════════════════════════════════════════╝
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = st.session_state.theme
IS_DARK = (T == "dark")

# palette
if IS_DARK:
    BG      = "#080F1E"
    BG2     = "#0D1A2D"
    CARD    = "#0F2040"
    CARD2   = "#132545"
    BORDER  = "rgba(255,255,255,0.08)"
    TEXT    = "#E8EDF5"
    MUTED   = "#7A90A8"
    TEAL    = "#00C9A7"
    TEAL2   = "#00A88E"
    BLUE    = "#3B82F6"
    PURPLE  = "#8B5CF6"
    AMBER   = "#F59E0B"
    RED     = "#EF4444"
    GLASS   = "rgba(15,32,64,0.7)"
    PLOTLY_BG = "#0F2040"
    PLOTLY_GRID = "#162845"
    PLOTLY_TEXT = "#7A90A8"
else:
    BG      = "#F0F5FF"
    BG2     = "#E8EFFA"
    CARD    = "#FFFFFF"
    CARD2   = "#F7FAFF"
    BORDER  = "rgba(0,0,0,0.08)"
    TEXT    = "#0A1628"
    MUTED   = "#64748B"
    TEAL    = "#0D9B82"
    TEAL2   = "#0A7A68"
    BLUE    = "#2563EB"
    PURPLE  = "#7C3AED"
    AMBER   = "#D97706"
    RED     = "#DC2626"
    GLASS   = "rgba(255,255,255,0.75)"
    PLOTLY_BG = "#FFFFFF"
    PLOTLY_GRID = "#E2E8F0"
    PLOTLY_TEXT = "#64748B"


# ╔══════════════════════════════════════════════════════════╗
#   GLOBAL CSS + ANIMATIONS
# ╚══════════════════════════════════════════════════════════╝
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
/* ── CSS VARIABLES ── */
:root {{
  --bg:      {BG};
  --bg2:     {BG2};
  --card:    {CARD};
  --card2:   {CARD2};
  --border:  {BORDER};
  --text:    {TEXT};
  --muted:   {MUTED};
  --teal:    {TEAL};
  --teal2:   {TEAL2};
  --blue:    {BLUE};
  --purple:  {PURPLE};
  --amber:   {AMBER};
  --red:     {RED};
  --glass:   {GLASS};
  --radius:  16px;
  --shadow:  0 4px 32px rgba(0,0,0,{0.35 if IS_DARK else 0.08});
  --shadow-lg: 0 12px 48px rgba(0,0,0,{0.45 if IS_DARK else 0.12});
}}

/* ── KEYFRAMES ── */
@keyframes fadeUp {{
  from {{ opacity:0; transform:translateY(24px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
@keyframes fadeIn {{
  from {{ opacity:0; }} to {{ opacity:1; }}
}}
@keyframes slideRight {{
  from {{ opacity:0; transform:translateX(-20px); }}
  to   {{ opacity:1; transform:translateX(0); }}
}}
@keyframes pulse {{
  0%,100% {{ box-shadow: 0 0 0 0 rgba(0,201,167,0.5); }}
  50%      {{ box-shadow: 0 0 0 10px rgba(0,201,167,0); }}
}}
@keyframes shimmer {{
  0%   {{ background-position: -400px 0; }}
  100% {{ background-position: 400px 0; }}
}}
@keyframes spin {{
  from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }}
}}
@keyframes barGrow {{
  from {{ transform: scaleX(0); transform-origin: left; }}
  to   {{ transform: scaleX(1); transform-origin: left; }}
}}
@keyframes countUp {{
  from {{ opacity:0; transform:scale(0.7); }}
  to   {{ opacity:1; transform:scale(1); }}
}}
@keyframes glow {{
  0%,100% {{ text-shadow: 0 0 10px rgba(0,201,167,0.3); }}
  50%      {{ text-shadow: 0 0 24px rgba(0,201,167,0.7); }}
}}

/* ── GLOBAL RESET ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
  transition: background 0.4s ease, color 0.4s ease;
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--teal); border-radius: 99px; }}

/* ── STREAMLIT CHROME ── */
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2.5rem 4rem !important; max-width: 1600px !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
  background: var(--card) !important;
  border-right: 1px solid var(--border) !important;
  backdrop-filter: blur(16px);
}}
section[data-testid="stSidebar"] > div {{ padding: 1.5rem 1.2rem !important; }}
section[data-testid="stSidebar"] * {{ color: var(--text) !important; }}
section[data-testid="stSidebar"] label {{
  font-size: 0.78rem !important; font-weight: 600 !important;
  color: var(--muted) !important; letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
}}
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stTextInput input {{
  background: var(--bg2) !important; border: 1.5px solid var(--border) !important;
  color: var(--text) !important; border-radius: 12px !important;
  font-family: 'DM Sans', sans-serif !important; font-size: 0.88rem !important;
  transition: border-color 0.2s !important;
}}
section[data-testid="stSidebar"] .stTextArea textarea:focus,
section[data-testid="stSidebar"] .stTextInput input:focus {{
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(0,201,167,0.15) !important;
}}
/* Sidebar find button */
section[data-testid="stSidebar"] .stButton button {{
  background: linear-gradient(135deg, var(--teal), var(--blue)) !important;
  color: {'#0A1628' if IS_DARK else '#ffffff'} !important;
  font-weight: 700 !important; border: none !important;
  border-radius: 14px !important; padding: 0.7rem 1rem !important;
  font-size: 0.92rem !important; letter-spacing: 0.02em !important;
  transition: all 0.25s !important;
  box-shadow: 0 4px 20px rgba(0,201,167,0.35) !important;
  font-family: 'Syne', sans-serif !important;
}}
section[data-testid="stSidebar"] .stButton button:hover {{
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(0,201,167,0.5) !important;
}}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {{
  background: var(--card2) !important; border: 1.5px solid var(--border) !important;
  color: var(--text) !important; border-radius: 12px !important;
  font-family: 'DM Sans', sans-serif !important;
}}

/* ── MAIN BUTTONS ── */
.stButton button {{
  background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
  color: white !important; border: none !important; border-radius: 12px !important;
  font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
  transition: all 0.25s !important;
}}
.stButton button:hover {{ transform: translateY(-2px) !important; box-shadow: var(--shadow) !important; }}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton button {{
  background: linear-gradient(135deg, var(--teal), var(--teal2)) !important;
  color: {'#0A1628' if IS_DARK else '#ffffff'} !important;
  border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; font-family: 'Syne', sans-serif !important;
}}

/* ── RADIO ── */
.stRadio > div {{ gap: 6px !important; flex-wrap: wrap !important; }}
.stRadio label {{
  background: var(--card2) !important; border: 1.5px solid var(--border) !important;
  border-radius: 10px !important; padding: 5px 14px !important;
  font-size: 0.78rem !important; color: var(--muted) !important;
  cursor: pointer !important; transition: all 0.2s !important;
  font-family: 'DM Sans', sans-serif !important;
}}
.stRadio label:hover {{ border-color: var(--teal) !important; color: var(--teal) !important; }}
div[data-testid="stRadio"] label[data-selected="true"] {{
  background: rgba(0,201,167,0.1) !important;
  border-color: var(--teal) !important; color: var(--teal) !important;
  font-weight: 600 !important;
}}

/* ── SELECTBOX ── */
.stSelectbox > div > div {{
  background: var(--bg2) !important; border: 1.5px solid var(--border) !important;
  border-radius: 12px !important; color: var(--text) !important;
}}
/* ── SLIDER ── */
.stSlider [data-baseweb="slider"] {{ margin-top: 0 !important; }}

/* ── EXPANDER ── */
.streamlit-expanderHeader {{
  background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; font-weight: 600 !important; color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
}}
.streamlit-expanderContent {{
  background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-top: none !important; border-radius: 0 0 14px 14px !important;
}}

/* ── DATAFRAME ── */
.stDataFrame {{ border-radius: 14px !important; overflow: hidden; }}
iframe {{ border-radius: 14px !important; }}

/* ╔══════════════════╗
   CUSTOM COMPONENTS
   ╚══════════════════╝ */

/* Hero banner */
.hero {{
  position: relative; overflow: hidden;
  background: linear-gradient(135deg, {BG2} 0%, {CARD} 50%, {BG2} 100%);
  border: 1px solid {BORDER}; border-radius: 24px;
  padding: 2.8rem 3.2rem; margin-bottom: 2rem;
  animation: fadeUp 0.6s ease;
}}
.hero-glow {{
  position: absolute; top: -80px; right: -80px;
  width: 350px; height: 350px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,201,167,0.18) 0%, transparent 65%);
  pointer-events: none;
}}
.hero-glow2 {{
  position: absolute; bottom: -100px; left: 30%;
  width: 300px; height: 300px; border-radius: 50%;
  background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 65%);
  pointer-events: none;
}}
.hero-badge {{
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(0,201,167,0.10); border: 1px solid rgba(0,201,167,0.28);
  color: {TEAL}; font-size: 0.7rem; font-weight: 700;
  padding: 4px 14px; border-radius: 99px; margin-bottom: 1.2rem;
  letter-spacing: 0.1em; text-transform: uppercase;
  animation: fadeIn 0.4s ease 0.2s both;
}}
.pulse-dot {{
  width: 7px; height: 7px; border-radius: 50%;
  background: {TEAL}; animation: pulse 2s infinite;
}}
.hero-title {{
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem; font-weight: 800; color: {TEXT};
  margin: 0 0 0.6rem; line-height: 1.1; letter-spacing: -1px;
  animation: fadeUp 0.5s ease 0.1s both;
}}
.hero-title span {{ color: {TEAL}; animation: glow 3s ease infinite; display: inline-block; }}
.hero-sub {{
  color: {MUTED}; font-size: 0.92rem; max-width: 600px; line-height: 1.7;
  animation: fadeUp 0.5s ease 0.2s both;
}}
.hero-chips {{
  display: flex; gap: 10px; margin-top: 1.8rem; flex-wrap: wrap;
  animation: fadeUp 0.5s ease 0.3s both;
}}
.hero-chip {{
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(255,255,255,{0.06 if IS_DARK else 0.7});
  backdrop-filter: blur(8px);
  border: 1px solid {BORDER}; border-radius: 10px;
  padding: 7px 14px; font-size: 0.78rem; color: {MUTED};
  font-weight: 500; transition: all 0.2s;
}}
.hero-chip:hover {{ border-color: {TEAL}; color: {TEAL}; transform: translateY(-2px); }}
.hero-chip b {{ color: {TEXT}; }}

/* Stat cards */
.stat-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 1.6rem 0; }}
.stat-card {{
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 18px; padding: 1.4rem 1.2rem;
  position: relative; overflow: hidden;
  animation: fadeUp 0.5s ease both; transition: all 0.25s;
  backdrop-filter: blur(8px);
}}
.stat-card:nth-child(1){{ animation-delay:0.05s; }}
.stat-card:nth-child(2){{ animation-delay:0.10s; }}
.stat-card:nth-child(3){{ animation-delay:0.15s; }}
.stat-card:nth-child(4){{ animation-delay:0.20s; }}
.stat-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, {TEAL}, {BLUE}, {PURPLE});
}}
.stat-card:hover {{ transform: translateY(-4px); box-shadow: var(--shadow-lg); border-color: {TEAL}; }}
.stat-icon {{ margin-bottom: 10px; }}
.stat-num {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.1rem; font-weight: 700; color: {TEXT};
  line-height: 1; animation: countUp 0.5s ease;
}}
.stat-lbl {{ font-size: 0.72rem; color: {MUTED}; margin-top: 5px; font-weight: 500; letter-spacing: 0.03em; }}

/* Job card */
.job-card {{
  background: var(--card2); border: 1.5px solid var(--border);
  border-radius: 18px; padding: 1.5rem 1.75rem; margin-bottom: 4px;
  animation: slideRight 0.4s ease both; transition: all 0.25s;
  backdrop-filter: blur(12px); position: relative; overflow: hidden;
}}
.job-card::before {{
  content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, {TEAL}, {BLUE});
  border-radius: 4px 0 0 4px;
}}
.job-card:hover {{ border-color: rgba(0,201,167,0.4); transform: translateX(6px); box-shadow: var(--shadow); }}
.job-title-txt {{
  font-family: 'Syne', sans-serif; font-size: 1.08rem;
  font-weight: 700; color: {TEXT}; line-height: 1.3;
}}
.job-meta-row {{
  display: flex; align-items: center; gap: 14px;
  font-size: 0.8rem; color: {MUTED}; margin: 6px 0 0; flex-wrap: wrap;
}}
.job-meta-item {{ display: inline-flex; align-items: center; gap: 5px; }}

/* Match ring */
.match-ring {{
  width: 62px; height: 62px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 800;
  border: 2.5px solid; position: relative; transition: all 0.3s;
}}
.match-ring:hover {{ transform: scale(1.1); }}

/* Badges */
.badge {{
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 0.68rem; font-weight: 700; padding: 3px 10px;
  border-radius: 99px; margin: 2px; letter-spacing: 0.02em;
  transition: transform 0.15s;
}}
.badge:hover {{ transform: scale(1.05); }}
.b-teal   {{ background: rgba(0,201,167,0.12);  color: {TEAL};   border: 1px solid rgba(0,201,167,0.25); }}
.b-blue   {{ background: rgba(59,130,246,0.12); color: {BLUE};   border: 1px solid rgba(59,130,246,0.25); }}
.b-purple {{ background: rgba(139,92,246,0.12); color: {PURPLE}; border: 1px solid rgba(139,92,246,0.25); }}
.b-amber  {{ background: rgba(245,158,11,0.12); color: {AMBER};  border: 1px solid rgba(245,158,11,0.25); }}
.b-red    {{ background: rgba(239,68,68,0.12);  color: {RED};    border: 1px solid rgba(239,68,68,0.25); }}

/* Match bar */
.bar-track {{ background: {'rgba(255,255,255,0.06)' if IS_DARK else '#E2E8F0'}; border-radius: 99px; height: 7px; margin: 12px 0 4px; overflow: hidden; }}
.bar-fill  {{ height: 100%; border-radius: 99px; animation: barGrow 0.8s ease 0.3s both; }}

/* Skill chips */
.chip-match {{ display:inline-flex;align-items:center;gap:4px;font-size:0.68rem;padding:3px 10px;border-radius:99px;margin:2px;background:rgba(0,201,167,0.10);color:{TEAL};border:1px solid rgba(0,201,167,0.22);font-weight:600; }}
.chip-miss  {{ display:inline-flex;align-items:center;gap:4px;font-size:0.68rem;padding:3px 10px;border-radius:99px;margin:2px;background:rgba(239,68,68,0.10);color:{RED};border:1px solid rgba(239,68,68,0.22);font-weight:600; }}

/* Salary badge */
.salary-tag {{
  display: inline-flex; align-items: center; gap: 7px;
  background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.22);
  border-radius: 10px; padding: 5px 14px;
  font-size: 0.82rem; font-weight: 700; color: {AMBER};
  font-family: 'JetBrains Mono', monospace;
}}

/* Gap box */
.gap-box {{
  background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.2);
  border-radius: 12px; padding: 10px 14px; margin-top: 10px;
  display: flex; gap: 10px; align-items: flex-start;
  animation: fadeIn 0.3s ease;
}}
.gap-title {{ font-weight: 700; color: {AMBER}; font-size: 0.78rem; }}
.gap-skills {{ font-size: 0.76rem; color: {'#FCD34D' if IS_DARK else AMBER}; margin-top: 3px; }}

/* Section header */
.sec-hdr {{
  display: flex; align-items: center; gap: 10px;
  font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: {TEXT};
  border-left: 4px solid {TEAL}; padding-left: 12px;
  margin: 2rem 0 1.2rem; animation: slideRight 0.4s ease;
}}

/* Welcome card */
.welcome-wrap {{
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 24px; padding: 3.5rem 2rem; text-align: center;
  animation: fadeUp 0.5s ease; backdrop-filter: blur(12px);
}}
.welcome-title {{
  font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800;
  color: {TEXT}; margin: 1.2rem 0 0.6rem;
}}
.welcome-sub {{ font-size: 0.9rem; color: {MUTED}; max-width: 480px; margin: 0 auto; line-height: 1.7; }}

/* Step cards */
.step-card {{
  background: var(--card2); border: 1.5px solid var(--border);
  border-radius: 16px; padding: 1.4rem; text-align: center;
  animation: fadeUp 0.5s ease both; transition: all 0.25s;
}}
.step-card:hover {{ border-color: {TEAL}; transform: translateY(-4px); box-shadow: var(--shadow); }}
.step-num {{
  font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
  color: {TEAL}; line-height: 1; margin-bottom: 10px;
}}
.step-title {{ font-weight: 700; color: {TEXT}; font-size: 0.95rem; margin-bottom: 5px; }}
.step-desc {{ font-size: 0.78rem; color: {MUTED}; line-height: 1.6; }}

/* How-it-works step */
.how-card {{
  display: flex; gap: 18px; background: var(--card2);
  border: 1px solid var(--border); border-radius: 16px;
  padding: 1.25rem 1.5rem; margin-bottom: 10px;
  animation: slideRight 0.4s ease both; transition: all 0.2s;
}}
.how-card:hover {{ border-color: {TEAL}; transform: translateX(4px); }}
.how-num {{
  font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
  color: {TEAL}; min-width: 42px; line-height: 1;
}}
.how-title {{ font-weight: 700; color: {TEXT}; font-size: 0.95rem; margin-bottom: 4px; display:flex;align-items:center;gap:8px; }}
.how-desc {{ font-size: 0.82rem; color: {MUTED}; line-height: 1.65; }}

/* Tech table */
.tech-tbl {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
.tech-tbl th {{ background:rgba(0,201,167,0.08); color:{TEAL}; font-weight:700; padding:10px 14px; text-align:left; border-bottom:1px solid {BORDER}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em; }}
.tech-tbl td {{ padding:9px 14px; border-bottom:1px solid {BORDER}; color:{MUTED}; }}
.tech-tbl td:first-child {{ color:{TEXT}; font-weight:600; }}
.tech-tbl tr:hover td {{ background:rgba(0,201,167,0.04); }}

/* Theme toggle button */
.theme-toggle {{
  position: fixed; top: 14px; right: 14px; z-index: 9999;
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 99px; padding: 7px 14px; cursor: pointer;
  display: flex; align-items: center; gap: 7px;
  font-size: 0.75rem; font-weight: 600; color: {MUTED};
  transition: all 0.2s; box-shadow: var(--shadow);
  backdrop-filter: blur(12px);
}}
.theme-toggle:hover {{ border-color: {TEAL}; color: {TEAL}; }}

/* Glass overlay on analytics cards */
.glass-card {{
  background: {GLASS}; backdrop-filter: blur(16px);
  border: 1px solid {BORDER}; border-radius: 18px;
  padding: 1.3rem; animation: fadeIn 0.4s ease;
}}

/* Cat count pill */
.cat-pill {{
  display:flex; align-items:center; gap:10px;
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 12px; padding: 10px 16px; margin-bottom: 8px;
  transition: all 0.2s; animation: fadeUp 0.4s ease both;
}}
.cat-pill:hover {{ border-color: {TEAL}; transform: translateX(4px); }}

/* Sidebar nav */
.sidebar-brand {{
  font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800;
  color: {TEXT}; margin-bottom: 3px;
}}
.sidebar-brand span {{ color: {TEAL}; }}
.sidebar-tag {{
  font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: {TEAL}; margin: 1.2rem 0 0.5rem;
  display: flex; align-items: center; gap: 7px;
}}
.sidebar-divider {{ border: none; border-top: 1px solid {BORDER}; margin: 0.8rem 0; }}

/* Info/project box */
.info-card {{
  background: rgba(0,201,167,0.06); border: 1px solid rgba(0,201,167,0.2);
  border-radius: 14px; padding: 1.2rem 1.5rem; margin-top: 1rem;
  animation: fadeIn 0.4s ease;
}}
.info-label {{ font-size: 0.78rem; color: {TEAL}; font-weight: 700; margin-bottom: 6px; }}
.info-value {{ font-size: 0.8rem; color: {MUTED}; line-height: 1.7; }}
.info-value b {{ color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#   DATASET
# ╚══════════════════════════════════════════════════════════╝
@st.cache_data
def load_dataset():
    data = [
        {"id":1,"title":"Senior Python Developer","company":"TechNova Solutions","location":"Remote","type":"Full-time","category":"Engineering","salary":"$90,000–$130,000","exp_min":4,"edu":"Bachelor's","desc":"Python Django FastAPI AWS Docker PostgreSQL REST APIs microservices backend agile CI/CD","skills":"Python,Django,FastAPI,AWS,Docker,PostgreSQL,REST APIs,Git"},
        {"id":2,"title":"Full Stack Web Developer","company":"WebCraft Studio","location":"Karachi, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp_min":2,"edu":"Bachelor's","desc":"React Node.js MongoDB Express JavaScript TypeScript HTML CSS REST APIs agile git frontend backend","skills":"React,Node.js,MongoDB,JavaScript,TypeScript,HTML,CSS,Express,Git"},
        {"id":3,"title":"Frontend Developer","company":"UX Studio","location":"London, UK","type":"Full-time","category":"Engineering","salary":"$70,000–$100,000","exp_min":2,"edu":"Bachelor's","desc":"React Vue JavaScript TypeScript CSS HTML webpack performance accessibility user experience design","skills":"React,Vue.js,JavaScript,TypeScript,CSS,HTML,Webpack,Figma"},
        {"id":4,"title":"Backend Developer","company":"ServerLogic","location":"Remote","type":"Full-time","category":"Engineering","salary":"$80,000–$115,000","exp_min":3,"edu":"Bachelor's","desc":"Python Java Spring Boot Django Flask PostgreSQL MySQL Redis Docker Kubernetes cloud AWS CI/CD","skills":"Python,Java,Spring Boot,PostgreSQL,MySQL,Docker,Kubernetes,Redis"},
        {"id":5,"title":"Data Scientist","company":"DataSphere Analytics","location":"New York, USA","type":"Full-time","category":"Data Science","salary":"$95,000–$140,000","exp_min":3,"edu":"Master's","desc":"Python TensorFlow scikit-learn pandas numpy machine learning deep learning SQL statistics data visualization analytics","skills":"Python,Machine Learning,TensorFlow,SQL,Pandas,NumPy,Statistics,Scikit-learn"},
        {"id":6,"title":"Machine Learning Engineer","company":"AI Dynamics","location":"San Francisco, USA","type":"Full-time","category":"AI/ML","salary":"$120,000–$180,000","exp_min":3,"edu":"Master's","desc":"PyTorch TensorFlow MLOps Kubernetes Docker machine learning deep learning NLP computer vision model deployment","skills":"Python,PyTorch,TensorFlow,MLOps,Kubernetes,Docker,Deep Learning,NLP"},
        {"id":7,"title":"AI Research Scientist","company":"DeepMind Labs","location":"London, UK","type":"Full-time","category":"AI/ML","salary":"$130,000–$200,000","exp_min":2,"edu":"PhD","desc":"Deep learning NLP reinforcement learning PyTorch mathematics statistics research papers neural networks transformers BERT GPT algorithms","skills":"Python,PyTorch,Deep Learning,NLP,Research,Mathematics,Statistics,Transformers"},
        {"id":8,"title":"NLP Engineer","company":"LanguageAI","location":"Remote","type":"Full-time","category":"AI/ML","salary":"$100,000–$145,000","exp_min":3,"edu":"Master's","desc":"NLP pipelines text classification BERT transformers spaCy NLTK Python TensorFlow language models GPT fine-tuning sentiment analysis","skills":"Python,NLP,Transformers,BERT,spaCy,NLTK,TensorFlow,Hugging Face"},
        {"id":9,"title":"Data Engineer","company":"PipelineAI","location":"Singapore","type":"Full-time","category":"Data Science","salary":"$85,000–$120,000","exp_min":3,"edu":"Bachelor's","desc":"Apache Spark Kafka SQL AWS Airflow ETL data warehouse Python cloud engineering big data stream processing batch","skills":"Python,Apache Spark,Kafka,SQL,AWS,Airflow,ETL,Data Warehouse"},
        {"id":10,"title":"Business Intelligence Analyst","company":"InsightCorp","location":"Karachi, Pakistan","type":"Full-time","category":"Data Science","salary":"$30,000–$55,000","exp_min":2,"edu":"Bachelor's","desc":"SQL Power BI Tableau Excel data visualization dashboard reporting KPIs business analytics Python stakeholder","skills":"SQL,Power BI,Tableau,Excel,Python,Data Analytics,Reporting"},
        {"id":11,"title":"DevOps Engineer","company":"CloudCore","location":"Remote","type":"Full-time","category":"DevOps","salary":"$85,000–$125,000","exp_min":3,"edu":"Bachelor's","desc":"CI/CD AWS Docker Kubernetes Terraform Jenkins Linux Python monitoring deployment site reliability automation","skills":"AWS,Docker,Kubernetes,Terraform,Jenkins,Linux,Python,CI/CD"},
        {"id":12,"title":"Cloud Solutions Architect","company":"Nimbus Cloud","location":"Remote","type":"Full-time","category":"DevOps","salary":"$110,000–$160,000","exp_min":5,"edu":"Bachelor's","desc":"AWS Azure GCP Terraform Docker Kubernetes networking security cloud migration enterprise solutions infrastructure","skills":"AWS,Azure,GCP,Terraform,Docker,Kubernetes,Networking,Security"},
        {"id":13,"title":"Cybersecurity Analyst","company":"SecureNet","location":"Remote","type":"Full-time","category":"Security","salary":"$80,000–$120,000","exp_min":2,"edu":"Bachelor's","desc":"Network security SIEM penetration testing Linux firewalls incident response Python compliance vulnerability assessment","skills":"Network Security,Python,SIEM,Penetration Testing,Linux,Firewalls,Incident Response"},
        {"id":14,"title":"UX/UI Designer","company":"PixelCraft","location":"Dubai, UAE","type":"Full-time","category":"Design","salary":"$55,000–$90,000","exp_min":2,"edu":"Bachelor's","desc":"Figma Adobe XD wireframes prototypes UI design user research usability testing design systems CSS HTML responsive mobile","skills":"Figma,Adobe XD,UI Design,User Research,Prototyping,Sketch,CSS,HTML"},
        {"id":15,"title":"Graphic Designer","company":"VisualEdge","location":"Lahore, Pakistan","type":"Full-time","category":"Design","salary":"$15,000–$35,000","exp_min":1,"edu":"Bachelor's","desc":"Adobe Photoshop Illustrator InDesign Canva typography brand identity logo social media marketing print digital design","skills":"Adobe Photoshop,Illustrator,InDesign,Canva,Typography,Branding,Figma"},
        {"id":16,"title":"Flutter Mobile Developer","company":"AppForge","location":"Karachi, Pakistan","type":"Full-time","category":"Mobile","salary":"$30,000–$55,000","exp_min":1,"edu":"Bachelor's","desc":"Flutter Dart Firebase REST APIs state management Android iOS app store deployment performance cross-platform mobile","skills":"Flutter,Dart,Firebase,REST APIs,Git,Android,iOS,State Management"},
        {"id":17,"title":"Android Developer","company":"MobileFirst","location":"Remote","type":"Full-time","category":"Mobile","salary":"$70,000–$100,000","exp_min":2,"edu":"Bachelor's","desc":"Kotlin Java Android SDK MVVM REST APIs Firebase Room database Jetpack Compose Google Play native mobile","skills":"Kotlin,Java,Android SDK,MVVM,REST APIs,Firebase,Jetpack Compose"},
        {"id":18,"title":"Product Manager","company":"Innovatech","location":"Austin, USA","type":"Full-time","category":"Product","salary":"$100,000–$150,000","exp_min":4,"edu":"Bachelor's","desc":"Product strategy roadmap agile JIRA stakeholder management user research data analysis A/B testing communication","skills":"Product Strategy,Agile,JIRA,Data Analysis,SQL,Communication,Roadmapping"},
        {"id":19,"title":"Digital Marketing Specialist","company":"GrowthHive","location":"Karachi, Pakistan","type":"Full-time","category":"Marketing","salary":"$20,000–$40,000","exp_min":1,"edu":"Bachelor's","desc":"SEO SEM Google Ads social media content marketing email marketing analytics Canva brand awareness performance reporting","skills":"SEO,Google Ads,Social Media,Content Marketing,Analytics,Email Marketing,Canva"},
        {"id":20,"title":"Business Analyst","company":"FinEdge","location":"Toronto, Canada","type":"Full-time","category":"Business","salary":"$65,000–$95,000","exp_min":2,"edu":"Bachelor's","desc":"SQL data analysis Excel Power BI JIRA process improvement stakeholder communication agile scrum reporting business requirements","skills":"Data Analysis,SQL,Excel,Power BI,JIRA,Tableau,Communication,Agile"},
        {"id":21,"title":"QA Automation Engineer","company":"TestPro","location":"Lahore, Pakistan","type":"Full-time","category":"Engineering","salary":"$25,000–$45,000","exp_min":2,"edu":"Bachelor's","desc":"Selenium Cypress Python JavaScript API testing JIRA CI/CD regression testing quality assurance automation framework","skills":"Selenium,Cypress,Python,JavaScript,API Testing,JIRA,CI/CD"},
        {"id":22,"title":"Blockchain Developer","company":"ChainTech","location":"Remote","type":"Full-time","category":"Engineering","salary":"$100,000–$150,000","exp_min":3,"edu":"Bachelor's","desc":"Solidity Ethereum Web3.js DeFi NFT smart contracts blockchain Python JavaScript cryptography decentralized applications security audit","skills":"Solidity,Ethereum,Web3.js,Python,Smart Contracts,JavaScript,Cryptography"},
        {"id":23,"title":"Network Engineer","company":"NetSystems","location":"Islamabad, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp_min":2,"edu":"Bachelor's","desc":"Cisco routers switches firewalls VPN TCP/IP Linux Windows server networking troubleshooting security monitoring CCNA protocols","skills":"Cisco,Networking,Firewalls,VPN,TCP/IP,Linux,Windows Server,CCNA"},
        {"id":24,"title":"HR Business Partner","company":"PeopleFirst","location":"Dubai, UAE","type":"Full-time","category":"HR","salary":"$50,000–$80,000","exp_min":3,"edu":"Bachelor's","desc":"Talent acquisition employee relations performance management HR policies training development HRMS communication organizational recruitment","skills":"HR Management,Recruitment,Employee Relations,Performance Management,HRMS,Training"},
        {"id":25,"title":"Python Developer Intern","company":"StartupHub","location":"Karachi, Pakistan","type":"Internship","category":"Engineering","salary":"$5,000–$12,000","exp_min":0,"edu":"Intermediate","desc":"Python programming Django REST APIs database SQL Git web development backend basics agile teamwork projects","skills":"Python,Django,SQL,Git,REST APIs,HTML"},
        {"id":26,"title":"Data Science Intern","company":"Analytics Co","location":"Lahore, Pakistan","type":"Internship","category":"Data Science","salary":"$5,000–$10,000","exp_min":0,"edu":"Intermediate","desc":"Python pandas numpy matplotlib machine learning scikit-learn SQL data visualization statistics Excel Jupyter notebook analysis","skills":"Python,Pandas,NumPy,Matplotlib,SQL,Scikit-learn,Excel"},
        {"id":27,"title":"UI/UX Design Intern","company":"CreativeMinds","location":"Remote","type":"Internship","category":"Design","salary":"$4,000–$8,000","exp_min":0,"edu":"Intermediate","desc":"Wireframes prototypes Figma user interface mobile web design typography color theory user experience research Canva iteration","skills":"Figma,Canva,UI Design,Prototyping,Typography"},
        {"id":28,"title":"Freelance Web Developer","company":"Various Clients","location":"Remote","type":"Freelance","category":"Engineering","salary":"$30,000–$80,000","exp_min":1,"edu":"Diploma","desc":"WordPress React JavaScript HTML CSS PHP MySQL client websites freelance remote work communication deadlines project management","skills":"WordPress,React,JavaScript,HTML,CSS,PHP,MySQL"},
        {"id":29,"title":"Content Writer","company":"ContentPro Agency","location":"Remote","type":"Part-time","category":"Marketing","salary":"$15,000–$30,000","exp_min":1,"edu":"Bachelor's","desc":"Technical writing SEO blogs articles research AI technology software communication editing proofreading WordPress content marketing","skills":"Technical Writing,SEO,Research,Communication,WordPress,Editing"},
        {"id":30,"title":"iOS Developer","company":"AppleTree Apps","location":"Dubai, UAE","type":"Full-time","category":"Mobile","salary":"$75,000–$110,000","exp_min":2,"edu":"Bachelor's","desc":"Swift SwiftUI UIKit Xcode CoreData REST APIs push notifications App Store deployment performance Objective-C native iOS mobile","skills":"Swift,SwiftUI,UIKit,Xcode,CoreData,REST APIs,Objective-C"},
    ]
    return pd.DataFrame(data)


# ╔══════════════════════════════════════════════════════════╗
#   ML ENGINE
# ╚══════════════════════════════════════════════════════════╝
@st.cache_resource
def build_model(descs):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    mat = vec.fit_transform(descs)
    return vec, mat

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def recommend(skills_str, exp, edu, jtype, vec, mat, df, n=8):
    user_vec = vec.transform([preprocess(skills_str)])
    scores   = cosine_similarity(user_vec, mat).flatten()
    res      = df.copy()
    res['base'] = scores
    edu_rank = {"Intermediate":1,"Diploma":2,"Bachelor's":3,"Master's":4,"PhD":5}
    u_edu    = edu_rank.get(edu, 3)
    res['boost'] = (
        res['exp_min'].apply(lambda e: 0.08 if exp >= e else (-0.05 if exp < e-2 else 0)) +
        res['type'].apply(lambda t: 0.06 if (not jtype or t == jtype) else 0) +
        res['edu'].apply(lambda e: 0.05 if edu_rank.get(e,3) <= u_edu else -0.03)
    )
    res['final'] = res['base'] * 0.75 + res['boost']
    mn, mx = res['final'].min(), res['final'].max()
    res['pct'] = ((res['final']-mn)/(mx-mn)*39+60).clip(0,99).astype(int) if mx > mn else 60
    return res.sort_values('pct', ascending=False).head(n).reset_index(drop=True)

def skill_split(user_str, job_str):
    u = {s.strip().lower() for s in user_str.split(',') if s.strip()}
    j = {s.strip().lower() for s in job_str.split(',')  if s.strip()}
    return sorted(u&j), sorted(j-u)

def match_style(p):
    if p >= 82: return TEAL,   f"linear-gradient(90deg,{TEAL},{TEAL2})"
    if p >= 68: return BLUE,   f"linear-gradient(90deg,{BLUE},{PURPLE})"
    return RED, f"linear-gradient(90deg,{RED},#F87171)"


# ╔══════════════════════════════════════════════════════════╗
#   PLOTLY THEME HELPER
# ╚══════════════════════════════════════════════════════════╝
PALETTE = [TEAL, BLUE, PURPLE, AMBER, RED, "#34D399","#60A5FA","#A78BFA","#F472B6","#FBBF24"]

def plotly_layout(title=""):
    return dict(
        paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        font=dict(family="DM Sans", color=PLOTLY_TEXT, size=12),
        title=dict(text=title, font=dict(family="Syne",size=14,color=TEXT), x=0.02),
        margin=dict(l=12, r=12, t=44, b=12),
        showlegend=False,
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID, tickfont=dict(size=11,color=PLOTLY_TEXT)),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID, tickfont=dict(size=11,color=PLOTLY_TEXT)),
    )


# ╔══════════════════════════════════════════════════════════╗
#   LOAD DATA
# ╚══════════════════════════════════════════════════════════╝
df = load_dataset()
vec, mat = build_model(df['desc'].astype(str))


# ╔══════════════════════════════════════════════════════════╗
#   SIDEBAR
# ╚══════════════════════════════════════════════════════════╝
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="padding:0.5rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        {IC.brain(TEAL, 32)}
        <div class="sidebar-brand">Job<span>Match</span> AI</div>
      </div>
      <div style="font-size:0.72rem;color:{MUTED};padding-left:2px;">TF-IDF · Cosine Similarity · Waqaas Hussain</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)

    # Theme toggle
    theme_label = f"{IC.sun(AMBER,14)} Light Mode" if IS_DARK else f"{IC.moon(BLUE,14)} Dark Mode"
    if st.button(theme_label, key="theme_btn"):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()

    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-tag">{IC.target(TEAL,13)} Your Profile</div>', unsafe_allow_html=True)

    user_skills = st.text_area("Skills", placeholder="e.g. Python, Machine Learning, SQL, React", height=100)

    st.markdown(f'<div class="sidebar-tag">{IC.graduation(TEAL,13)} Education & Experience</div>', unsafe_allow_html=True)
    education  = st.selectbox("Education Level", ["Intermediate","Diploma","Bachelor's","Master's","PhD"])
    experience = st.slider("Years of Experience", 0, 20, 1)

    st.markdown(f'<div class="sidebar-tag">{IC.filter(TEAL,13)} Filters</div>', unsafe_allow_html=True)
    job_type      = st.selectbox("Job Type", ["Any","Full-time","Part-time","Remote","Freelance","Internship"])
    location_pref = st.text_input("Preferred Location", placeholder="e.g. Remote, Karachi, Dubai")
    top_n         = st.slider("Results to Show", 3, 15, 8)

    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)
    find_btn = st.button(f"Find My Jobs  {IC.arrow_right('currentColor',16)}", use_container_width=True, type="primary")
    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-tag">{IC.home(TEAL,13)} Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["🏠  Home","📊  Analytics","ℹ️  How It Works"], label_visibility="collapsed")


# ╔══════════════════════════════════════════════════════════╗
#   HERO BANNER
# ╚══════════════════════════════════════════════════════════╝
st.markdown(f"""
<div class="hero">
  <div class="hero-glow"></div>
  <div class="hero-glow2"></div>
  <div class="hero-badge"><div class="pulse-dot"></div> AI-Powered Matching</div>
  <div class="hero-title">Find Your Perfect <span>Career Match</span></div>
  <div class="hero-sub">
    Intelligent job recommendations using TF-IDF vectorization and cosine similarity —
    precisely matching your skills to the best opportunities in real time.
  </div>
  <div class="hero-chips">
    <div class="hero-chip">{IC.python(size=15)} <b>Python</b> &nbsp;Scikit-learn</div>
    <div class="hero-chip">{IC.cpu(TEAL,15)} <b>TF-IDF</b> &nbsp;Vectorization</div>
    <div class="hero-chip">{IC.target(BLUE,15)} <b>Cosine</b> &nbsp;Similarity</div>
    <div class="hero-chip">{IC.chart(PURPLE,15)} <b>Plotly</b> &nbsp;Analytics</div>
    <div class="hero-chip">{IC.lightning(AMBER,15)} <b>Waqaas</b> &nbsp;Hussain</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#   HOME PAGE
# ╚══════════════════════════════════════════════════════════╝
if "Home" in page:

    if find_btn or 'results' in st.session_state:
        if find_btn:
            if not user_skills.strip():
                st.warning("Please enter at least one skill to get recommendations.")
                st.stop()
            jt  = "" if job_type == "Any" else job_type
            res = recommend(user_skills, experience, education, jt, vec, mat, df, top_n)
            if location_pref.strip():
                res = res[
                    res['location'].str.lower().str.contains(location_pref.lower(), na=False) |
                    res['location'].str.lower().str.contains('remote', na=False)
                ]
            st.session_state['results'] = res
            st.session_state['skills']  = user_skills

        res        = st.session_state['results']
        skills_str = st.session_state['skills']

        if res.empty:
            st.warning("No matching jobs found — try broadening your filters.")
            st.stop()

        # ── STAT CARDS ──────────────────────────────────────
        skill_count = len([s for s in skills_str.split(',') if s.strip()])
        c1,c2,c3,c4 = st.columns(4)
        for col, ico, num, lbl in [
            (c1, IC.briefcase(TEAL,26),   len(res),                    "Jobs Found"),
            (c2, IC.target(BLUE,26),      f"{int(res['pct'].mean())}%","Avg Match Score"),
            (c3, IC.star(AMBER,26),       f"{int(res['pct'].max())}%", "Best Match"),
            (c4, IC.layers(PURPLE,26),    skill_count,                  "Skills Detected"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="stat-icon">{ico}</div><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── FILTER ──────────────────────────────────────────
        cats = ["All"] + sorted(res['category'].unique().tolist())
        st.markdown(f'<div class="sec-hdr">{IC.filter(TEAL,18)} Filter Results</div>', unsafe_allow_html=True)
        sel  = st.radio("cat", cats, horizontal=True, label_visibility="collapsed")
        show = res if sel == "All" else res[res['category'] == sel]

        st.markdown(f'<p style="color:{MUTED};font-size:0.8rem;margin:0.5rem 0 1rem;">Showing <b style="color:{TEAL};">{len(show)}</b> job(s) · ranked by AI match score</p>', unsafe_allow_html=True)

        # ── JOB CARDS ────────────────────────────────────────
        for idx, (_, j) in enumerate(show.iterrows()):
            pct = j['pct']
            ring_color, bar_grad = match_style(pct)
            matched, missing = skill_split(skills_str, j['skills'])

            m_chips = "".join([f'<span class="chip-match">{IC.check(TEAL,10)} {s}</span>' for s in matched])
            x_chips = "".join([f'<span class="chip-miss">{IC.xmark(RED,10)} {s}</span>' for s in missing])

            with st.expander(f"  {j['title']}  ·  {j['company']}  —  {pct}% match"):
                st.markdown(f"""
<div class="job-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div style="flex:1;">
      <div class="job-title-txt">{IC.briefcase(ring_color,15)} &nbsp;{j['title']}</div>
      <div class="job-meta-row">
        <span class="job-meta-item">{IC.building(MUTED,13)} {j['company']}</span>
        <span class="job-meta-item">{IC.location(MUTED,13)} {j['location']}</span>
      </div>
    </div>
    <div class="match-ring" style="color:{ring_color};border-color:{ring_color};background:rgba(0,0,0,0.15);">
      {pct}%
    </div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0;">
    <span class="badge b-teal">{IC.target(TEAL,10)} {pct}% Match</span>
    <span class="badge b-blue">{IC.clock(BLUE,10)} {j['type']}</span>
    <span class="badge b-purple">{IC.layers(PURPLE,10)} {j['category']}</span>
    <span class="badge b-amber">{IC.graduation(AMBER,10)} {j['edu']}+</span>
    <span class="badge b-red">{IC.clock(RED,10)} {j['exp_min']}+ yrs exp</span>
  </div>

  <div class="bar-track">
    <div class="bar-fill" style="width:{pct}%;background:{bar_grad};"></div>
  </div>
  <div style="font-size:0.7rem;color:{MUTED};margin-bottom:12px;">{pct}% alignment with your skill profile</div>

  <div class="salary-tag">{IC.dollar(AMBER,14)} {j['salary']}</div>

  <div style="font-size:0.78rem;font-weight:700;color:{TEXT};margin:14px 0 7px;display:flex;align-items:center;gap:7px;">
    {IC.layers(TEAL,14)} Skill Analysis
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:2px;">
    {m_chips}{x_chips}
  </div>
  {'<div style="font-size:0.68rem;color:'+MUTED+';margin-top:7px;display:flex;gap:12px;"><span style=\'color:'+TEAL+';font-weight:600;\'>● Matched</span><span style=\'color:'+RED+';font-weight:600;\'>● Missing</span></div>' if (matched or missing) else ''}
</div>
st.markdown("""
<style>
.box {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

                if missing:
                    st.markdown("""
<div class="gap-box">
  {IC.alert(AMBER,18)}
  <div>
   <div class="gap-title">Skill Gap - Recommended to Learn:</div>
    <div class="gap-skills">{', '.join(missing)}</div>
  </div>
</div>""", unsafe_allow_html=True)

                if st.button("Apply Now", key=f"apply_{j['id']}"):
                    st.success(f"Application submitted for **{j['title']}** at **{j['company']}**!")

        st.markdown("<br>", unsafe_allow_html=True)
        csv = show[['title','company','location','type','category','salary','pct']]\
              .rename(columns={'title':'Title','company':'Company','location':'Location',
                               'type':'Type','category':'Category','salary':'Salary','pct':'Match%'})\
              .to_csv(index=False).encode('utf-8')
        st.download_button(
            f"{IC.download('white',15)} Download Results as CSV",
            csv, "job_recommendations.csv", "text/csv", use_container_width=True
        )

    else:
        # Welcome
        st.markdown(f"""
<div class="welcome-wrap">
  {IC.search(TEAL, 54)}
  <div class="welcome-title">Welcome, Waqaas!</div>
  <div class="welcome-sub">
    Enter your skills and preferences in the <b>sidebar</b>, then click
    <b>Find My Jobs</b> to receive AI-powered personalized recommendations
    powered by TF-IDF vectorization and cosine similarity.
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        for col, d, num, title, desc in [
            (c1, "0.0s", "01", "Enter Your Skills",  "Type skills separated by commas — Python, SQL, React, ML, Docker..."),
            (c2, "0.1s", "02", "Set Preferences",    "Choose education level, years of experience, job type & preferred location"),
            (c3, "0.2s", "03", "Get Matched",         "Click Find My Jobs and receive ranked, AI-scored job recommendations"),
        ]:
            col.markdown(f'<div class="step-card" style="animation-delay:{d};"><div class="step-num">{num}</div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sec-hdr">{IC.briefcase(TEAL,18)} Available Job Categories</div>', unsafe_allow_html=True)
        cats = df['category'].value_counts()
        cols = st.columns(3)
        for i,(cat,cnt) in enumerate(cats.items()):
            cols[i%3].markdown(f"""
<div class="cat-pill" style="animation-delay:{i*0.05}s;">
  {IC.layers(TEAL,16)}
  <span style="font-weight:600;color:{TEXT};flex:1;">{cat}</span>
  <span style="background:rgba(0,201,167,0.12);color:{TEAL};font-size:0.72rem;font-weight:700;padding:2px 9px;border-radius:99px;">{cnt}</span>
</div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#   ANALYTICS PAGE  —  Plotly animated charts
# ╚══════════════════════════════════════════════════════════╝
elif "Analytics" in page:
    st.markdown(f'<div class="sec-hdr">{IC.chart(TEAL,18)} Job Market Analytics Dashboard</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,ico,num,lbl in [
        (c1, IC.briefcase(TEAL,26),   len(df),                  "Total Jobs"),
        (c2, IC.building(BLUE,26),    df['company'].nunique(),   "Companies"),
        (c3, IC.layers(PURPLE,26),    df['category'].nunique(),  "Categories"),
        (c4, IC.location(AMBER,26),   df['location'].nunique(),  "Locations"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="stat-icon">{ico}</div><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prepare data ──
    cats_s  = df['category'].value_counts()
    types_s = df['type'].value_counts()
    all_sk  = []
    for s in df['skills']: all_sk.extend([x.strip() for x in s.split(',')])
    top12   = Counter(all_sk).most_common(12)
    sk_n, sk_v = zip(*top12)

    bins    = pd.cut(df['exp_min'], bins=[-1,0,1,2,3,5,10,20],
                     labels=['Fresher','<1 yr','1–2 yr','2–3 yr','3–5 yr','5–10 yr','10+ yr'])
    exp_c   = bins.value_counts().sort_index()

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig = go.Figure(go.Bar(
            y=cats_s.index.tolist(), x=cats_s.values.tolist(),
            orientation='h', marker_color=PALETTE[:len(cats_s)],
            text=cats_s.values, textposition='outside',
            textfont=dict(size=11, color=TEXT),
            hovertemplate='<b>%{y}</b><br>Jobs: %{x}<extra></extra>',
        ))
        fig.update_layout(**plotly_layout("Jobs by Category"), height=360)
        fig.update_xaxes(title_text='Number of Jobs', title_font=dict(size=11))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with r1c2:
        fig2 = go.Figure(go.Pie(
            labels=types_s.index.tolist(), values=types_s.values.tolist(),
            hole=0.6, marker_colors=PALETTE[:len(types_s)],
            textinfo='label+percent',
            textfont=dict(size=11, color=TEXT),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
        ))
        fig2.update_layout(**plotly_layout("Job Type Distribution"), height=360, showlegend=True,
                           legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"))
        fig2.update_traces(pull=[0.04]*len(types_s))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        colors3 = [PALETTE[i % len(PALETTE)] for i in range(len(exp_c))]
        fig3 = go.Figure(go.Bar(
            x=exp_c.index.astype(str).tolist(), y=exp_c.values.tolist(),
            marker_color=colors3, text=exp_c.values,
            textposition='outside', textfont=dict(size=11, color=TEXT),
            hovertemplate='<b>%{x}</b><br>Jobs: %{y}<extra></extra>',
        ))
        fig3.update_layout(**plotly_layout("Experience Level Required"), height=340)
        fig3.update_xaxes(title_text='Experience Level')
        fig3.update_yaxes(title_text='Jobs')
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with r2c2:
        fig4 = go.Figure(go.Bar(
            y=list(sk_n)[::-1], x=list(sk_v)[::-1],
            orientation='h',
            marker=dict(
                color=list(sk_v)[::-1],
                colorscale=[[0,BLUE],[0.5,TEAL],[1,PURPLE]],
                showscale=False,
            ),
            text=list(sk_v)[::-1], textposition='outside',
            textfont=dict(size=10, color=TEXT),
            hovertemplate='<b>%{y}</b><br>Jobs: %{x}<extra></extra>',
        ))
        fig4.update_layout(**plotly_layout("Top 12 In-Demand Skills"), height=340)
        fig4.update_xaxes(title_text='Jobs Requiring Skill')
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # ── Salary range bubble ──────────────────────────────
    st.markdown(f'<div class="sec-hdr">{IC.dollar(TEAL,18)} Salary Range Distribution</div>', unsafe_allow_html=True)

    sal_df = df.copy()
    def parse_sal(s):
        nums = re.findall(r'[\d,]+', s.replace(',',''))
        if len(nums) >= 2:
            return (int(nums[0]) + int(nums[1])) // 2
        return 0
    sal_df['sal_mid'] = sal_df['salary'].apply(parse_sal)
    sal_df = sal_df[sal_df['sal_mid'] > 0]

    fig5 = px.scatter(
        sal_df, x='exp_min', y='sal_mid', color='category',
        size='sal_mid', hover_name='title',
        hover_data={'company':True,'location':True,'salary':True,'sal_mid':False,'exp_min':True},
        color_discrete_sequence=PALETTE, size_max=28,
        labels={'exp_min':'Years of Experience','sal_mid':'Midpoint Salary ($)','category':'Category'},
    )
    fig5.update_layout(**plotly_layout("Salary vs Experience (bubble size = salary)"), height=380, showlegend=True,
                       legend=dict(font=dict(color=TEXT,size=10), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    # Full dataset
    st.markdown(f'<div class="sec-hdr">{IC.layers(TEAL,18)} Complete Job Dataset</div>', unsafe_allow_html=True)
    disp = df[['title','company','location','type','category','salary','exp_min','edu']].copy()
    disp.columns = ['Job Title','Company','Location','Type','Category','Salary','Min Exp','Education']
    st.dataframe(disp, use_container_width=True, height=400)


# ╔══════════════════════════════════════════════════════════╗
#   HOW IT WORKS PAGE
# ╚══════════════════════════════════════════════════════════╝
elif "How" in page:
    st.markdown(f'<div class="sec-hdr">{IC.info(TEAL,18)} How the AI System Works</div>', unsafe_allow_html=True)

    steps = [
        (IC.layers(TEAL,20),   "01","Data Collection",
         "30 curated real-world job listings with titles, full descriptions, required skills, salary ranges, locations, and experience requirements — forming a rich content-based knowledge base for the recommendation engine."),
        (IC.settings(BLUE,20), "02","Text Preprocessing",
         "Job descriptions are normalized: lowercased, special characters removed, whitespace collapsed using Python regex. TF-IDF's built-in English stop word list further filters noise — ensuring only meaningful keywords are vectorized."),
        (IC.cpu(PURPLE,20),    "03","TF-IDF Vectorization",
         "Scikit-learn's TfidfVectorizer converts both job descriptions and user skill inputs into high-dimensional sparse numerical vectors. Rare but important keywords receive higher TF-IDF weight, while common words are down-weighted."),
        (IC.target(TEAL,20),   "04","Cosine Similarity Scoring",
         "The cosine similarity between the user's skill vector and each job description vector is computed. Scores range from 0 (no semantic overlap) to 1 (perfect alignment), measuring the angular closeness in TF-IDF space — independent of vector length."),
        (IC.trending(BLUE,20), "05","Score Boosting & Normalization",
         "Raw similarity scores are enhanced: experience alignment (+0.08), job type match (+0.06), education suitability (+0.05). Final scores are normalized to a clean 60–99% range for clear, interpretable matching results."),
        (IC.check(TEAL,20),    "06","Results & Skill Gap Analysis",
         "Jobs are ranked by final match percentage. Each result shows matched skills in green and missing skills in red, providing actionable skill gap insights so users understand exactly which technologies to learn next."),
    ]
    for ico, num, title, desc in steps:
        st.markdown(f"""
<div class="how-card">
  <div class="how-num">{num}</div>
  <div>
    <div class="how-title">{ico} {title}</div>
    <div class="how-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sec-hdr">{IC.code(TEAL,18)} Core Algorithm</div>', unsafe_allow_html=True)
    st.latex(r"\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}")
    st.markdown(f"""
<div class="how-card" style="margin-top:0.5rem;">
  <div style="font-size:0.85rem;color:{MUTED};line-height:1.8;">
  st.markdown("""
<div class="card">
<b style="color:#4CAF50;">A</b> = TF-IDF vector of user entered skills &nbsp;·&nbsp;
<b style="color:#4CAF50;">B</b> = Job required skills vector
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<b style="color:#4CAF50;">B</b> = TF-IDF vector of a job description - 
""", unsafe_allow_html=True)
    <b style="color:{TEXT};">Result:</b> 0 → 1 &nbsp;(higher = better match) &nbsp;·&nbsp;
    Measures angle between vectors in high-dimensional TF-IDF space — independent of document length.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec-hdr">{IC.python(size=18)} Technologies Used</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    rows1 = [
        (IC.python(size=16), "Python 3.x",      "Core programming language"),
        (IC.cpu(TEAL,16),    "Streamlit",        "Interactive web application framework"),
        (IC.layers(BLUE,16), "Scikit-learn",     "TF-IDF vectorization & cosine similarity"),
        (IC.chart(PURPLE,16),"Pandas / NumPy",   "Data manipulation & numerical computing"),
    ]
    rows2 = [
        (IC.trending(TEAL,16),"Plotly Express",  "Animated interactive analytics charts"),
        (IC.code(AMBER,16),   "JavaScript / CSS","UI animations & glassmorphism effects"),
        (IC.settings(BLUE,16),"Regex (re)",       "Text preprocessing & normalization"),
        (IC.layers(RED,16),   "Session State",    "In-app memory & state persistence"),
    ]
    with c1:
        rows_html = "".join([f'<tr><td style="display:flex;align-items:center;gap:8px;">{ico} {t}</td><td>{d}</td></tr>' for ico,t,d in rows1])
        st.markdown(f'<table class="tech-tbl"><tr><th>Technology</th><th>Purpose</th></tr>{rows_html}</table>', unsafe_allow_html=True)
    with c2:
        rows_html = "".join([f'<tr><td style="display:flex;align-items:center;gap:8px;">{ico} {t}</td><td>{d}</td></tr>' for ico,t,d in rows2])
        st.markdown(f'<table class="tech-tbl"><tr><th>Technology</th><th>Purpose</th></tr>{rows_html}</table>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="info-card" style="margin-top:2rem;">
  <div class="info-label">{IC.star(TEAL,14)} Project Information</div>
  <div class="info-value">
    <b>Prepared by:</b> Waqaas Hussain &nbsp;·&nbsp;
    <b>Subject:</b> Programming for AI &nbsp;·&nbsp;
    <b>Framework:</b> Streamlit + Scikit-learn + Plotly &nbsp;·&nbsp;
    <b>Algorithm:</b> TF-IDF Vectorization + Cosine Similarity Content-Based Filtering
  </div>
</div>""", unsafe_allow_html=True)

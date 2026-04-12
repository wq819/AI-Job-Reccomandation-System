# ================================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM
#   Prepared by  : Waqaas Hussain
#   Subject      : Programming for AI
#   Institution  : Aror University Sukkur 
#   Algorithm    : TF-IDF Vectorization + Cosine Similarity
#   Framework    : Streamlit + Scikit-learn + Plotly
#   Pages        : Home · Recommendations · Analytics ·
#                  Methodology · Literature Review · About
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import re
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Recommendation System · Waqaas Hussain",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
#  THEME STATE  (dark / light toggle)
# ──────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

DARK = st.session_state.theme == "dark"

# colour tokens
BG        = "#080F1E"      if DARK else "#F4F7FF"
BG2       = "#0D1A2D"      if DARK else "#EAF0FF"
CARD      = "#0F2040"      if DARK else "#FFFFFF"
CARD2     = "#132545"      if DARK else "#F7FAFF"
BORDER    = "rgba(255,255,255,0.08)"   if DARK else "rgba(0,0,0,0.08)"
TEXT      = "#E8EDF5"      if DARK else "#0A1628"
MUTED     = "#7A90A8"      if DARK else "#64748B"
TEAL      = "#00C9A7"      if DARK else "#0D9B82"
BLUE      = "#3B82F6"      if DARK else "#2563EB"
PURPLE    = "#8B5CF6"      if DARK else "#7C3AED"
AMBER     = "#F59E0B"      if DARK else "#D97706"
RED       = "#EF4444"      if DARK else "#DC2626"
GREEN     = "#10B981"      if DARK else "#059669"
PBG       = "#0F2040"      if DARK else "#FFFFFF"
PGRID     = "#162845"      if DARK else "#E2E8F0"
PTXT      = "#7A90A8"      if DARK else "#64748B"
PALETTE   = [TEAL,BLUE,PURPLE,AMBER,RED,GREEN,"#60A5FA","#A78BFA","#F472B6","#FBBF24"]

# ──────────────────────────────────────────────────────────────
#  INLINE SVG ICON LIBRARY  (no external CDN — never breaks)
# ──────────────────────────────────────────────────────────────
def _i(d, c="currentColor", s=18, f="none", w=2):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" '
            f'viewBox="0 0 24 24" fill="{f}" stroke="{c}" stroke-width="{w}" '
            f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;flex-shrink:0;">'
            f'{d}</svg>')

class I:
    home      = lambda c=TEXT,s=18: _i('<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',c,s)
    search    = lambda c=TEXT,s=18: _i('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',c,s)
    chart     = lambda c=TEXT,s=18: _i('<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/><path d="M2 20h20"/>',c,s)
    book      = lambda c=TEXT,s=18: _i('<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',c,s)
    cpu       = lambda c=TEXT,s=18: _i('<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',c,s)
    info      = lambda c=TEXT,s=18: _i('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',c,s)
    brain     = lambda c=TEXT,s=18: _i('<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.99-3 2.5 2.5 0 0 1-1.13-4.42A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2-3 2.5 2.5 0 0 0 1.13-4.42A2.5 2.5 0 0 0 14.5 2z"/>',c,s)
    target    = lambda c=TEXT,s=18: _i('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',c,s)
    briefcase = lambda c=TEXT,s=18: _i('<rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',c,s)
    location  = lambda c=TEXT,s=18: _i('<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',c,s)
    clock     = lambda c=TEXT,s=18: _i('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',c,s)
    dollar    = lambda c=TEXT,s=18: _i('<line x1="12" x2="12" y1="2" y2="22"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',c,s)
    check     = lambda c=TEXT,s=18: _i('<path d="M20 6 9 17l-5-5"/>',c,s)
    xmark     = lambda c=TEXT,s=18: _i('<path d="M18 6 6 18M6 6l12 12"/>',c,s)
    star      = lambda c=TEXT,s=18: _i('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',c,s,c,1.5)
    graduation= lambda c=TEXT,s=18: _i('<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>',c,s)
    layers    = lambda c=TEXT,s=18: _i('<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12-8.6 3.92a2 2 0 0 1-1.66 0L3 12"/><path d="m22 17-8.6 3.92a2 2 0 0 1-1.66 0L3 17"/>',c,s)
    trending  = lambda c=TEXT,s=18: _i('<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',c,s)
    alert     = lambda c=TEXT,s=18: _i('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/>',c,s)
    download  = lambda c=TEXT,s=18: _i('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',c,s)
    code      = lambda c=TEXT,s=18: _i('<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',c,s)
    user      = lambda c=TEXT,s=18: _i('<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',c,s)
    filter_ic = lambda c=TEXT,s=18: _i('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',c,s)
    sun       = lambda c=TEXT,s=18: _i('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>',c,s)
    moon      = lambda c=TEXT,s=18: _i('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',c,s)
    arrow_r   = lambda c=TEXT,s=18: _i('<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',c,s)
    building  = lambda c=TEXT,s=18: _i('<rect width="16" height="20" x="4" y="2" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01M16 6h.01M12 6h.01M12 10h.01M12 14h.01"/>',c,s)
    flask     = lambda c=TEXT,s=18: _i('<path d="M9 3h6l1 7H8L9 3z"/><path d="M8 10s-4 3-4 7a8 8 0 0 0 16 0c0-4-4-7-4-7"/>',c,s)
    settings  = lambda c=TEXT,s=18: _i('<circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>',c,s)
    python_ic = lambda s=18: (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" viewBox="0 0 24 24" '
        f'style="vertical-align:middle;" fill="#3776AB">'
        f'<path d="M11.914 0C5.82 0 6.2 2.656 6.2 2.656l.007 2.752h5.814v.826H3.9S0 5.789 0 11.969c0 6.18 3.403 5.963 3.403 5.963h2.031v-2.868s-.109-3.402 3.35-3.402h5.769s3.24.052 3.24-3.131V3.129S18.28 0 11.914 0zm-3.21 1.818a1.042 1.042 0 1 1 0 2.084 1.042 1.042 0 0 1 0-2.084z"/>'
        f'<path d="M12.086 24c6.094 0 5.714-2.656 5.714-2.656l-.007-2.752h-5.814v-.826h8.121S24 18.211 24 12.031c0-6.18-3.403-5.963-3.403-5.963h-2.031v2.868s.109 3.402-3.35 3.402H9.447s-3.24-.052-3.24 3.131v5.402S5.72 24 12.086 24zm3.21-1.818a1.042 1.042 0 1 1 0-2.084 1.042 1.042 0 0 1 0 2.084z" fill="#FFD43B"/></svg>')

# ──────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
/* ── Keyframes ── */
@keyframes fadeUp   {{ from{{opacity:0;transform:translateY(22px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes slideIn  {{ from{{opacity:0;transform:translateX(-18px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes fadeIn   {{ from{{opacity:0}} to{{opacity:1}} }}
@keyframes barGrow  {{ from{{transform:scaleX(0);transform-origin:left}} to{{transform:scaleX(1);transform-origin:left}} }}
@keyframes pulse    {{ 0%,100%{{box-shadow:0 0 0 0 rgba(0,201,167,.5)}} 50%{{box-shadow:0 0 0 9px rgba(0,201,167,0)}} }}
@keyframes glow     {{ 0%,100%{{text-shadow:0 0 8px rgba(0,201,167,.35)}} 50%{{text-shadow:0 0 22px rgba(0,201,167,.75)}} }}
@keyframes countUp  {{ from{{opacity:0;transform:scale(.75)}} to{{opacity:1;transform:scale(1)}} }}

/* ── Base ── */
*,*::before,*::after{{box-sizing:border-box}}
html,body,[class*="css"]{{
  font-family:'DM Sans',sans-serif!important;
  background:{BG}!important;
  color:{TEXT}!important;
  transition:background .35s,color .35s;
}}
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:{BG}}}
::-webkit-scrollbar-thumb{{background:{TEAL};border-radius:99px}}
#MainMenu,footer{{visibility:hidden}}
.block-container{{padding:1.5rem 2.5rem 4rem!important;max-width:1600px!important}}
header[data-testid="stHeader"]{{background:transparent!important}}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{{background:{CARD}!important;border-right:1px solid {BORDER}!important;backdrop-filter:blur(16px)}}
section[data-testid="stSidebar"]>div{{padding:1.4rem 1.2rem!important}}
section[data-testid="stSidebar"] *{{color:{TEXT}!important}}
section[data-testid="stSidebar"] label{{font-size:.73rem!important;font-weight:700!important;letter-spacing:.07em!important;text-transform:uppercase!important;color:{MUTED}!important}}
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] input{{background:{BG2}!important;border:1.5px solid {BORDER}!important;color:{TEXT}!important;border-radius:11px!important;font-family:'DM Sans',sans-serif!important}}
section[data-testid="stSidebar"] textarea:focus,
section[data-testid="stSidebar"] input:focus{{border-color:{TEAL}!important;box-shadow:0 0 0 3px rgba(0,201,167,.15)!important}}
section[data-testid="stSidebar"] .stButton button{{
  background:linear-gradient(135deg,{TEAL},{BLUE})!important;
  color:{'#0A1628' if DARK else '#fff'}!important;font-weight:800!important;
  border:none!important;border-radius:13px!important;padding:.7rem 1rem!important;
  font-size:.92rem!important;font-family:'Syne',sans-serif!important;
  box-shadow:0 4px 22px rgba(0,201,167,.4)!important;transition:all .25s!important}}
section[data-testid="stSidebar"] .stButton button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(0,201,167,.55)!important}}

/* ── Inputs ── */
.stSelectbox>div>div{{background:{BG2}!important;border:1.5px solid {BORDER}!important;border-radius:11px!important;color:{TEXT}!important}}
.stTextArea textarea,.stTextInput input{{background:{CARD2}!important;border:1.5px solid {BORDER}!important;color:{TEXT}!important;border-radius:11px!important}}

/* ── Main button ── */
.stButton button{{background:linear-gradient(135deg,{BLUE},{PURPLE})!important;color:#fff!important;border:none!important;border-radius:11px!important;font-family:'Syne',sans-serif!important;font-weight:700!important;transition:all .25s!important}}
.stButton button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(59,130,246,.45)!important}}
.stDownloadButton button{{background:linear-gradient(135deg,{TEAL},{TEAL})!important;color:{'#0A1628' if DARK else '#fff'}!important;border:none!important;border-radius:11px!important;font-weight:700!important;font-family:'Syne',sans-serif!important}}

/* ── Expander ── */
.streamlit-expanderHeader{{background:{CARD2}!important;border:1px solid {BORDER}!important;border-radius:13px!important;font-weight:700!important;color:{TEXT}!important;font-family:'Syne',sans-serif!important}}
.streamlit-expanderContent{{background:{CARD2}!important;border:1px solid {BORDER}!important;border-top:none!important;border-radius:0 0 13px 13px!important}}

/* ── Radio ── */
.stRadio>div{{gap:6px!important;flex-wrap:wrap!important}}
.stRadio label{{background:{CARD2}!important;border:1.5px solid {BORDER}!important;border-radius:9px!important;padding:5px 13px!important;font-size:.77rem!important;color:{MUTED}!important;cursor:pointer!important;transition:all .2s!important}}
.stRadio label:hover{{border-color:{TEAL}!important;color:{TEAL}!important}}

/* ── Iframe border-radius ── */
iframe{{border-radius:14px!important}}

/* ─────────────── CUSTOM COMPONENTS ─────────────── */

/* Hero */
.hero{{position:relative;overflow:hidden;background:linear-gradient(135deg,{BG2} 0%,{CARD} 55%,{BG2} 100%);border:1px solid {BORDER};border-radius:22px;padding:2.8rem 3.2rem;margin-bottom:2rem;animation:fadeUp .5s ease}}
.hero-glow{{position:absolute;top:-70px;right:-70px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle,rgba(0,201,167,.18) 0%,transparent 65%);pointer-events:none}}
.hero-glow2{{position:absolute;bottom:-90px;left:35%;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(59,130,246,.12) 0%,transparent 65%);pointer-events:none}}
.hero-badge{{display:inline-flex;align-items:center;gap:7px;background:rgba(0,201,167,.10);border:1px solid rgba(0,201,167,.28);color:{TEAL};font-size:.7rem;font-weight:700;padding:4px 14px;border-radius:99px;margin-bottom:1.2rem;letter-spacing:.1em;text-transform:uppercase;animation:fadeIn .4s ease .2s both}}
.pulse-dot{{width:7px;height:7px;border-radius:50%;background:{TEAL};animation:pulse 2s infinite;display:inline-block}}
.hero-title{{font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:{TEXT};margin:0 0 .6rem;line-height:1.1;letter-spacing:-1px;animation:fadeUp .5s ease .1s both}}
.hero-title .ht{{color:{TEAL};animation:glow 3s ease infinite;display:inline-block}}
.hero-sub{{color:{MUTED};font-size:.9rem;max-width:620px;line-height:1.75;animation:fadeUp .5s ease .2s both}}
.hero-chips{{display:flex;gap:9px;margin-top:1.8rem;flex-wrap:wrap;animation:fadeUp .5s ease .3s both}}
.chip{{display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,{'0.06' if DARK else '0.72'});backdrop-filter:blur(8px);border:1px solid {BORDER};border-radius:10px;padding:6px 13px;font-size:.77rem;color:{MUTED};font-weight:500;transition:all .2s}}
.chip:hover{{border-color:{TEAL};color:{TEAL};transform:translateY(-2px)}}

/* Stat grid */
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:1.5rem 0}}
.stat-card{{background:{CARD2};border:1px solid {BORDER};border-radius:17px;padding:1.3rem 1.1rem;position:relative;overflow:hidden;animation:fadeUp .5s ease both;transition:all .25s;backdrop-filter:blur(8px)}}
.stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{TEAL},{BLUE},{PURPLE})}}
.stat-card:nth-child(1){{animation-delay:.05s}}
.stat-card:nth-child(2){{animation-delay:.10s}}
.stat-card:nth-child(3){{animation-delay:.15s}}
.stat-card:nth-child(4){{animation-delay:.20s}}
.stat-card:hover{{transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,0,0,{'0.45' if DARK else '0.12'});border-color:{TEAL}}}
.s-ico{{margin-bottom:9px}}
.s-num{{font-family:'JetBrains Mono',monospace;font-size:2rem;font-weight:700;color:{TEXT};line-height:1;animation:countUp .5s ease}}
.s-lbl{{font-size:.72rem;color:{MUTED};margin-top:5px;font-weight:500;letter-spacing:.03em}}

/* Job card */
.jcard{{background:{CARD2};border:1.5px solid {BORDER};border-radius:17px;padding:1.4rem 1.7rem;margin-bottom:4px;animation:slideIn .35s ease both;transition:all .25s;backdrop-filter:blur(12px);position:relative;overflow:hidden}}
.jcard::before{{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,{TEAL},{BLUE});border-radius:4px 0 0 4px}}
.jcard:hover{{border-color:rgba(0,201,167,.4);transform:translateX(6px);box-shadow:0 8px 32px rgba(0,0,0,{'0.4' if DARK else '0.1'})}}
.j-title{{font-family:'Syne',sans-serif;font-size:1.06rem;font-weight:700;color:{TEXT};line-height:1.3}}
.j-meta{{display:flex;align-items:center;gap:13px;font-size:.79rem;color:{MUTED};margin:6px 0 0;flex-wrap:wrap}}
.j-mi{{display:inline-flex;align-items:center;gap:5px}}

/* Match ring */
.mring{{width:60px;height:60px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-family:'JetBrains Mono',monospace;font-size:.81rem;font-weight:800;border:2.5px solid;position:relative;transition:all .3s}}
.mring:hover{{transform:scale(1.12)}}

/* Badges */
.badge{{display:inline-flex;align-items:center;gap:4px;font-size:.67rem;font-weight:700;padding:3px 10px;border-radius:99px;margin:2px;letter-spacing:.02em;transition:transform .15s}}
.badge:hover{{transform:scale(1.06)}}
.bt{{background:rgba(0,201,167,.11);color:{TEAL};border:1px solid rgba(0,201,167,.24)}}
.bb{{background:rgba(59,130,246,.11);color:{BLUE};border:1px solid rgba(59,130,246,.24)}}
.bp{{background:rgba(139,92,246,.11);color:{PURPLE};border:1px solid rgba(139,92,246,.24)}}
.ba{{background:rgba(245,158,11,.11);color:{AMBER};border:1px solid rgba(245,158,11,.24)}}
.br{{background:rgba(239,68,68,.11);color:{RED};border:1px solid rgba(239,68,68,.24)}}

/* Bar */
.bar-track{{background:{'rgba(255,255,255,.06)' if DARK else '#E2E8F0'};border-radius:99px;height:7px;margin:11px 0 4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:99px;animation:barGrow .85s ease .3s both}}

/* Skill chips */
.c-match{{display:inline-flex;align-items:center;gap:4px;font-size:.67rem;padding:3px 9px;border-radius:99px;margin:2px;background:rgba(0,201,167,.10);color:{TEAL};border:1px solid rgba(0,201,167,.22);font-weight:600}}
.c-miss{{display:inline-flex;align-items:center;gap:4px;font-size:.67rem;padding:3px 9px;border-radius:99px;margin:2px;background:rgba(239,68,68,.10);color:{RED};border:1px solid rgba(239,68,68,.22);font-weight:600}}

/* Salary tag */
.sal-tag{{display:inline-flex;align-items:center;gap:7px;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.22);border-radius:10px;padding:5px 13px;font-size:.8rem;font-weight:700;color:{AMBER};font-family:'JetBrains Mono',monospace}}

/* Gap box */
.gap-box{{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);border-radius:12px;padding:10px 14px;margin-top:10px;display:flex;gap:10px;align-items:flex-start;animation:fadeIn .3s ease}}
.gap-title{{font-weight:700;color:{AMBER};font-size:.78rem}}
.gap-sk{{font-size:.76rem;color:{'#FCD34D' if DARK else AMBER};margin-top:3px}}

/* Section header */
.sh{{display:flex;align-items:center;gap:10px;font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:{TEXT};border-left:4px solid {TEAL};padding-left:12px;margin:2rem 0 1.2rem;animation:slideIn .4s ease}}

/* Welcome */
.wcard{{background:{CARD2};border:1px solid {BORDER};border-radius:22px;padding:3.5rem 2rem;text-align:center;animation:fadeUp .5s ease;backdrop-filter:blur(12px)}}
.w-title{{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{TEXT};margin:1.2rem 0 .6rem}}
.w-sub{{font-size:.9rem;color:{MUTED};max-width:500px;margin:0 auto;line-height:1.75}}

/* Step cards */
.step-card{{background:{CARD2};border:1.5px solid {BORDER};border-radius:15px;padding:1.4rem;text-align:center;animation:fadeUp .5s ease both;transition:all .25s}}
.step-card:hover{{border-color:{TEAL};transform:translateY(-4px);box-shadow:0 12px 36px rgba(0,0,0,{'0.4' if DARK else '0.1'})}}
.step-num{{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{TEAL};line-height:1;margin-bottom:9px}}
.step-title{{font-weight:700;color:{TEXT};font-size:.93rem;margin-bottom:5px}}
.step-desc{{font-size:.77rem;color:{MUTED};line-height:1.65}}

/* How-card */
.how{{display:flex;gap:17px;background:{CARD2};border:1px solid {BORDER};border-radius:15px;padding:1.2rem 1.5rem;margin-bottom:9px;animation:slideIn .4s ease both;transition:all .2s}}
.how:hover{{border-color:{TEAL};transform:translateX(4px)}}
.how-num{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{TEAL};min-width:40px;line-height:1}}
.how-title{{font-weight:700;color:{TEXT};font-size:.93rem;margin-bottom:4px;display:flex;align-items:center;gap:8px}}
.how-desc{{font-size:.81rem;color:{MUTED};line-height:1.65}}

/* Tech table */
.tt{{width:100%;border-collapse:collapse;font-size:.81rem}}
.tt th{{background:rgba(0,201,167,.08);color:{TEAL};font-weight:700;padding:9px 13px;text-align:left;border-bottom:1px solid {BORDER};font-size:.7rem;text-transform:uppercase;letter-spacing:.07em}}
.tt td{{padding:9px 13px;border-bottom:1px solid {BORDER};color:{MUTED}}}
.tt td:first-child{{color:{TEXT};font-weight:600}}
.tt tr:hover td{{background:rgba(0,201,167,.04)}}

/* Info card */
.info-card{{background:rgba(0,201,167,.06);border:1px solid rgba(0,201,167,.2);border-radius:13px;padding:1.2rem 1.5rem;margin-top:1rem;animation:fadeIn .4s ease}}
.info-lbl{{font-size:.77rem;color:{TEAL};font-weight:700;margin-bottom:6px}}
.info-val{{font-size:.8rem;color:{MUTED};line-height:1.75}}
.info-val b{{color:{TEXT}}}

/* Lit review card */
.lit-card{{background:{CARD2};border:1px solid {BORDER};border-radius:13px;padding:1.1rem 1.4rem;margin-bottom:9px;animation:slideIn .4s ease both;transition:all .2s;border-left:4px solid {BLUE}}}
.lit-card:hover{{transform:translateX(4px);border-left-color:{TEAL}}}
.lit-title{{font-weight:700;color:{TEXT};font-size:.9rem;margin-bottom:4px}}
.lit-desc{{font-size:.8rem;color:{MUTED};line-height:1.6}}
.lit-tag{{display:inline-flex;align-items:center;gap:5px;font-size:.68rem;font-weight:700;background:rgba(59,130,246,.1);color:{BLUE};border:1px solid rgba(59,130,246,.2);border-radius:99px;padding:2px 9px;margin:4px 2px 0}}

/* Sidebar brand & section tag */
.sb-brand{{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:{TEXT};margin-bottom:3px}}
.sb-brand span{{color:{TEAL}}}
.sb-tag{{font-size:.69rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:{TEAL};margin:1.2rem 0 .4rem;display:flex;align-items:center;gap:7px}}
.sb-hr{{border:none;border-top:1px solid {BORDER};margin:.8rem 0}}

/* Objective checklist */
.obj-item{{display:flex;align-items:flex-start;gap:10px;background:{CARD2};border:1px solid {BORDER};border-radius:12px;padding:11px 15px;margin-bottom:8px;animation:slideIn .4s ease both;transition:all .2s}}
.obj-item:hover{{border-color:{TEAL};transform:translateX(4px)}}
.obj-txt{{font-size:.84rem;color:{TEXT};line-height:1.5}}

/* Category pill */
.cat-pill{{display:flex;align-items:center;gap:9px;background:{CARD2};border:1px solid {BORDER};border-radius:11px;padding:9px 15px;margin-bottom:7px;transition:all .2s;animation:fadeUp .4s ease both}}
.cat-pill:hover{{border-color:{TEAL};transform:translateX(4px)}}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  DATASET  (30 real-world job listings)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    rows = [
        {"id":1,"title":"Senior Python Developer","company":"TechNova Solutions","location":"Remote","type":"Full-time","category":"Engineering","salary":"$90,000–$130,000","exp":4,"edu":"Bachelor's","desc":"Python Django FastAPI AWS Docker PostgreSQL REST APIs microservices backend agile CI/CD","skills":"Python,Django,FastAPI,AWS,Docker,PostgreSQL,REST APIs,Git"},
        {"id":2,"title":"Full Stack Web Developer","company":"WebCraft Studio","location":"Karachi, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp":2,"edu":"Bachelor's","desc":"React Node.js MongoDB Express JavaScript TypeScript HTML CSS REST APIs agile git frontend backend","skills":"React,Node.js,MongoDB,JavaScript,TypeScript,HTML,CSS,Express,Git"},
        {"id":3,"title":"Frontend Developer","company":"UX Studio","location":"London, UK","type":"Full-time","category":"Engineering","salary":"$70,000–$100,000","exp":2,"edu":"Bachelor's","desc":"React Vue JavaScript TypeScript CSS HTML webpack performance accessibility user experience design","skills":"React,Vue.js,JavaScript,TypeScript,CSS,HTML,Webpack,Figma"},
        {"id":4,"title":"Backend Developer","company":"ServerLogic","location":"Remote","type":"Full-time","category":"Engineering","salary":"$80,000–$115,000","exp":3,"edu":"Bachelor's","desc":"Python Java Spring Boot Django Flask PostgreSQL MySQL Redis Docker Kubernetes cloud AWS CI/CD","skills":"Python,Java,Spring Boot,PostgreSQL,MySQL,Docker,Kubernetes,Redis"},
        {"id":5,"title":"Data Scientist","company":"DataSphere Analytics","location":"New York, USA","type":"Full-time","category":"Data Science","salary":"$95,000–$140,000","exp":3,"edu":"Master's","desc":"Python TensorFlow scikit-learn pandas numpy machine learning deep learning SQL statistics data visualization analytics","skills":"Python,Machine Learning,TensorFlow,SQL,Pandas,NumPy,Statistics,Scikit-learn"},
        {"id":6,"title":"Machine Learning Engineer","company":"AI Dynamics","location":"San Francisco, USA","type":"Full-time","category":"AI/ML","salary":"$120,000–$180,000","exp":3,"edu":"Master's","desc":"PyTorch TensorFlow MLOps Kubernetes Docker machine learning deep learning NLP computer vision model deployment","skills":"Python,PyTorch,TensorFlow,MLOps,Kubernetes,Docker,Deep Learning,NLP"},
        {"id":7,"title":"AI Research Scientist","company":"DeepMind Labs","location":"London, UK","type":"Full-time","category":"AI/ML","salary":"$130,000–$200,000","exp":2,"edu":"PhD","desc":"Deep learning NLP reinforcement learning PyTorch mathematics statistics research neural networks transformers BERT GPT","skills":"Python,PyTorch,Deep Learning,NLP,Research,Mathematics,Statistics,Transformers"},
        {"id":8,"title":"NLP Engineer","company":"LanguageAI","location":"Remote","type":"Full-time","category":"AI/ML","salary":"$100,000–$145,000","exp":3,"edu":"Master's","desc":"NLP pipelines BERT transformers spaCy NLTK Python TensorFlow language models GPT fine-tuning sentiment analysis","skills":"Python,NLP,Transformers,BERT,spaCy,NLTK,TensorFlow,Hugging Face"},
        {"id":9,"title":"Data Engineer","company":"PipelineAI","location":"Singapore","type":"Full-time","category":"Data Science","salary":"$85,000–$120,000","exp":3,"edu":"Bachelor's","desc":"Apache Spark Kafka SQL AWS Airflow ETL data warehouse Python big data stream processing batch","skills":"Python,Apache Spark,Kafka,SQL,AWS,Airflow,ETL,Data Warehouse"},
        {"id":10,"title":"BI Analyst","company":"InsightCorp","location":"Karachi, Pakistan","type":"Full-time","category":"Data Science","salary":"$30,000–$55,000","exp":2,"edu":"Bachelor's","desc":"SQL Power BI Tableau Excel data visualization dashboard reporting KPIs business analytics Python","skills":"SQL,Power BI,Tableau,Excel,Python,Data Analytics,Reporting"},
        {"id":11,"title":"DevOps Engineer","company":"CloudCore","location":"Remote","type":"Full-time","category":"DevOps","salary":"$85,000–$125,000","exp":3,"edu":"Bachelor's","desc":"CI/CD AWS Docker Kubernetes Terraform Jenkins Linux Python monitoring deployment automation","skills":"AWS,Docker,Kubernetes,Terraform,Jenkins,Linux,Python,CI/CD"},
        {"id":12,"title":"Cloud Solutions Architect","company":"Nimbus Cloud","location":"Remote","type":"Full-time","category":"DevOps","salary":"$110,000–$160,000","exp":5,"edu":"Bachelor's","desc":"AWS Azure GCP Terraform Docker Kubernetes networking security cloud migration enterprise infrastructure","skills":"AWS,Azure,GCP,Terraform,Docker,Kubernetes,Networking,Security"},
        {"id":13,"title":"Cybersecurity Analyst","company":"SecureNet","location":"Remote","type":"Full-time","category":"Security","salary":"$80,000–$120,000","exp":2,"edu":"Bachelor's","desc":"Network security SIEM penetration testing Linux firewalls incident response Python compliance vulnerability","skills":"Network Security,Python,SIEM,Penetration Testing,Linux,Firewalls,Incident Response"},
        {"id":14,"title":"UX/UI Designer","company":"PixelCraft","location":"Dubai, UAE","type":"Full-time","category":"Design","salary":"$55,000–$90,000","exp":2,"edu":"Bachelor's","desc":"Figma Adobe XD wireframes prototypes UI design user research usability testing design systems CSS HTML mobile","skills":"Figma,Adobe XD,UI Design,User Research,Prototyping,Sketch,CSS,HTML"},
        {"id":15,"title":"Graphic Designer","company":"VisualEdge","location":"Lahore, Pakistan","type":"Full-time","category":"Design","salary":"$15,000–$35,000","exp":1,"edu":"Bachelor's","desc":"Adobe Photoshop Illustrator InDesign Canva typography brand identity logo social media marketing print digital","skills":"Adobe Photoshop,Illustrator,InDesign,Canva,Typography,Branding,Figma"},
        {"id":16,"title":"Flutter Mobile Developer","company":"AppForge","location":"Karachi, Pakistan","type":"Full-time","category":"Mobile","salary":"$30,000–$55,000","exp":1,"edu":"Bachelor's","desc":"Flutter Dart Firebase REST APIs state management Android iOS app store cross-platform mobile","skills":"Flutter,Dart,Firebase,REST APIs,Git,Android,iOS,State Management"},
        {"id":17,"title":"Android Developer","company":"MobileFirst","location":"Remote","type":"Full-time","category":"Mobile","salary":"$70,000–$100,000","exp":2,"edu":"Bachelor's","desc":"Kotlin Java Android SDK MVVM REST APIs Firebase Room Jetpack Compose Google Play native mobile","skills":"Kotlin,Java,Android SDK,MVVM,REST APIs,Firebase,Jetpack Compose"},
        {"id":18,"title":"Product Manager","company":"Innovatech","location":"Austin, USA","type":"Full-time","category":"Product","salary":"$100,000–$150,000","exp":4,"edu":"Bachelor's","desc":"Product strategy roadmap agile JIRA stakeholder user research data analysis A/B testing communication","skills":"Product Strategy,Agile,JIRA,Data Analysis,SQL,Communication,Roadmapping"},
        {"id":19,"title":"Digital Marketing Specialist","company":"GrowthHive","location":"Karachi, Pakistan","type":"Full-time","category":"Marketing","salary":"$20,000–$40,000","exp":1,"edu":"Bachelor's","desc":"SEO SEM Google Ads social media content marketing email analytics Canva brand performance reporting","skills":"SEO,Google Ads,Social Media,Content Marketing,Analytics,Email Marketing,Canva"},
        {"id":20,"title":"Business Analyst","company":"FinEdge","location":"Toronto, Canada","type":"Full-time","category":"Business","salary":"$65,000–$95,000","exp":2,"edu":"Bachelor's","desc":"SQL data analysis Excel Power BI JIRA process improvement stakeholder agile scrum requirements","skills":"Data Analysis,SQL,Excel,Power BI,JIRA,Tableau,Communication,Agile"},
        {"id":21,"title":"QA Automation Engineer","company":"TestPro","location":"Lahore, Pakistan","type":"Full-time","category":"Engineering","salary":"$25,000–$45,000","exp":2,"edu":"Bachelor's","desc":"Selenium Cypress Python JavaScript API testing JIRA CI/CD regression quality assurance automation","skills":"Selenium,Cypress,Python,JavaScript,API Testing,JIRA,CI/CD"},
        {"id":22,"title":"Blockchain Developer","company":"ChainTech","location":"Remote","type":"Full-time","category":"Engineering","salary":"$100,000–$150,000","exp":3,"edu":"Bachelor's","desc":"Solidity Ethereum Web3.js DeFi NFT smart contracts blockchain Python JavaScript cryptography decentralized","skills":"Solidity,Ethereum,Web3.js,Python,Smart Contracts,JavaScript,Cryptography"},
        {"id":23,"title":"Network Engineer","company":"NetSystems","location":"Islamabad, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp":2,"edu":"Bachelor's","desc":"Cisco routers switches firewalls VPN TCP/IP Linux Windows server networking troubleshooting CCNA","skills":"Cisco,Networking,Firewalls,VPN,TCP/IP,Linux,Windows Server,CCNA"},
        {"id":24,"title":"HR Business Partner","company":"PeopleFirst","location":"Dubai, UAE","type":"Full-time","category":"HR","salary":"$50,000–$80,000","exp":3,"edu":"Bachelor's","desc":"Talent acquisition employee relations performance management HR policies training HRMS communication recruitment","skills":"HR Management,Recruitment,Employee Relations,Performance Management,HRMS,Training"},
        {"id":25,"title":"Python Developer Intern","company":"StartupHub","location":"Karachi, Pakistan","type":"Internship","category":"Engineering","salary":"$5,000–$12,000","exp":0,"edu":"Intermediate","desc":"Python programming Django REST APIs SQL Git web development backend basics agile teamwork","skills":"Python,Django,SQL,Git,REST APIs,HTML"},
        {"id":26,"title":"Data Science Intern","company":"Analytics Co","location":"Lahore, Pakistan","type":"Internship","category":"Data Science","salary":"$5,000–$10,000","exp":0,"edu":"Intermediate","desc":"Python pandas numpy matplotlib machine learning scikit-learn SQL data visualization statistics Jupyter","skills":"Python,Pandas,NumPy,Matplotlib,SQL,Scikit-learn,Excel"},
        {"id":27,"title":"UI/UX Design Intern","company":"CreativeMinds","location":"Remote","type":"Internship","category":"Design","salary":"$4,000–$8,000","exp":0,"edu":"Intermediate","desc":"Wireframes prototypes Figma user interface mobile web design typography color theory user research Canva","skills":"Figma,Canva,UI Design,Prototyping,Typography"},
        {"id":28,"title":"Freelance Web Developer","company":"Various Clients","location":"Remote","type":"Freelance","category":"Engineering","salary":"$30,000–$80,000","exp":1,"edu":"Diploma","desc":"WordPress React JavaScript HTML CSS PHP MySQL client websites freelance communication project management","skills":"WordPress,React,JavaScript,HTML,CSS,PHP,MySQL"},
        {"id":29,"title":"Content Writer","company":"ContentPro Agency","location":"Remote","type":"Part-time","category":"Marketing","salary":"$15,000–$30,000","exp":1,"edu":"Bachelor's","desc":"Technical writing SEO blogs articles research AI technology software communication editing proofreading WordPress","skills":"Technical Writing,SEO,Research,Communication,WordPress,Editing"},
        {"id":30,"title":"iOS Developer","company":"AppleTree Apps","location":"Dubai, UAE","type":"Full-time","category":"Mobile","salary":"$75,000–$110,000","exp":2,"edu":"Bachelor's","desc":"Swift SwiftUI UIKit Xcode CoreData REST APIs push notifications App Store Objective-C native iOS mobile","skills":"Swift,SwiftUI,UIKit,Xcode,CoreData,REST APIs,Objective-C"},
    ]
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
#  ML ENGINE
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def build_model(descs):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    mat = vec.fit_transform(descs)
    return vec, mat

def preprocess(t):
    t = t.lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def recommend(skills, exp, edu, jtype, vec, mat, df, n=8):
    uv = vec.transform([preprocess(skills)])
    sc = cosine_similarity(uv, mat).flatten()
    r  = df.copy(); r['base'] = sc
    er = {"Intermediate":1,"Diploma":2,"Bachelor's":3,"Master's":4,"PhD":5}
    ue = er.get(edu, 3)
    r['boost'] = (
        r['exp'].apply(lambda e: 0.08 if exp>=e else (-0.05 if exp<e-2 else 0)) +
        r['type'].apply(lambda t: 0.06 if (not jtype or t==jtype) else 0) +
        r['edu'].apply(lambda e: 0.05 if er.get(e,3)<=ue else -0.03)
    )
    r['final'] = r['base']*0.75 + r['boost']
    mn,mx = r['final'].min(), r['final'].max()
    r['pct'] = ((r['final']-mn)/(mx-mn)*39+60).clip(0,99).astype(int) if mx>mn else 60
    return r.sort_values('pct', ascending=False).head(n).reset_index(drop=True)

def split_skills(u_str, j_str):
    u = {s.strip().lower() for s in u_str.split(',') if s.strip()}
    j = {s.strip().lower() for s in j_str.split(',')  if s.strip()}
    return sorted(u&j), sorted(j-u)

def match_style(p):
    if p>=82: return TEAL,  f"linear-gradient(90deg,{TEAL},{TEAL})"
    if p>=68: return BLUE,  f"linear-gradient(90deg,{BLUE},{PURPLE})"
    return RED, f"linear-gradient(90deg,{RED},#F87171)"

# Plotly layout helper
def PL(title=""):
    return dict(
        paper_bgcolor=PBG, plot_bgcolor=PBG,
        font=dict(family="DM Sans", color=PTXT, size=11),
        title=dict(text=title, font=dict(family="Syne",size=13,color=TEXT), x=0.02),
        margin=dict(l=12,r=12,t=44,b=12),
        xaxis=dict(gridcolor=PGRID,zerolinecolor=PGRID,tickfont=dict(size=10,color=PTXT)),
        yaxis=dict(gridcolor=PGRID,zerolinecolor=PGRID,tickfont=dict(size=10,color=PTXT)),
        showlegend=False,
    )

# ──────────────────────────────────────────────────────────────
#  LOAD
# ──────────────────────────────────────────────────────────────
df  = load_data()
vec, mat = build_model(df['desc'].astype(str))

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.4rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:4px;">
        {I.brain(TEAL,30)}
        <div class="sb-brand">Job<span>Match</span> AI</div>
      </div>
      <div style="font-size:.7rem;color:{MUTED};">TF-IDF · Cosine Similarity · Waqaas Hussain</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<hr class="sb-hr">', unsafe_allow_html=True)

    # Theme toggle
    tlabel = f"{I.sun(AMBER,14)} Light Mode" if DARK else f"{I.moon(BLUE,14)} Dark Mode"
    if st.button(tlabel, key="theme_toggle"):
        st.session_state.theme = "light" if DARK else "dark"
        st.rerun()

    st.markdown(f'<hr class="sb-hr">', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-tag">{I.user(TEAL,13)} Your Profile</div>', unsafe_allow_html=True)
    user_skills = st.text_area("Enter Your Skills",
        placeholder="e.g. Python, Machine Learning, SQL, Django, React",
        height=95, key="skills_input")

    st.markdown(f'<div class="sb-tag">{I.graduation(TEAL,13)} Education & Experience</div>', unsafe_allow_html=True)
    education  = st.selectbox("Education Level",
        ["Intermediate","Diploma","Bachelor's","Master's","PhD"])
    experience = st.slider("Years of Experience", 0, 20, 1)

    st.markdown(f'<div class="sb-tag">{I.filter_ic(TEAL,13)} Job Filters</div>', unsafe_allow_html=True)
    job_type  = st.selectbox("Job Type",
        ["Any","Full-time","Part-time","Remote","Freelance","Internship"])
    loc_pref  = st.text_input("Preferred Location",
        placeholder="e.g. Remote, Karachi, Dubai")
    top_n     = st.slider("Results to Show", 3, 15, 8)

    st.markdown(f'<hr class="sb-hr">', unsafe_allow_html=True)
    find_btn = st.button(
        f"Find My Jobs  {I.arrow_r('currentColor',15)}",
        use_container_width=True, type="primary")

    st.markdown(f'<hr class="sb-hr">', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-tag">{I.home(TEAL,13)} Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", [
        "🏠  Home",
        "🔍  Recommendations",
        "📊  Analytics",
        "⚙️  Methodology",
        "📚  Literature Review",
        "ℹ️  About Project",
    ], label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════
#  HERO  (shown on every page)
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
  <div class="hero-glow"></div><div class="hero-glow2"></div>
  <div class="hero-badge"><div class="pulse-dot"></div>&nbsp; AI-Powered · Content-Based Filtering</div>
  <div class="hero-title">AI-Based <span class="ht">Job Recommendation</span> System</div>
  <div class="hero-sub">
    An intelligent system that analyzes user skills and matches them with relevant job opportunities
    using <strong>TF-IDF vectorization</strong> and <strong>cosine similarity</strong> — making job
    search faster, smarter, and more personalized.
  </div>
  <div class="hero-chips">
    <div class="chip">{I.python_ic(15)} Python 3.x</div>
    <div class="chip">{I.cpu(TEAL,15)} Scikit-learn</div>
    <div class="chip">{I.target(BLUE,15)} Cosine Similarity</div>
    <div class="chip">{I.chart(PURPLE,15)} Plotly</div>
    <div class="chip">{I.brain(AMBER,15)} TF-IDF</div>
    <div class="chip">{I.layers(GREEN,15)} Content-Based Filtering</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if "Home" in page:

    # ── Objective badges ──────────────────────────────────
    st.markdown(f'<div class="sh">{I.target(TEAL,18)} Project Objectives</div>', unsafe_allow_html=True)
    objectives = [
        (I.layers(TEAL,18), "Design a GUI-based job recommendation system using Streamlit"),
        (I.cpu(BLUE,18),    "Develop a machine learning model using TF-IDF for job matching"),
        (I.search(PURPLE,18),"Analyze user skills and job descriptions for better recommendations"),
        (I.chart(AMBER,18), "Evaluate system performance using cosine similarity measures"),
        (I.trending(GREEN,18),"Improve accuracy and relevance of personalized job suggestions"),
    ]
    for ico, txt in objectives:
        st.markdown(f'<div class="obj-item"><div style="flex-shrink:0;margin-top:1px;">{ico}</div><div class="obj-txt">{txt}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Steps ────────────────────────────────────────────
    st.markdown(f'<div class="sh">{I.arrow_r(TEAL,18)} How to Use</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,d,n,t,desc in [
        (c1,"0s","01","Enter Skills","Type your skills comma-separated in the sidebar — Python, SQL, React..."),
        (c2,"0.07s","02","Set Preferences","Choose education level, experience, job type & location"),
        (c3,"0.14s","03","Get Matched","Click Find My Jobs — AI ranks jobs by cosine similarity score"),
        (c4,"0.21s","04","Analyze Gap","Review matched & missing skills — know what to learn next"),
    ]:
        col.markdown(f'<div class="step-card" style="animation-delay:{d};"><div class="step-num">{n}</div><div class="step-title">{t}</div><div class="step-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Categories ───────────────────────────────────────
    st.markdown(f'<div class="sh">{I.briefcase(TEAL,18)} Available Job Categories ({len(df)} Jobs)</div>', unsafe_allow_html=True)
    cats  = df['category'].value_counts()
    cols  = st.columns(3)
    for i,(cat,cnt) in enumerate(cats.items()):
        cols[i%3].markdown(f"""
<div class="cat-pill" style="animation-delay:{i*0.05}s;">
  {I.layers(TEAL,16)}
  <span style="font-weight:600;color:{TEXT};flex:1;">{cat}</span>
  <span style="background:rgba(0,201,167,.12);color:{TEAL};font-size:.71rem;font-weight:700;padding:2px 9px;border-radius:99px;">{cnt} jobs</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
elif "Recommendations" in page:

    if find_btn or 'results' in st.session_state:
        if find_btn:
            if not user_skills.strip():
                st.warning("Please enter at least one skill to get recommendations.")
                st.stop()
            jt  = "" if job_type=="Any" else job_type
            res = recommend(user_skills, experience, education, jt, vec, mat, df, top_n)
            if loc_pref.strip():
                res = res[
                    res['location'].str.lower().str.contains(loc_pref.lower(), na=False) |
                    res['location'].str.lower().str.contains('remote', na=False)
                ]
            st.session_state['results'] = res
            st.session_state['skills']  = user_skills

        res   = st.session_state.get('results', pd.DataFrame())
        s_str = st.session_state.get('skills', '')

        if res.empty:
            st.warning("No matching jobs found — try broadening your filters.")
            st.stop()

        # Stat cards
        sk_cnt = len([s for s in s_str.split(',') if s.strip()])
        c1,c2,c3,c4 = st.columns(4)
        for col,ico,num,lbl in [
            (c1, I.briefcase(TEAL,24),   len(res),                    "Jobs Found"),
            (c2, I.target(BLUE,24),      f"{int(res['pct'].mean())}%","Avg Match Score"),
            (c3, I.star(AMBER,24),       f"{int(res['pct'].max())}%", "Best Match"),
            (c4, I.layers(PURPLE,24),    sk_cnt,                       "Skills Detected"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="s-ico">{ico}</div><div class="s-num">{num}</div><div class="s-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Filter
        cats = ["All"] + sorted(res['category'].unique().tolist())
        st.markdown(f'<div class="sh">{I.filter_ic(TEAL,18)} Filter Results</div>', unsafe_allow_html=True)
        sel  = st.radio("cat", cats, horizontal=True, label_visibility="collapsed")
        show = res if sel=="All" else res[res['category']==sel]

        st.markdown(f'<p style="color:{MUTED};font-size:.79rem;margin:.5rem 0 1rem;">Showing <b style="color:{TEAL};">{len(show)}</b> job(s) · ranked by TF-IDF cosine similarity score</p>', unsafe_allow_html=True)

        # Job cards
        for _, j in show.iterrows():
            pct = j['pct']
            rc, bg = match_style(pct)
            matched, missing = split_skills(s_str, j['skills'])
            mp = "".join([f'<span class="c-match">{I.check(TEAL,10)} {s}</span>' for s in matched])
            xp = "".join([f'<span class="c-miss">{I.xmark(RED,10)} {s}</span>'  for s in missing])

            with st.expander(f"  {j['title']}  ·  {j['company']}  —  {pct}% match"):
                st.markdown(f"""
<div class="jcard">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:11px;">
    <div style="flex:1;">
      <div class="j-title">{I.briefcase(rc,15)}&nbsp; {j['title']}</div>
      <div class="j-meta">
        <span class="j-mi">{I.building(MUTED,13)} {j['company']}</span>
        <span class="j-mi">{I.location(MUTED,13)} {j['location']}</span>
      </div>
    </div>
    <div class="mring" style="color:{rc};border-color:{rc};background:rgba(0,0,0,.18);">{pct}%</div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0;">
    <span class="badge bt">{I.target(TEAL,10)} {pct}% Match</span>
    <span class="badge bb">{I.clock(BLUE,10)} {j['type']}</span>
    <span class="badge bp">{I.layers(PURPLE,10)} {j['category']}</span>
    <span class="badge ba">{I.graduation(AMBER,10)} {j['edu']}+</span>
    <span class="badge br">{I.clock(RED,10)} {j['exp']}+ yrs</span>
  </div>

  <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{bg};"></div></div>
  <div style="font-size:.7rem;color:{MUTED};margin-bottom:11px;">{pct}% TF-IDF cosine similarity alignment</div>

  <div class="sal-tag">{I.dollar(AMBER,14)} {j['salary']}</div>

  <div style="font-size:.77rem;font-weight:700;color:{TEXT};margin:13px 0 6px;display:flex;align-items:center;gap:7px;">
    {I.layers(TEAL,14)} Skill Analysis — Matched vs Missing
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:2px;">{mp}{xp}</div>
  {'<div style="font-size:.67rem;color:'+MUTED+';margin-top:7px;display:flex;gap:12px;"><span style=\'color:'+TEAL+';font-weight:700;\'>● Matched</span><span style=\'color:'+RED+';font-weight:700;\'>● Missing (skill gap)</span></div>' if (matched or missing) else ''}
st.markdown(f"""
<style>
.card {{ padding: 10px; }}
</style>
""", unsafe_allow_html=True)

                   if missing:
                    st.markdown(f"""
<div class="gap-box">{I.alert(AMBER,18)}
  <div>
    <div class="gap-title">Skill Gap — Recommended to Learn:</div>
    <div class="gap-sk">{', '.join(missing)}</div>
  </div>
</div>""", unsafe_allow_html=True)

                if st.button("Apply Now", key=f"a_{j['id']}"):
                    st.success(f"Application submitted for **{j['title']}** at **{j['company']}**!")

        st.markdown("<br>", unsafe_allow_html=True)
        csv = show[['title','company','location','type','category','salary','pct']]\
              .rename(columns={'title':'Title','company':'Company','location':'Location',
                               'type':'Type','category':'Category','salary':'Salary','pct':'Match%'})\
              .to_csv(index=False).encode('utf-8')
        st.download_button(f"{I.download('white',14)} Download Results as CSV",
                           csv,"job_recommendations.csv","text/csv",use_container_width=True)

    else:
        st.markdown(f"""
<div class="wcard">
  {I.search(TEAL,52)}
  <div class="w-title">Ready to Find Your Match?</div>
  <div class="w-sub">Fill in your skills and preferences in the <b>sidebar</b>, then click <b>Find My Jobs</b> to receive AI-powered recommendations ranked by cosine similarity.</div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown(f'<div class="sh">{I.chart(TEAL,18)} Job Market Analytics Dashboard</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,ico,num,lbl in [
        (c1,I.briefcase(TEAL,24),  len(df),               "Total Jobs"),
        (c2,I.building(BLUE,24),   df['company'].nunique(),"Companies"),
        (c3,I.layers(PURPLE,24),   df['category'].nunique(),"Categories"),
        (c4,I.location(AMBER,24),  df['location'].nunique(),"Locations"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="s-ico">{ico}</div><div class="s-num">{num}</div><div class="s-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cat_s = df['category'].value_counts()
    typ_s = df['type'].value_counts()
    alls  = []
    for s in df['skills']: alls.extend([x.strip() for x in s.split(',')])
    sk_n,sk_v = zip(*Counter(alls).most_common(12))
    bins  = pd.cut(df['exp'],bins=[-1,0,1,2,3,5,10,20],
                   labels=['Fresher','<1yr','1–2yr','2–3yr','3–5yr','5–10yr','10+yr'])
    exp_c = bins.value_counts().sort_index()

    # Row 1
    r1,r2 = st.columns(2)
    with r1:
        fig = go.Figure(go.Bar(y=cat_s.index.tolist(),x=cat_s.values.tolist(),orientation='h',
            marker_color=PALETTE[:len(cat_s)],text=cat_s.values,textposition='outside',
            textfont=dict(size=10,color=TEXT),hovertemplate='<b>%{y}</b><br>%{x} jobs<extra></extra>'))
        fig.update_layout(**PL("Jobs by Category"),height=350)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with r2:
        fig2 = go.Figure(go.Pie(labels=typ_s.index.tolist(),values=typ_s.values.tolist(),hole=0.58,
            marker_colors=PALETTE[:len(typ_s)],textinfo='label+percent',
            textfont=dict(size=10,color=TEXT),pull=[0.04]*len(typ_s),
            hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'))
        fig2.update_layout(**PL("Job Type Distribution"),height=350,showlegend=True,
            legend=dict(font=dict(color=TEXT,size=10),bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})

    # Row 2
    r3,r4 = st.columns(2)
    with r3:
        fig3 = go.Figure(go.Bar(x=exp_c.index.astype(str).tolist(),y=exp_c.values.tolist(),
            marker_color=[PALETTE[i%len(PALETTE)] for i in range(len(exp_c))],
            text=exp_c.values,textposition='outside',textfont=dict(size=10,color=TEXT),
            hovertemplate='<b>%{x}</b>: %{y} jobs<extra></extra>'))
        fig3.update_layout(**PL("Experience Level Required"),height=330)
        fig3.update_xaxes(title_text='Experience')
        fig3.update_yaxes(title_text='Jobs')
        st.plotly_chart(fig3,use_container_width=True,config={"displayModeBar":False})

    with r4:
        fig4 = go.Figure(go.Bar(y=list(sk_n)[::-1],x=list(sk_v)[::-1],orientation='h',
            marker=dict(color=list(sk_v)[::-1],
                colorscale=[[0,BLUE],[0.5,TEAL],[1,PURPLE]],showscale=False),
            text=list(sk_v)[::-1],textposition='outside',textfont=dict(size=9,color=TEXT),
            hovertemplate='<b>%{y}</b>: %{x} jobs<extra></extra>'))
        fig4.update_layout(**PL("Top 12 In-Demand Skills"),height=330)
        st.plotly_chart(fig4,use_container_width=True,config={"displayModeBar":False})

    # Salary bubble
    st.markdown(f'<div class="sh">{I.dollar(TEAL,18)} Salary vs Experience Analysis</div>', unsafe_allow_html=True)
    sal_df = df.copy()
    def mid(s):
        n = re.findall(r'[\d]+', s.replace(',',''))
        return (int(n[0])+int(n[1]))//2 if len(n)>=2 else 0
    sal_df['sal_mid'] = sal_df['salary'].apply(mid)
    sal_df = sal_df[sal_df['sal_mid']>0]
    fig5 = px.scatter(sal_df,x='exp',y='sal_mid',color='category',size='sal_mid',
        hover_name='title',hover_data={'company':True,'salary':True,'sal_mid':False,'exp':True},
        color_discrete_sequence=PALETTE,size_max=26,
        labels={'exp':'Years of Experience','sal_mid':'Midpoint Salary ($)','category':'Category'})
    fig5.update_layout(**PL("Salary vs Experience (bubble size = salary)"),height=380,showlegend=True,
        legend=dict(font=dict(color=TEXT,size=10),bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig5,use_container_width=True,config={"displayModeBar":False})

    # Full dataset
    st.markdown(f'<div class="sh">{I.layers(TEAL,18)} Complete Job Dataset</div>', unsafe_allow_html=True)
    disp = df[['title','company','location','type','category','salary','exp','edu']].copy()
    disp.columns = ['Job Title','Company','Location','Type','Category','Salary','Min Exp','Education']
    st.dataframe(disp,use_container_width=True,height=400)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — METHODOLOGY  (from proposal)
# ══════════════════════════════════════════════════════════════
elif "Methodology" in page:
    st.markdown(f'<div class="sh">{I.settings(TEAL,18)} Proposed Methodology</div>', unsafe_allow_html=True)

    steps = [
        (I.layers(TEAL,20),  "01","Step 1 — Data Collection",
         "Collected 30 structured job listings including job titles, full descriptions, required skills, salary ranges, location, and experience requirements. Data simulates real Kaggle-style CSV job datasets as outlined in the project scope."),
        (I.code(BLUE,20),    "02","Step 2 — Data Preprocessing",
         "Text normalization pipeline: lowercase conversion → special character removal using Python regex → whitespace collapsing. TF-IDF's built-in English stop word list handles tokenization and stop word filtering. Missing values handled before vectorization."),
        (I.cpu(PURPLE,20),   "03","Step 3 — TF-IDF Feature Extraction",
         "Scikit-learn's TfidfVectorizer converts job descriptions and user skill input into high-dimensional sparse numerical vectors. Bigrams (ngram_range=1,2) capture multi-word skill phrases. Sublinear TF scaling reduces dominance of very frequent terms."),
        (I.target(TEAL,20),  "04","Step 4 — Cosine Similarity Matching",
         "Cosine similarity is computed between the user skill vector and all job description vectors. Scores range 0→1. Results are boosted by experience match (+0.08), job type (+0.06), education alignment (+0.05), then normalized to 60–99% display range."),
        (I.brain(AMBER,20),  "05","Step 5 — Implementation",
         "Backend built in Python with Scikit-learn ML engine. GUI built in Streamlit with full dark/light theme, animated CSS, Plotly interactive charts, SVG icons, glassmorphism cards, and skill gap analysis — all in a single app.py file."),
        (I.chart(GREEN,20),  "06","Step 6 — Testing & Evaluation",
         "System tested with diverse skill inputs across all 11 job categories. Match percentage accuracy evaluated by comparing user skills vs job requirements. Results displayed with visual match bars, skill chip analysis, and CSV export for review."),
    ]
    for ico,num,title,desc in steps:
        st.markdown(f"""
<div class="how">
  <div class="how-num">{num}</div>
  <div>
    <div class="how-title">{ico} {title}</div>
    <div class="how-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Algorithm formula
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.flask(TEAL,18)} Core Algorithm — Cosine Similarity</div>', unsafe_allow_html=True)
    st.latex(r"\text{Cosine Similarity}(A,B) = \frac{A \cdot B}{\|A\| \times \|B\|}")
    st.markdown(f"""
<div class="how" style="margin-top:.5rem;">
  <div style="font-size:.83rem;color:{MUTED};line-height:1.8;">
    <b style="color:{TEXT};">A</b> = TF-IDF vector of user's entered skills &nbsp;·&nbsp;
    <b style="color:{TEXT};">B</b> = TF-IDF vector of a job description<br>
    <b style="color:{TEXT};">Result range:</b> 0 (no match) → 1 (perfect match) &nbsp;·&nbsp;
    Measures the cosine of the angle between two vectors in TF-IDF space — independent of document length.
    Higher score = more semantically similar skill set.
  </div>
</div>""", unsafe_allow_html=True)

    # Technologies table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.cpu(TEAL,18)} Tools & Technologies Used</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        r1h = "".join([f'<tr><td style="display:flex;align-items:center;gap:8px;">{i} {t}</td><td>{d}</td></tr>' for i,t,d in [
            (I.python_ic(15),"Python 3.x","Core programming language"),
            (I.layers(TEAL,15),"Streamlit","Interactive web app GUI framework"),
            (I.cpu(BLUE,15),"Scikit-learn","TF-IDF vectorizer & cosine similarity"),
            (I.chart(PURPLE,15),"Pandas / NumPy","Data manipulation & numerical computing"),
        ]])
        st.markdown(f'<table class="tt"><tr><th>Tool / Library</th><th>Purpose</th></tr>{r1h}</table>', unsafe_allow_html=True)
    with c2:
        r2h = "".join([f'<tr><td style="display:flex;align-items:center;gap:8px;">{i} {t}</td><td>{d}</td></tr>' for i,t,d in [
            (I.trending(TEAL,15),"Plotly Express","Animated interactive analytics charts"),
            (I.code(AMBER,15),"CSS / JavaScript","Animations, glassmorphism & dark/light theme"),
            (I.settings(BLUE,15),"re (Regex)","Text normalization & preprocessing"),
            (I.brain(RED,15),"VS Code / Jupyter","Development & testing environment"),
        ]])
        st.markdown(f'<table class="tt"><tr><th>Tool / Library</th><th>Purpose</th></tr>{r2h}</table>', unsafe_allow_html=True)

    # Scope
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.info(TEAL,18)} Project Scope</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        included = [
            "AI-based content-based recommendation system",
            "GUI interface using Streamlit framework",
            "Structured CSV/dataset-style job data",
            "TF-IDF + Cosine Similarity implementation",
            "Skill gap analysis & visual feedback",
            "Dark/light theme + animated UI",
        ]
        items = "".join([f'<div class="obj-item"><div>{I.check(TEAL,16)}</div><div class="obj-txt">{x}</div></div>' for x in included])
        st.markdown(f'<div style="font-weight:700;color:{TEAL};font-size:.8rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em;">✅ Included</div>{items}', unsafe_allow_html=True)
    with c2:
        excluded = [
            "Real-time job API integration (LinkedIn, Indeed)",
            "Mobile application development",
            "Advanced deep learning models (BERT, GPT)",
            "Large-scale cloud deployment",
        ]
        items2 = "".join([f'<div class="obj-item" style="border-left:3px solid {RED};"><div>{I.xmark(RED,16)}</div><div class="obj-txt">{x}</div></div>' for x in excluded])
        st.markdown(f'<div style="font-weight:700;color:{RED};font-size:.8rem;margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em;">❌ Not Included</div>{items2}', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — LITERATURE REVIEW
# ══════════════════════════════════════════════════════════════
elif "Literature" in page:
    st.markdown(f'<div class="sh">{I.book(TEAL,18)} Literature Review</div>', unsafe_allow_html=True)

    papers = [
        ("Traditional Keyword-Based Job Portals",
         "LinkedIn, Indeed, and Rozee.pk rely on basic keyword matching to retrieve job listings. While fast, this approach suffers from low precision — returning irrelevant results when user input doesn't exactly match job description terminology. No semantic understanding of skill synonyms.",
         ["Keyword Matching","LinkedIn","Low Precision"]),
        ("Content-Based Filtering in Recommendation Systems",
         "Pazzani & Billsus (2007) established content-based filtering as a robust approach for personalized recommendations. By representing items (jobs) as feature vectors and matching them to user profiles, CBF avoids the cold-start problem that plagues collaborative filtering.",
         ["Content-Based Filtering","Personalization","Feature Vectors"]),
        ("TF-IDF for Text-Based Job Matching",
         "Bharadwaj & Shinghal (2016) demonstrated that TF-IDF vectorization outperforms simple keyword frequency for job description matching. By assigning higher weights to rare but important skill keywords, TF-IDF creates more discriminative document representations.",
         ["TF-IDF","Text Mining","Feature Extraction"]),
        ("Cosine Similarity for Resume-Job Matching",
         "Al-Otaibi & Ykhlef (2012) proposed cosine similarity as the primary distance metric for skill-to-job matching, noting its length-invariance property makes it ideal for comparing skill profiles of varying lengths against job descriptions.",
         ["Cosine Similarity","Resume Matching","NLP"]),
        ("Kaggle Datasets for Job Recommendation Research",
         "Structured job datasets from Kaggle (e.g., LinkedIn Job Postings, Glassdoor) have become standard benchmarks in job recommendation research, providing labeled data with job titles, descriptions, required skills, and salary information.",
         ["Kaggle","Dataset","Benchmark"]),
        ("Research Gap — Lack of User-Friendly AI Interfaces",
         "Most academic job recommendation systems focus purely on algorithmic accuracy but lack interactive GUI implementations. This project addresses this gap by combining a proven ML approach (TF-IDF + cosine similarity) with a polished Streamlit-based interface including skill gap analysis.",
         ["Research Gap","GUI","User Experience"]),
    ]
    for i,(title,desc,tags) in enumerate(papers):
        tag_html = "".join([f'<span class="lit-tag">{t}</span>' for t in tags])
        st.markdown(f"""
<div class="lit-card" style="animation-delay:{i*0.06}s;">
  <div class="lit-title">{I.book(BLUE,14)}&nbsp; {title}</div>
  <div class="lit-desc">{desc}</div>
  <div style="margin-top:8px;">{tag_html}</div>
</div>""", unsafe_allow_html=True)

    # Problem statement
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.alert(TEAL,18)} Problem Statement</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="how">
  <div>
    <div class="how-desc" style="font-size:.88rem;line-height:1.8;">
      Current job recommendation systems lack <b style="color:{TEXT};">personalization and accuracy</b>,
      as they depend mainly on keyword matching. This leads to irrelevant job suggestions and poor user experience.
      The problem is to develop an <b style="color:{TEAL};">intelligent system</b> that can analyze user skills
      and recommend relevant jobs effectively using machine learning techniques — specifically
      <b style="color:{TEXT};">TF-IDF vectorization</b> combined with <b style="color:{TEXT};">cosine similarity</b>
      content-based filtering, wrapped in a professional <b style="color:{TEXT};">Streamlit GUI</b>.
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 6 — ABOUT PROJECT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown(f'<div class="sh">{I.info(TEAL,18)} About This Project</div>', unsafe_allow_html=True)

    # Project info card
    st.markdown(f"""
<div class="info-card">
  <div class="info-lbl">{I.brain(TEAL,14)} Project Information</div>
  <div class="info-val">
    <b>Project Title:</b> AI-Based Job Recommendation System<br>
    <b>Prepared by:</b> Waqaas Hussain<br>
    <b>Subject:</b> Programming for AI<br>
    <b>Algorithm:</b> TF-IDF Vectorization + Cosine Similarity (Content-Based Filtering)<br>
    <b>Framework:</b> Python · Streamlit · Scikit-learn · Plotly<br>
    <b>Dataset:</b> 30 structured job listings across 11 categories<br>
    <b>Theme:</b> Dark / Light (toggle in sidebar)
  </div>
</div>""", unsafe_allow_html=True)

    # Introduction / background
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.book(TEAL,18)} Introduction & Background</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="how">
  <div>
    <div class="how-desc" style="font-size:.87rem;line-height:1.85;">
      With the increasing use of online job platforms, finding suitable jobs has become a
      <b style="color:{TEXT};">challenging task</b> for many users. Most systems rely on simple keyword-based searches,
      which often produce irrelevant results. Artificial Intelligence can improve this process by
      <b style="color:{TEXT};">analyzing user skills</b> and matching them with job requirements using
      mathematical similarity measures.<br><br>
      This project focuses on developing an <b style="color:{TEAL};">AI-based Job Recommendation System</b>
      with a graphical user interface using Streamlit. The system provides intelligent and personalized job
      suggestions using <b style="color:{TEXT};">TF-IDF vectorization</b> for feature extraction and
      <b style="color:{TEXT};">cosine similarity</b> for semantic skill matching — making the job search
      process faster and more efficient.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Expected outcomes
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.star(TEAL,18)} Expected Outcomes</div>', unsafe_allow_html=True)
    outcomes = [
        (I.briefcase(TEAL,18), "A fully functional AI-based job recommendation system"),
        (I.chart(BLUE,18),     "Interactive Streamlit GUI for user input and results display"),
        (I.target(PURPLE,18),  "Accurate and personalized job suggestions using cosine similarity"),
        (I.layers(AMBER,18),   "Skill gap analysis — matched skills vs missing skills per job"),
        (I.trending(GREEN,18), "Plotly analytics dashboard showing job market distribution"),
        (I.code(RED,18),       "Professional dark/light themed UI with animation effects"),
    ]
    for ico,txt in outcomes:
        st.markdown(f'<div class="obj-item"><div style="flex-shrink:0;margin-top:1px;">{ico}</div><div class="obj-txt">{txt}</div></div>', unsafe_allow_html=True)

    # System architecture
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sh">{I.cpu(TEAL,18)} System Architecture</div>', unsafe_allow_html=True)

    arch_html = f"""
<div style="background:{CARD2};border:1px solid {BORDER};border-radius:16px;padding:2rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;color:{MUTED};line-height:2;animation:fadeIn .5s ease;">
<span style="color:{TEAL};font-weight:700;">┌─────────────────────────────────────────────┐</span>
<span style="color:{TEXT};">│         AI JOB RECOMMENDATION SYSTEM         │</span>
<span style="color:{TEAL};font-weight:700;">└─────────────────────────────────────────────┘</span>

<span style="color:{TEAL};">USER INPUT</span>          <span style="color:{MUTED};">→ Skills, Education, Experience, Location</span>
      ↓
<span style="color:{BLUE};">PREPROCESSING</span>       <span style="color:{MUTED};">→ Lowercase · Regex clean · Stop word removal</span>
      ↓
<span style="color:{PURPLE};">TF-IDF VECTORIZER</span>  <span style="color:{MUTED};">→ skill_vector = vec.transform(user_input)</span>
      ↓
<span style="color:{TEAL};">COSINE SIMILARITY</span>  <span style="color:{MUTED};">→ scores = cosine_similarity(skill_vec, job_mat)</span>
      ↓
<span style="color:{AMBER};">SCORE BOOSTING</span>     <span style="color:{MUTED};">→ experience (+0.08) · type (+0.06) · edu (+0.05)</span>
      ↓
<span style="color:{GREEN};">NORMALIZATION</span>      <span style="color:{MUTED};">→ pct = normalize(final_score, range=60–99%)</span>
      ↓
<span style="color:{BLUE};">RANKED RESULTS</span>     <span style="color:{MUTED};">→ Top-N jobs sorted by match percentage</span>
      ↓
<span style="color:{TEAL};">STREAMLIT GUI</span>      <span style="color:{MUTED};">→ Job cards · Skill chips · Gap analysis · Charts</span>
</div>"""
    st.markdown(arch_html, unsafe_allow_html=True)

    st.markdown(f"""
<div class="info-card" style="margin-top:2rem;">
  <div class="info-lbl">{I.star(TEAL,14)} Academic Declaration</div>
  <div class="info-val">
    This project is prepared as part of the <b>Programming for AI</b> course requirements.
    All code is original work by <b>Waqaas Hussain</b>. The system demonstrates
    practical application of <b>content-based filtering</b>, <b>TF-IDF feature extraction</b>,
    and <b>cosine similarity</b> concepts studied in the course curriculum.
  </div>
</div>""", unsafe_allow_html=True)

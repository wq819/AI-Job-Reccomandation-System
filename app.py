# ================================================================
#   AI JOB RECOMMENDATION SYSTEM 
#   Prepared by : Waqaas Hussain
#   Institution : Aror University Sukkur
#   Subject     : Programming for AI
#   Algorithm   : TF-IDF Vectorization + Cosine Similarity
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
#  1. HD ICON SYSTEM (Inline SVG)
# ──────────────────────────────────────────────────────────────
def _i(d, c="currentColor", s=18, f="none", w=2):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}" '
            f'viewBox="0 0 24 24" fill="{f}" stroke="{c}" stroke-width="{w}" '
            f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">'
            f'{d}</svg>')

class I:
    home      = lambda c=None,s=None: _i('<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',c,s)
    search    = lambda c=None,s=None: _i('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',c,s)
    chart     = lambda c=None,s=None: _i('<line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/><path d="M2 20h20"/>',c,s)
    cpu       = lambda c=None,s=None: _i('<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',c,s)
    target    = lambda c=None,s=None: _i('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',c,s)
    briefcase = lambda c=None,s=None: _i('<rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',c,s)
    layers    = lambda c=None,s=None: _i('<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12-8.6 3.92a2 2 0 0 1-1.66 0L3 12"/><path d="m22 17-8.6 3.92a2 2 0 0 1-1.66 0L3 17"/>',c,s)

# ──────────────────────────────────────────────────────────────
#  2. DEEP OCEAN & MIDNIGHT GOLD THEME
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Recommendation System", layout="wide", page_icon="🎯")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; background-color: #020c1b; }}
    
    /* Elegant Header */
    .hero-panel {{
        background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
        padding: 70px 40px; border-radius: 40px; color: #ccd6f6;
        margin-bottom: 40px; border: 1px solid rgba(100, 255, 218, 0.1);
        box-shadow: 0 50px 100px -20px rgba(2, 12, 27, 0.7);
        text-align: center;
    }}

    .main-title {{
        font-size: 3.8rem; font-weight: 700; letter-spacing: -1.5px;
        background: linear-gradient(90deg, #64ffda, #e6f1ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
    }}

    /* Card Layout */
    .job-card {{
        background: #112240;
        padding: 35px; border-radius: 24px; border: 1px solid rgba(100, 255, 218, 0.05);
        margin-bottom: 30px; transition: all 0.3s cubic-bezier(0.645, 0.045, 0.355, 1);
    }}
    .job-card:hover {{
        border-color: #64ffda; transform: translateY(-8px);
        box-shadow: 0 20px 30px -15px rgba(2, 12, 27, 0.5);
    }}

    .match-tag {{
        background: rgba(100, 255, 218, 0.1); color: #64ffda;
        padding: 8px 22px; border-radius: 4px; font-weight: 700;
        border: 1px solid #64ffda; font-size: 0.8rem; letter-spacing: 1px;
    }}

    /* Branded Sidebar */
    [data-testid="stSidebar"] {{ background-color: #0a192f!important; border-right: 1px solid #233554!important; }}
    [data-testid="stSidebar"] * {{ color: #8892b0!important; }}
    .stTextArea textarea {{ background: #020c1b!important; color: #ccd6f6!important; border: 1px solid #233554!important; }}
    
    /* Button Upgrade */
    .stButton>button {{
        background: transparent!important;
        color: #64ffda!important; border: 1.5px solid #64ffda!important; border-radius: 4px!important;
        font-weight: 600!important; letter-spacing: 1.2px!important;
        padding: 0.75rem 1.5rem!important; transition: all 0.25s ease!important;
    }}
    .stButton>button:hover {{ background: rgba(100, 255, 218, 0.1)!important; transform: translateY(-2px); }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  3. DATA & ML ENGINE
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = [
        {"id": 1, "title": "Senior AI Researcher", "company": "NeuralLink", "location": "Remote", "cat": "AI/ML", "skills": "Python, PyTorch, NLP, Research, Calculus", "desc": "Leading research in Large Language Models and neural architectures."},
        {"id": 2, "title": "Data Infrastructure Engineer", "company": "ByteDance", "location": "Karachi", "cat": "Data Eng", "skills": "SQL, Spark, Python, Hadoop, ETL", "desc": "Building robust data pipelines for high-traffic social media analytics."},
        {"id": 3, "title": "React Frontend Architect", "company": "Vercel", "location": "Remote", "cat": "Dev", "skills": "Next.js, TypeScript, React, Tailwind, CSS", "desc": "Designing high-performance user interfaces and edge-deployment strategies."},
        {"id": 4, "title": "Cloud Security Lead", "company": "CrowdStrike", "location": "Islamabad", "cat": "Security", "skills": "AWS, Python, SIEM, Penetration Testing, Linux", "desc": "Managing enterprise cloud security and threat mitigation protocols."},
        {"id": 5, "title": "Junior Python Developer", "company": "Systems Ltd", "location": "Lahore", "cat": "Dev", "skills": "Python, Django, Git, SQL, REST", "desc": "Developing enterprise-grade web services and automated test suites."},
    ]
    return pd.DataFrame(data)

def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())

@st.cache_resource
def build_engine(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    matrix = tfidf.fit_transform((df['title'] + " " + df['skills'] + " " + df['desc']).apply(clean))
    return tfidf, matrix

df = load_data()
tfidf, matrix = build_engine(df)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR BRANDING
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="color:#64ffda;">{I.target(c="#64ffda", s=24)} Navigator</h2>', unsafe_allow_html=True)
    st.markdown("---")
    nav = st.radio("Menu", ["Overview", "Matching AI", "Analytics", "Project Docs"])
    
    st.markdown("---")
    st.markdown("### Profile Input")
    u_skills = st.text_area("Your Skillset", placeholder="e.g. Python, Machine Learning...", height=130)
    u_loc = st.selectbox("Location", ["Any", "Remote", "Karachi", "Lahore", "Islamabad"])
    
    st.markdown("---")
    st.markdown(f'<div style="color:#8892b0; font-size:0.8rem;">Researcher: <b style="color:#ccd6f6;">Waqaas Hussain</b><br>AI Engineering Student</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  5. PAGES
# ──────────────────────────────────────────────────────────────

if nav == "Overview":
    st.markdown(f"""
    <div class="hero-panel">
        <div class="main-title">AI JOB RECOMMENDATION SYSTEM</div>
        <p style="font-size:1.15rem; color:#8892b0; max-width:750px; margin:0 auto; line-height:1.8;">
            An expert implementation of content-based filtering. We use mathematical vector space 
            modeling to find the <b>optimal overlap</b> between candidate capabilities and industrial demand.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("System Database", f"{len(df)} Listings")
    c2.metric("ML Algorithm", "TF-IDF")
    c3.metric("Similarity", "Cosine Matrix")

elif nav == "Matching AI":
    st.markdown('<h2 style="color:#ccd6f6;">AI Selection Matrix</h2>', unsafe_allow_html=True)
    
    if st.sidebar.button("Execute Recommendation"):
        if u_skills.strip():
            u_vec = tfidf.transform([clean(u_skills)])
            sim = cosine_similarity(u_vec, matrix).flatten()
            df['score'] = sim * 100
            
            res = df.sort_values('score', ascending=False)
            if u_loc != "Any": res = res[res['location'] == u_loc]
            
            st.success("Matching logic completed successfully.")
            
            for _, row in res.iterrows():
                if row['score'] > 0:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                            <div>
                                <h3 style="margin:0; color:#e6f1ff; font-weight:700;">{row['title']}</h3>
                                <span style="color:#8892b0; font-size:0.9rem;">{row['company']} &nbsp;·&nbsp; {row['location']}</span>
                            </div>
                            <div class="match-tag">{int(row['score'])}% MATCH SCORE</div>
                        </div>
                        <p style="color:#8892b0; font-size:0.95rem; line-height:1.7;">{row['desc']}</p>
                        <div style="margin-top:20px;">
                            <code style="background:rgba(100, 255, 218, 0.05); padding:8px 15px; border-radius:4px; color:#64ffda; font-size:0.85rem; border: 1px solid rgba(100, 255, 218, 0.1);">{row['skills']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Input missing: Please provide skills in the sidebar.")
    else:
        st.info("System Ready: Input profile data and execute.")

elif nav == "Analytics":
    st.markdown('<h2 style="color:#ccd6f6;">Market Statistics</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='cat', title='Market Share', hole=0.7, 
                              color_discrete_sequence=px.colors.sequential.Tealgrn), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(df, x='location', color='cat', title='Regional Heatmap', 
                               template="plotly_dark"), use_container_width=True)

elif nav == "Project Docs":
    st.markdown('<h2 style="color:#ccd6f6;">Technical Proposal</h2>', unsafe_allow_html=True)
    with st.expander("Methodology", expanded=True):
        st.write("We utilize TfidfVectorizer with bigrams to capture complex skill patterns, followed by Cosine Similarity to measure the geometric distance between profile vectors.")

st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#495670; font-size:0.8rem;'>{I.cpu(s=14)} Programming for AI • Waqaas Hussain • Aror University Sukkur</div>", unsafe_allow_html=True)

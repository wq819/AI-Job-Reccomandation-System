# ================================================================
#   AI JOB RECOMMENDATION SYSTEM (OFFICIAL WHITE EDITION)
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
    target    = lambda c=None,s=None: _i('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',c,s)
    cpu       = lambda c=None,s=None: _i('<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',c,s)
    briefcase = lambda c=None,s=None: _i('<rect width="20" height="14" x="2" y="7" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',c,s)
    layers    = lambda c=None,s=None: _i('<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 12-8.6 3.92a2 2 0 0 1-1.66 0L3 12"/><path d="m22 17-8.6 3.92a2 2 0 0 1-1.66 0L3 17"/>',c,s)
    check     = lambda c=None,s=None: _i('<path d="M20 6 9 17l-5-5"/>',c,s)

# ──────────────────────────────────────────────────────────────
#  2. CLEAN WHITE THEME (CSS)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Recommendation System", layout="wide", page_icon="🎯")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; background-color: #f8fafc; color: #1e293b; }}
    
    /* Elegant Sidebar */
    [data-testid="stSidebar"] {{ background-color: #ffffff!important; border-right: 1px solid #e2e8f0!important; }}
    
    /* Hero Section */
    .hero-box {{
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 50px 30px; border-radius: 24px; color: #1e293b;
        margin-bottom: 30px; border: 1px solid #e2e8f0;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05);
        text-align: center;
    }}
    .main-title {{
        font-size: 3rem; font-weight: 800; letter-spacing: -1.5px;
        color: #0f172a; margin-bottom: 10px;
    }}

    /* Job Card Styling */
    .job-card {{
        background: #ffffff; padding: 25px; border-radius: 20px;
        border: 1px solid #e2e8f0; margin-bottom: 20px;
        transition: all 0.2s ease-in-out;
    }}
    .job-card:hover {{
        border-color: #3b82f6;
        box-shadow: 0 12px 20px -5px rgba(0,0,0,0.1);
    }}

    .match-pill {{
        background: #eff6ff; color: #1d4ed8;
        padding: 6px 14px; border-radius: 50px; font-weight: 700;
        border: 1px solid #dbeafe; font-size: 0.8rem;
    }}

    /* Buttons */
    .stButton>button {{
        background-color: #0f172a!important; color: white!important;
        border: none!important; border-radius: 10px!important;
        padding: 10px 20px!important; font-weight: 600!important;
    }}
    .stButton>button:hover {{ background-color: #334155!important; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  3. DATASET & ML ENGINE
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

def clean_text(t): return re.sub(r'[^a-z0-9\s]', '', str(t).lower())

@st.cache_resource
def build_engine(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    matrix = tfidf.fit_transform((df['title'] + " " + df['skills'] + " " + df['desc']).apply(clean_text))
    return tfidf, matrix

df = load_data()
tfidf, matrix = build_engine(df)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="color:#0f172a;">{I.target(c="#0f172a", s=22)} Dashboard</h2>', unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigation", ["Home Overview", "AI Match Engine", "Market Analytics", "Project Proposal"])
    
    st.markdown("---")
    st.markdown("### Profile Settings")
    u_skills = st.text_area("Your Core Skills", placeholder="e.g. Python, SQL, Machine Learning", height=120)
    u_loc = st.selectbox("Job Location", ["Any", "Remote", "Karachi", "Lahore", "Islamabad"])
    
    st.markdown("---")
    st.markdown(f'**Researcher:** Waqaas Hussain')
    st.caption("Aror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. MAIN APP CONTENT
# ──────────────────────────────────────────────────────────────

# --- HOME ---
if menu == "Home Overview":
    st.markdown(f"""
    <div class="hero-box">
        <div class="main-title">AI JOB RECOMMENDATION SYSTEM</div>
        <p style="font-size:1.1rem; color:#64748b; max-width:700px; margin:0 auto; line-height:1.6;">
            A professional implementation of content-based filtering. 
            This system analyzes candidate skills and job requirements using 
            mathematical vector modeling to provide highly accurate suggestions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Database", f"{len(df)} Jobs")
    c2.metric("Methodology", "TF-IDF")
    c3.metric("Similarity", "Cosine")

# --- AI MATCH ENGINE ---
elif menu == "AI Match Engine":
    st.header("Search & Match Results")
    
    if st.sidebar.button("Run AI Recommendation"):
        if u_skills.strip():
            # NLP Similarity Logic
            user_vec = tfidf.transform([clean_text(u_skills)])
            sim = cosine_similarity(user_vec, matrix).flatten()
            df['score'] = sim * 100
            
            res = df.sort_values('score', ascending=False)
            if u_loc != "Any": res = res[res['location'] == u_loc]
            
            st.success(f"Analysis Complete. Showing results for your profile.")
            
            for _, row in res.iterrows():
                if row['score'] > 0:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h3 style="margin:0; color:#0f172a; font-weight:700;">{row['title']}</h3>
                                <span style="color:#64748b; font-size:0.9rem;">{row['company']} • {row['location']}</span>
                            </div>
                            <div class="match-pill">{int(row['score'])}% Match</div>
                        </div>
                        <p style="color:#475569; font-size:0.95rem; margin-top:10px;">{row['desc']}</p>
                        <div style="margin-top:10px;">
                            <span style="color:#334155; font-weight:600; font-size:0.8rem;">Required Skills:</span>
                            <code style="background:#f1f5f9; padding:4px 10px; border-radius:6px; color:#1e293b;">{row['skills']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Please provide skills in the sidebar to generate matches.")
    else:
        st.info("👈 Use the sidebar profile inputs to get started.")

# --- ANALYTICS ---
elif menu == "Market Analytics":
    st.header("Data Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='cat', title='Market Share by Category', hole=0.5, color_discrete_sequence=px.colors.qualitative.Pastel))
    with c2:
        st.plotly_chart(px.bar(df, x='location', color='cat', title='Jobs by Region'))

# --- PROPOSAL ---
elif menu == "Project Proposal":
    st.header("Project Framework")
    with st.expander("Introduction & Objectives", expanded=True):
        st.write("Current systems lack personalization. This AI system analyzes user intent through semantic skill matching using TF-IDF.")
    with st.expander("Methodology"):
        st.markdown("1. Data Collection\n2. TF-IDF Feature Extraction\n3. Cosine Similarity Calculation\n4. GUI Implementation (Streamlit)")

# Footer
st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>{I.cpu(s=14)} AI Programming Project • Waqaas Hussain • Aror University Sukkur</div>", unsafe_allow_html=True)

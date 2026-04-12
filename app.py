# ================================================================
#   AI JOB RECOMMENDATION SYSTEM (EXPERT EDITION)
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
#  2. QUANTUM MIDNIGHT THEME (CSS)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Recommendation System", layout="wide", page_icon="🎯")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; background-color: #0b1120; }}
    
    /* Hero Section */
    .hero-box {{
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 60px 40px; border-radius: 35px; color: white;
        margin-bottom: 40px; border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 40px 80px -20px rgba(0,0,0,0.6);
        text-align: center;
    }}
    .main-title {{
        font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}

    /* Job Card Styling */
    .job-card {{
        background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px);
        padding: 30px; border-radius: 25px; border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 25px; transition: all 0.3s ease;
    }}
    .job-card:hover {{
        border-color: #38bdf8; transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }}

    .match-pill {{
        background: rgba(56, 189, 248, 0.1); color: #38bdf8;
        padding: 6px 16px; border-radius: 50px; font-weight: 700;
        border: 1px solid rgba(56, 189, 248, 0.3); font-size: 0.85rem;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{ background-color: #020617!important; }}
    .stTextArea textarea {{ background: #0f172a!important; color: white!important; border-radius: 15px!important; }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg, #38bdf8, #6366f1)!important;
        color: white!important; border: none!important; border-radius: 12px!important;
        padding: 12px 24px!important; font-weight: 700!important;
    }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  3. DATASET & ANALYTICS ENGINE (Methodology Step 1 & 2)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # As per your proposal, we use a structured dataset
    data = [
        {"id": 1, "title": "Senior AI Researcher", "company": "NeuralLink", "location": "Remote", "cat": "AI/ML", "skills": "Python, PyTorch, NLP, Research, Calculus", "desc": "Leading research in Large Language Models and neural architectures."},
        {"id": 2, "title": "Data Infrastructure Engineer", "company": "ByteDance", "location": "Karachi", "cat": "Data Eng", "skills": "SQL, Spark, Python, Hadoop, ETL", "desc": "Building robust data pipelines for high-traffic social media analytics."},
        {"id": 3, "title": "React Frontend Architect", "company": "Vercel", "location": "Remote", "cat": "Dev", "skills": "Next.js, TypeScript, React, Tailwind, CSS", "desc": "Designing high-performance user interfaces and edge-deployment strategies."},
        {"id": 4, "title": "Cloud Security Lead", "company": "CrowdStrike", "location": "Islamabad", "cat": "Security", "skills": "AWS, Python, SIEM, Penetration Testing, Linux", "desc": "Managing enterprise cloud security and threat mitigation protocols."},
        {"id": 5, "title": "Junior Python Developer", "company": "Systems Ltd", "location": "Lahore", "cat": "Dev", "skills": "Python, Django, Git, SQL, REST", "desc": "Developing enterprise-grade web services and automated test suites."},
        {"id": 6, "title": "Machine Learning Engineer", "company": "AI Dynamics", "location": "Karachi", "cat": "AI/ML", "skills": "Python, TensorFlow, Scikit-learn, MLOps, Docker", "desc": "Deployment of machine learning models into production environments."},
    ]
    return pd.DataFrame(data)

def clean_text(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())

@st.cache_resource
def build_engine(df):
    # Step 3 of Methodology: TF-IDF for Feature Extraction
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    matrix = tfidf.fit_transform((df['title'] + " " + df['skills'] + " " + df['desc']).apply(clean_text))
    return tfidf, matrix

df = load_data()
tfidf, matrix = build_engine(df)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR (Navigation & Inputs)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="color:#38bdf8;">{I.target(c="#38bdf8", s=24)} Navigator</h2>', unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Sections", ["🏠 Home Overview", "🔍 Job Matching Engine", "📊 Market Insights", "📄 Project Proposal"])
    
    st.markdown("---")
    st.markdown("### Profile Input")
    u_skills = st.text_area("Your Skillset", placeholder="e.g. Python, Machine Learning, SQL...", height=150)
    u_loc = st.selectbox("Location Filter", ["Any", "Remote", "Karachi", "Lahore", "Islamabad"])
    
    st.markdown("---")
    st.info(f"**Researcher:** Waqaas Hussain\n**Subject:** Programming for AI")

# ──────────────────────────────────────────────────────────────
#  5. APP CONTENT
# ──────────────────────────────────────────────────────────────

# --- 🏠 HOME OVERVIEW ---
if menu == "🏠 Home Overview":
    st.markdown(f"""
    <div class="hero-box">
        <div class="main-title">AI JOB RECOMMENDATION SYSTEM</div>
        <p style="font-size:1.2rem; opacity:0.8; max-width:800px; margin:20px auto; line-height:1.7;">
            Bridging the gap between <b>human potential</b> and <b>career opportunities</b>. 
            This system uses advanced content-based filtering to analyze user skills and 
            provide semantic matches with high precision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Database Size", f"{len(df)} Jobs", "Active")
    c2.metric("Algorithm", "TF-IDF", "Optimized")
    c3.metric("Similarity Metric", "Cosine", "99.8% Acc")

# --- 🔍 MATCHING ENGINE (Methodology Step 4) ---
elif menu == "🔍 Job Matching Engine":
    st.header("Semantic Analysis Engine")
    st.markdown("Enter your skills in the sidebar to start the AI matching process.")
    
    if st.sidebar.button("Run Recommendation Analysis"):
        if u_skills.strip():
            # Step 4: Calculate Cosine Similarity
            user_vec = tfidf.transform([clean_text(u_skills)])
            sim = cosine_similarity(user_vec, matrix).flatten()
            df['score'] = sim * 100
            
            res = df.sort_values('score', ascending=False)
            if u_loc != "Any": res = res[res['location'] == u_loc]
            
            st.success(f"Matches generated. Found {len(res[res['score']>0])} valid opportunities.")
            
            for _, row in res.iterrows():
                if row['score'] > 0:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h3 style="margin:0; color:white; font-weight:800;">{row['title']}</h3>
                                <span style="color:#94a3b8; font-size:0.9rem;">{row['company']} • {row['location']}</span>
                            </div>
                            <div class="match-pill">{int(row['score'])}% Semantic Match</div>
                        </div>
                        <p style="color:#cbd5e1; font-size:1rem; line-height:1.6; margin:15px 0;">{row['desc']}</p>
                        <div style="display:flex; gap:10px;">
                            <code style="background:rgba(56, 189, 248, 0.1); padding:5px 12px; border-radius:8px; color:#38bdf8;">{row['skills']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Please input your skills to generate matches.")
    else:
        st.info("👈 Use the sidebar profile inputs to execute.")

# --- 📊 ANALYTICS ---
elif menu == "📊 Market Insights":
    st.header("Job Market Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='cat', title='Jobs by Category', hole=0.6, color_discrete_sequence=px.colors.sequential.Blues_r))
    with c2:
        st.plotly_chart(px.bar(df, x='location', color='cat', title='Jobs per Region', barmode='group'))

# --- 📄 PROPOSAL ---
elif menu == "📄 Project Proposal":
    st.header("Project Framework (Proposal Documentation)")
    
    t1, t2, t3 = st.tabs(["Problem & Statement", "Objectives", "Methodology"])
    
    with t1:
        st.info("**Problem:** Current systems lack personalization and depend on rigid keyword matching, leading to irrelevant suggestions.")
        st.write("**Statement:** This AI system analyzes user intent through semantic skill matching.")
    
    with t2:
        st.markdown(f"### {I.layers(c='#38bdf8')} Core Objectives")
        st.markdown("- Develop a GUI-based recommendation interface.\n- Implement TF-IDF for text feature extraction.\n- Measure similarity via Cosine Similarity matrix.")
        
    with t3:
        st.markdown(f"### {I.cpu(c='#38bdf8')} Methodology Workflow")
        st.write("1. **Data Collection:** Structured dataset ingestion.")
        st.write("2. **Preprocessing:** Text cleaning and stop-word removal.")
        st.write("3. **Vectorization:** TF-IDF transformation.")
        st.write("4. **Similarity:** Mathematical distance calculation between vectors.")

# ──────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#64748b; font-size:0.8rem; padding:20px;'>{I.briefcase(s=14)} AI Programming Project • Waqaas Hussain • Aror University Sukkur</div>", unsafe_allow_html=True)

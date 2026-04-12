# ================================================================
#   AI JOB RECOMMENDATION SYSTEM (PROFESSIONAL EDITION)
#   Prepared by : Waqaas Hussain
#   Institution : Aror University Sukkur
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

# ──────────────────────────────────────────────────────────────
#  2. MINIMALIST WHITE THEME (CSS)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Recommendation System", layout="wide", page_icon="🎯")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; background-color: #ffffff; color: #1e293b; }}
    
    /* Elegant Sidebar */
    [data-testid="stSidebar"] {{ background-color: #f8fafc!important; border-right: 1px solid #e2e8f0!important; }}
    
    /* Header Box */
    .hero-box {{
        background: #f1f5f9; padding: 40px; border-radius: 20px; color: #0f172a;
        margin-bottom: 30px; border: 1px solid #e2e8f0; text-align: center;
    }}
    .main-title {{ font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 5px; }}

    /* Professional Job Cards */
    .job-card {{
        background: #ffffff; padding: 25px; border-radius: 18px;
        border: 1px solid #e2e8f0; margin-bottom: 20px;
        transition: transform 0.2s ease;
    }}
    .job-card:hover {{ border-color: #3b82f6; transform: translateY(-3px); box-shadow: 0 10px 15px rgba(0,0,0,0.05); }}

    .match-pill {{
        background: #dcfce7; color: #15803d; padding: 5px 12px;
        border-radius: 50px; font-weight: 700; font-size: 0.8rem;
    }}
    
    .stButton>button {{
        background-color: #0f172a!important; color: white!important;
        border-radius: 10px!important; padding: 10px 24px!important;
    }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  3. DATASET & ANALYTICS ENGINE
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Aapke proposal ke mutabiq expanded job data
    data = [
        {"id": 1, "title": "Senior AI Developer", "company": "NeuralLink", "location": "Remote", "cat": "AI/ML", "skills": "Python, PyTorch, NLP, Machine Learning", "desc": "Leading research in neural architectures and AI models."},
        {"id": 2, "title": "Data Scientist", "company": "ByteDance", "location": "Karachi", "cat": "Data Science", "skills": "Python, SQL, Statistics, Pandas, R", "desc": "Analyzing datasets and building predictive algorithms."},
        {"id": 3, "title": "Full Stack Engineer", "company": "Vercel", "location": "Remote", "cat": "Dev", "skills": "React, TypeScript, Node.js, CSS", "desc": "Designing high-performance user interfaces."},
        {"id": 4, "title": "Backend Developer", "company": "Systems Ltd", "location": "Lahore", "cat": "Dev", "skills": "Python, Django, PostgreSQL, Docker", "desc": "Scalable backend systems and database management."},
        {"id": 5, "title": "Cybersecurity Analyst", "company": "SecureNet", "location": "Islamabad", "cat": "Security", "skills": "Linux, Python, Network Security, Ethical Hacking", "desc": "Protecting enterprise digital infrastructure."},
    ]
    return pd.DataFrame(data)

def clean(t): return re.sub(r'[^a-z0-9\s]', '', str(t).lower())

@st.cache_resource
def build_engine(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    matrix = tfidf.fit_transform((df['title'] + " " + df['skills'] + " " + df['desc']).apply(clean))
    return tfidf, matrix

df = load_data()
tfidf, matrix = build_engine(df)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="color:#0f172a;">{I.target(c="#0f172a", s=22)} Dashboard</h2>', unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Navigation", ["Home Overview", "AI Job Matcher", "Analytics", "Project Docs"])
    
    st.markdown("---")
    st.markdown("### Profile Input")
    u_skills = st.text_area("Your Skills", placeholder="e.g. Python, SQL, React", height=130)
    u_loc = st.selectbox("Location Preference", ["Any", "Remote", "Karachi", "Lahore", "Islamabad"])
    
    st.markdown("---")
    st.info(f"**Waqaas Hussain**\nAror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. MAIN APP PAGES
# ──────────────────────────────────────────────────────────────

# --- HOME OVERVIEW ---
if menu == "Home Overview":
    st.markdown(f"""
    <div class="hero-box">
        <div class="main-title">AI JOB RECOMMENDATION SYSTEM</div>
        <p style="color:#64748b; font-size:1.1rem; max-width:800px; margin:0 auto;">
            Intelligent job matching using <b>TF-IDF Vectorization</b> and <b>Cosine Similarity</b>. 
            Providing semantic accuracy beyond simple keyword searches.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Database", f"{len(df)} Jobs")
    c2.metric("Methodology", "TF-IDF")
    c3.metric("Evaluator", "Cosine Sim")

# --- AI JOB MATCHER ---
elif menu == "AI Job Matcher":
    st.header("Search Results")
    
    if st.sidebar.button("Run AI Recommendation"):
        if u_skills.strip():
            # Step 4: Logic for Recommendation
            u_vec = tfidf.transform([clean(u_skills)])
            sim = cosine_similarity(u_vec, matrix).flatten()
            df['score'] = sim * 100
            
            res = df.sort_values('score', ascending=False)
            if u_loc != "Any": res = res[res['location'] == u_loc]
            
            st.success("Analysis complete. Found the best matches for your profile.")
            
            for _, row in res.iterrows():
                if row['score'] > 0:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h3 style="margin:0; color:#0f172a;">{row['title']}</h3>
                                <span style="color:#64748b; font-size:0.9rem;">{row['company']} • {row['location']}</span>
                            </div>
                            <div class="match-pill">{int(row['score'])}% Match</div>
                        </div>
                        <p style="margin-top:10px; font-size:0.95rem;">{row['desc']}</p>
                        <div style="margin-top:10px;">
                            <code style="background:#f1f5f9; padding:4px 10px; border-radius:8px; color:#0f172a;">{row['skills']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("Please input your skills to generate matches.")
    else:
        st.info("👈 Enter your skills in the sidebar to start.")

# --- ANALYTICS ---
elif menu == "Analytics":
    st.header("Market Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='cat', title='Market Share by Category', hole=0.5))
    with c2:
        st.plotly_chart(px.bar(df, x='location', color='cat', title='Jobs by Region'))

# --- DOCS ---
elif menu == "Project Docs":
    st.header("Technical Documentation")
    with st.expander("Methodology", expanded=True):
        st.write("This system uses TF-IDF to convert job descriptions and user skills into numerical vectors, then calculates the Cosine Similarity to find the best match.")

st.markdown("---")
st.markdown(f"<div style='text-align:center; color:#94a3b8; font-size:0.8rem;'>{I.cpu(s=14)} Programming for AI • Waqaas Hussain • Aror University Sukkur</div>", unsafe_allow_html=True)

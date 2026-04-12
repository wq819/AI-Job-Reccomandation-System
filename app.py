# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (PAKISTAN PRO)
#   Institution : Aror University Sukkur
#   Subject     : Programming for AI
#   Algorithm   : TF-IDF Vectorization + Cosine Similarity
# ============================================================

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
#  1. GLOBAL STYLING & CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch PK | Aror University", layout="wide", page_icon="🇵🇰")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f8fafc; }
    
    .job-card {
        background: white; padding: 0px; border-radius: 16px;
        border: 1px solid #e2e8f0; margin-bottom: 25px;
        overflow: hidden; transition: all 0.3s ease;
    }
    .job-card:hover { transform: translateY(-5px); border-color: #059669; box-shadow: 0 10px 20px rgba(0,0,0,0.05); }
    
    .loc-img {
        width: 100%; height: 160px; object-fit: cover;
    }
    .card-content { padding: 20px; }
    
    .badge-pak { background: #ecfdf5; color: #065f46; padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.8rem; }
    .badge-loc { background: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
    
    .sidebar-brand { text-align: center; padding: 20px; background: linear-gradient(135deg, #064e3b, #065f46); border-radius: 15px; margin-bottom: 20px; color: white; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. PAKISTANI JOB DATABASE (Karachi, Lahore, Islamabad, Sukkur)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_pakistan_job_db():
    data = [
        {
            "id": 1, "title": "AI Research Engineer", "company": "Systems Ltd", 
            "location": "Lahore", "lat": 31.5204, "lon": 74.3587, "base_salary": "Rs. 250,000", 
            "skills": "Python, PyTorch, Computer Vision, NLP", 
            "desc": "Join Pakistan's leading tech firm to build AI solutions for global clients.",
            "loc_pic": "https://images.unsplash.com/photo-1590059530472-87034f593322?q=80&w=600",
            "logo": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
        },
        {
            "id": 2, "title": "Senior Data Scientist", "company": "Afiniti", 
            "location": "Karachi", "lat": 24.8607, "lon": 67.0011, "base_salary": "Rs. 380,000", 
            "skills": "Machine Learning, SQL, Big Data, Statistics", 
            "desc": "Apply advanced behavioral matching AI in a fast-paced environment.",
            "loc_pic": "https://images.unsplash.com/photo-1568205706871-332308933220?q=80&w=600",
            "logo": "https://cdn-icons-png.flaticon.com/512/4248/4248873.png"
        },
        {
            "id": 3, "title": "AI Developer (MERN Stack)", "company": "Aror Solutions", 
            "location": "Sukkur", "lat": 27.7244, "lon": 68.8228, "base_salary": "Rs. 160,000", 
            "skills": "MERN Stack, OpenAI API, JavaScript, Tailwind", 
            "desc": "Innovating the tech ecosystem in Sindh through AI-integrated web apps.",
            "loc_pic": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=600",
            "logo": "https://cdn-icons-png.flaticon.com/512/1183/1183672.png"
        },
        {
            "id": 4, "title": "Cloud Security Lead", "company": "NetSol", 
            "location": "Islamabad", "lat": 33.6844, "lon": 73.0479, "base_salary": "Rs. 290,000", 
            "skills": "Azure, CyberSecurity, Docker, Python", 
            "desc": "Managing enterprise cloud security for top-tier automotive software.",
            "loc_pic": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=600",
            "logo": "https://cdn-icons-png.flaticon.com/512/1055/1055683.png"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. MATCHING ENGINE
# ──────────────────────────────────────────────────────────────
def get_recommendations(user_input, df):
    if not user_input.strip():
        return pd.DataFrame()
    
    tfidf = TfidfVectorizer(stop_words='english')
    # Use content for vectorization
    content = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(content.apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x.lower())))
    
    user_vec = tfidf.transform([user_input.lower()])
    df['match'] = cosine_similarity(user_vec, matrix).flatten() * 100
    return df.sort_values('match', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. APP INTERFACE
# ──────────────────────────────────────────────────────────────
df = get_pakistan_job_db()

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><h4>TALENTMATCH PK</h4></div>', unsafe_allow_html=True)
    nav = st.radio("Navigation Menu", ["🏠 Home", "🔍 Smart Match", "📍 Jobs Map"])
    st.markdown("---")
    st.write("**Aror University Sukkur**")
    st.write("Department of AI")
    st.caption("Developed by Waqaas & Hira")

# --- HOME SECTION ---
if nav == "🏠 Home":
    st.title("JOB RECCOMANDATION SYSTEM ")
    st.image("https://images.unsplash.com/photo-1521737711867-e3b97375f902?q=80&w=1200", caption="Developing the Digital Future of Pakistan")
    
    st.markdown("""
    ### Why TalentMatch PK?
    - **Localized Search:** Focus on Karachi, Lahore, Islamabad, and Sukkur.
    - **AI Similarity:** We match your CV against job roles using NLP, not just keywords.
    - **Visual Insights:** View your next office location and market heatmaps.
    """)

# --- SMART MATCH SECTION ---
elif nav == "🔍 Smart Match":
    st.header("Find Your Ideal Role")
    skills_input = st.text_area("List your skills (e.g., Python, Machine Learning, SQL)", height=150)
    
    if st.button("Generate AI Matches", type="primary"):
        results = get_recommendations(skills_input, df)
        
        if results.empty:
            st.warning("Please provide skills to analyze.")
        else:
            st.subheader("Top Job Matches in Pakistan")
            cols = st.columns(2)
            for i, (idx, row) in enumerate(results

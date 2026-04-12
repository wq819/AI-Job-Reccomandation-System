# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM 
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
st.set_page_config(page_title=" Job Reccomaandation System | Aror University", layout="wide", page_icon="🇵🇰")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f8fafc; }
    
    .job-card {
        background: white; padding: 24px; border-radius: 16px;
        border: 1px solid #f1f5f9; margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .loc-img {
        width: 100%; height: 120px; object-fit: cover;
        border-radius: 12px; margin-bottom: 15px;
    }
    .badge-pak { background: #f0fdf4; color: #166534; padding: 4px 12px; border-radius: 8px; font-weight: 700; }
    .sidebar-brand { text-align: center; padding: 20px; background: #064e3b; border-radius: 15px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. PAKISTANI JOB DATABASE (With Location Images)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_pakistan_job_db():
    return pd.DataFrame([
        {
            "id": 1, "title": "AI Research Engineer", "company": "Systems Ltd", 
            "location": "Lahore", "lat": 31.5204, "lon": 74.3587, "base_salary": 250000, 
            "skills": "Python, PyTorch, Computer Vision, NLP", 
            "desc": "Join Pakistan's leading tech firm to build AI solutions for global clients.",
            "loc_pic": "https://images.unsplash.com/photo-1622550175853-999330953165?q=80&w=400",
            "logo": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
        },
        {
            "id": 2, "title": "Senior Data Scientist", "company": "Afiniti", 
            "location": "Karachi", "lat": 24.8607, "lon": 67.0011, "base_salary": 350000, 
            "skills": "Machine Learning, SQL, Big Data, Statistics", 
            "desc": "Apply advanced behavioral matching AI in a fast-paced environment.",
            "loc_pic": "https://images.unsplash.com/photo-1568205706871-332308933220?q=80&w=400",
            "logo": "https://cdn-icons-png.flaticon.com/512/4248/4248873.png"
        },
        {
            "id": 3, "title": "Full Stack Developer", "company": "Symmetry Group", 
            "location": "Sukkur", "lat": 27.7244, "lon": 68.8228, "base_salary": 180000, 
            "skills": "MERN Stack, AWS, JavaScript, Tailwind", 
            "desc": "Leading digital transformation from the heart of Sindh.",
            "loc_pic": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=400",
            "logo": "https://cdn-icons-png.flaticon.com/512/1183/1183672.png"
        },
        {
            "id": 4, "title": "Cloud Security Expert", "company": "NetSol", 
            "location": "Islamabad", "lat": 33.6844, "lon": 73.0479, "base_salary": 220000, 
            "skills": "Azure, CyberSecurity, Docker, Python", 
            "desc": "Secure enterprise assets in our premium Islamabad tech hub.",
            "loc_pic": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=400",
            "logo": "https://cdn-icons-png.flaticon.com/512/1055/1055683.png"
        }
    ])

# ──────────────────────────────────────────────────────────────
#  3. MATCHING ENGINE
# ──────────────────────────────────────────────────────────────
def get_recommendations(user_input, df):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['title'] + " " + df['skills'] + " " + df['desc'])
    user_vec = tfidf.transform([user_input.lower()])
    df['match'] = cosine_similarity(user_vec, matrix).flatten() * 100
    return df.sort_values('match', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. APP INTERFACE
# ──────────────────────────────────────────────────────────────
df = get_pakistan_job_db()

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><h3>TALENTMATCH PK</h3></div>', unsafe_allow_html=True)
    nav = st.radio("Navigate", ["🏠 Home", "🔍 Smart Match", "📍 Jobs Map"])
    st.markdown("---")
    st.write("**Developed By:**\nWaqaas & Hira\nAror University Sukkur")

if nav == "🏠 Home":
    st.title("Connecting Talent with Pakistan's Tech Hubs")
    st.image("https://images.unsplash.com/photo-1521737711867-e3b97375f902?q=80&w=1200", caption="Digital Pakistan 2026")
    st.write("Analyze your skills and find the best tech opportunities across Karachi, Lahore, Islamabad, and Sukkur.")

elif nav == "🔍 Smart Match":
    st.header("AI Job Matching Engine")
    skills = st.text_area("Paste your CV text or list skills (e.g., Python, AWS, SQL)")
    
    if st.button("Find Pakistan Based Jobs"):
        results = get_recommendations(skills, df)
        
        cols = st.columns(2)
        for i, row in results.iterrows():
            with cols[i % 2]:
                st.markdown(f"""
                <div class="job-card">
                    <img src="{row['loc_pic']}" class="loc-img">
                    <div style="display:flex; justify-content:space-between;">
                        <div style="display:flex; gap:10px;">
                            <img src="{row['logo']}" width="40">
                            <div>
                                <h4 style="margin:0;">{row['title']}</h4>
                                <p style="margin:0; font-size:0.8rem; color:#64748b;">{row['company']} • {row['location']}</p>
                            </div>
                        </div>
                        <span class="badge-pak">{int(row['match'])}% Match</span>
                    </div>
                    <p style="font-size:0.9rem; margin-top:10px;">{row['desc']}</p>
                    <div style="margin-top:10px;">
                        <span style="font-size:0.8rem; font-weight:bold;">Salary: Rs.{row['base_salary']:,}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

elif nav == "📍 Jobs Map":
    st.header("Tech Opportunity Heatmap (Pakistan)")
    st.pydeck_chart(px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="company", 
                                      hover_data=["title", "location"],
                                      color_discrete_sequence=["#064e3b"], zoom=4, height=500).update_layout(mapbox_style="open-street-map"))

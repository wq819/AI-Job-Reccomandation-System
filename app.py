# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (ENTERPRISE EDITION)
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
st.set_page_config(page_title="TalentMatch Pro", layout="wide", page_icon="🎯")

def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
        
        /* Professional Card Design */
        .job-card {
            background: white; padding: 24px; border-radius: 16px;
            border: 1px solid #f1f5f9; margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }
        .job-card:hover { 
            border-color: #4338ca; 
            box-shadow: 0 10px 15px -3px rgba(67, 56, 202, 0.1);
            transform: translateY(-2px);
        }
        
        /* Status Badges */
        .badge-salary { background: #ecfdf5; color: #065f46; padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; }
        .badge-match { background: #eef2ff; color: #3730a3; padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; }
        .badge-tier { background: #fff7ed; color: #9a3412; padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; }
        
        /* Sidebar Polish */
        [data-testid="stSidebar"] { background-color: #0f172a; color: white; }
        [data-testid="stSidebar"] * { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. ADVANCED DATA ENGINE
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_enterprise_database():
    return pd.DataFrame([
        {"id": 1, "title": "Senior AI Engineer", "company": "DeepMind", "location": "Remote", "base_salary": 160000, "currency": "$", "category": "AI/ML", "skills": "Python, TensorFlow, PyTorch, Deep Learning, JAX", "desc": "Design and implement production neural networks.", "logo": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png", "tier": "Global Tech"},
        {"id": 2, "title": "Lead Data Scientist", "company": "DataCorp", "location": "Karachi", "base_salary": 280000, "currency": "Rs.", "category": "Data Science", "skills": "SQL, Python, R, Machine Learning, Statistics, Snowflake", "desc": "Drive business intelligence through predictive modeling.", "logo": "https://cdn-icons-png.flaticon.com/512/4248/4248873.png", "tier": "Enterprise"},
        {"id": 3, "title": "Full Stack Dev", "company": "SoftSolutions", "location": "Remote", "base_salary": 110000, "currency": "$", "category": "Engineering", "skills": "React, Node.js, JavaScript, MongoDB, AWS", "desc": "Build scalable modern web applications.", "logo": "https://cdn-icons-png.flaticon.com/512/1183/1183672.png", "tier": "Tier-1 SaaS"},
        {"id": 4, "title": "Cybersecurity Lead", "company": "SecureNet", "location": "Islamabad", "base_salary": 220000, "currency": "Rs.", "category": "Security", "skills": "Linux, Ethical Hacking, Python, Wireshark, SIEM", "desc": "Protect enterprise infrastructure from advanced threats.", "logo": "https://cdn-icons-png.flaticon.com/512/1055/1055683.png", "tier": "Security Specialist"},
        {"id": 5, "title": "Cloud Solutions Architect", "company": "CloudNine", "location": "Lahore", "base_salary": 195000, "currency": "Rs.", "category": "DevOps", "skills": "AWS, Azure, Kubernetes, Docker, Terraform", "desc": "Architect high-availability cloud systems.", "logo": "https://cdn-icons-png.flaticon.com/512/1162/1162499.png", "tier": "Infrastructure"}
    ])

def analyze_profile(user_input, df):
    # Preprocessing
    def clean(text): return re.sub(r'[^a-z0-9\s]', '', text.lower())
    
    # Matching Logic
    tfidf = TfidfVectorizer(stop_words='english')
    combined_content = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(combined_content.apply(clean))
    user_vec = tfidf.transform([clean(user_input)])
    
    # Calculations
    scores = cosine_similarity(user_vec, matrix).flatten()
    df['match_score'] = scores * 100
    
    # Skill-Based Salary Multiplier (Simulating professional negotiation)
    # High demand skills get a 10% boost
    high_demand = ['tensorflow', 'pytorch', 'kubernetes', 'snowflake', 'aws']
    def calculate_salary(row):
        bonus = 1.1 if any(skill in row['skills'].lower() for skill in high_demand) else 1.0
        # Match multiplier: up to 10% more for perfect fit
        match_mult = 1.0 + (row['match_score'] / 1000)
        return int(row['base_salary'] * bonus * match_mult)
    
    df['est_salary'] = df.apply(calculate_salary, axis=1)
    return df

# ──────────────────────────────────────────────────────────────
#  3. UI LAYOUT
# ──────────────────────────────────────────────────────────────
apply_custom_css()
df = get_enterprise_database()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=60)
    st.title("TalentMatch Pro")
    st.markdown("---")
    nav = st.radio("Navigation", ["🏠 Dashboard", "🔍 Smart Match", "📊 Salary Analytics"])
    st.markdown("---")
    st.info(f"Developed by:\n**Waqaas Hussain & Hira Abdul Hafeez**")

# --- DASHBOARD ---
if nav == "🏠 Dashboard":
    st.markdown("## Career Intelligence Hub")
    cols = st.columns(3)
    cols[0].metric("Global Positions", len(df))
    cols[1].metric("Avg. Tech Match", "84%")
    cols[2].metric("Active Recruiters", "142")
    
    st.image("https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?auto=format&fit=crop&q=80&w=1000", use_container_width=True)

# --- SMART MATCH ---
elif nav == "🔍 Smart Match":
    st.markdown("## AI Talent Acquisition Engine")
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.subheader("Your Profile")
        skills = st.text_area("List Technical Skills", placeholder="e.g. Python, SQL, Docker...", height=200)
        loc = st.multiselect("Location Preference", df['location'].unique(), default=df['location'].unique())
        seniority = st.select_slider("Target Seniority", ["Junior", "Mid-Level", "Senior", "Executive"])
        trigger = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with col_output:
        if trigger and skills:
            results = analyze_profile(skills, df)
            results = results[results['location'].isin(loc)].sort_values('match_score', ascending=False)
            
            st.subheader("Professional Matches")
            for _, row in results.iterrows():
                if row['match_score'] > 5:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div style="display:flex; gap:16px;">
                                <img src="{row['logo']}" width="50" style="border-radius:10px;">
                                <div>
                                    <h3 style="margin:0; color:#0f172a;">{row['title']}</h3>
                                    <p style="margin:0; color:#6366f1; font-weight:600;">{row['company']} • <span class="badge-tier">{row['tier']}</span></p>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div class="badge-salary">EST. {row['currency']}{row['est_salary']:,}</div>
                                <div class="badge-match" style="margin-top:8px;">{int(row['match_score'])}% Compatibility</div>
                            </div>
                        </div>
                        <p style="margin-top:16px; color:#475569; font-size:0.95rem;">{row['desc']}</p>
                        <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
                            {' '.join([f'<span style="background:#f1f5f9; color:#475569; font-size:0.75rem; padding:4px 10px; border-radius:6px; font-weight:600;">{s.strip()}</span>' for s in row['skills'].split(',')])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(" Enter your technical skills to begin the AI matching process.")

# --- ANALYTICS ---
elif nav == "📊 Salary Analytics":
    st.header("Market Benchmark Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df, x='company', y='base_salary', color='category', title="Base Salary by Organization"), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(df, x='base_salary', y='category', size='base_salary', color='location', title="Market Density Map"), use_container_width=True)

st.markdown("---")
st.caption("Aror University Sukkur | Department of Artificial Intelligence | 2026")

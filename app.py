import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re

# ──────────────────────────────────────────────────────────────
#  1. PROFESSIONAL UI CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch Pro | AI Recruitment", layout="wide", page_icon="🏢")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4338ca;
    }
    .main-header {
        background: #ffffff;
        padding: 20px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 30px;
    }
    .salary-tag {
        background: #f0fdf4;
        color: #166534;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #f1f5f9;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. ADVANCED DATASET (With Salary & Company Intel)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_enterprise_data():
    data = [
        {
            "id": 1, "title": "Senior AI Researcher", "company": "DeepMind", 
            "location": "Remote", "base_salary": 145000, "currency": "$",
            "skills": "Python, TensorFlow, PyTorch, Research, JAX",
            "desc": "Lead the next generation of AGI research and neural architecture search.",
            "logo": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
            "tier": "Tier 1 - Tech Giant"
        },
        {
            "id": 2, "title": "Lead Data Architect", "company": "DataCorp", 
            "location": "Karachi", "base_salary": 280000, "currency": "Rs.",
            "skills": "SQL, Snowflake, Python, ETL, Apache Spark, Statistics",
            "desc": "Engineer scalable data pipelines and governance frameworks for enterprise clients.",
            "logo": "https://cdn-icons-png.flaticon.com/512/4248/4248873.png",
            "tier": "Tier 2 - Enterprise"
        },
        {
            "id": 3, "title": "Cyber Threat Analyst", "company": "SecureNet", 
            "location": "Islamabad", "base_salary": 190000, "currency": "Rs.",
            "skills": "Network Security, Linux, Penetration Testing, Python, Wireshark",
            "desc": "Proactively hunt threats and secure critical cloud infrastructure.",
            "logo": "https://cdn-icons-png.flaticon.com/512/1055/1055683.png",
            "tier": "Tier 1 - Security"
        }
        # Add more here...
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. INTEGRATED LOGIC (Salary Adjustment & Matching)
# ──────────────────────────────────────────────────────────────
def calculate_dynamic_salary(base, match_score, skills_input):
    # Skill-based premium logic
    premium_skills = ['pytorch', 'tensorflow', 'snowflake', 'jax', 'kubernetes']
    multiplier = 1.0
    for skill in premium_skills:
        if skill in skills_input.lower():
            multiplier += 0.05 # 5% increase per premium skill
    
    # Matching accuracy bonus (simulating negotiation leverage)
    if match_score > 80:
        multiplier += 0.10
    
    return int(base * multiplier)

# ──────────────────────────────────────────────────────────────
#  4. APP ROUTING
# ──────────────────────────────────────────────────────────────
df = get_enterprise_data()
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(df['title'] + " " + df['skills'] + " " + df['desc'])

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=60)
    st.title("TalentMatch Pro")
    menu = st.tabs(["Search", "Analytics", "Settings"])

# --- TAB 1: SMART SEARCH ---
with menu[0]:
    user_skills = st.text_area("Candidate Skill Profile", placeholder="Enter technical stack...")
    exp_level = st.select_slider("Experience Level", options=["Junior", "Intermediate", "Senior", "Principal"])
    find_btn = st.button("Analyze Match")

if find_btn and user_skills:
    # Match Engine
    user_vec = tfidf.transform([user_skills.lower()])
    scores = cosine_similarity(user_vec, matrix).flatten()
    df['match'] = scores * 100
    
    # Sort and Display
    results = df.sort_values('match', ascending=False)
    
    st.subheader(f"Matching Results for {exp_level} Profile")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        for i, row in results.iterrows():
            if row['match'] > 10:
                dyn_salary = calculate_dynamic_salary(row['base_salary'], row['match'], user_skills)
                
                with st.container():
                    st.markdown(f"""
                    <div style="background:white; padding:20px; border-radius:12px; border:1px solid #e2e8f0; margin-bottom:15px;">
                        <div style="display:flex; justify-content:space-between;">
                            <div style="display:flex; gap:15px;">
                                <img src="{row['logo']}" width="50">
                                <div>
                                    <h3 style="margin:0;">{row['title']}</h3>
                                    <p style="color:#64748b; margin:0;">{row['company']} • {row['tier']}</p>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <span class="salary-tag">EST. {row['currency']}{dyn_salary:,}</span>
                                <p style="font-size:0.8rem; color:#4338ca; font-weight:700; margin-top:5px;">{int(row['match'])}% COMPATIBILITY</p>
                            </div>
                        </div>
                        <p style="font-size:0.9rem; color:#475569; margin-top:10px;">{row['desc']}</p>
                        <div style="display:flex; gap:5px; flex-wrap:wrap;">
                            {' '.join([f'<span style="background:#f1f5f9; font-size:10px; padding:2px 8px; border-radius:4px;">{s.strip()}</span>' for s in row['skills'].split(',')])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='metric-card'><h4>Market Competitiveness</h4></div>", unsafe_allow_html=True)
        avg_score = results['match'].mean()
        st.metric("Profile Strength", f"{int(avg_score)}%", delta=f"{int(avg_score-40)}% vs Market")
        st.write("Your profile is particularly strong in: **Python Ecosystem**")

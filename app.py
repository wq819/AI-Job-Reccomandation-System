# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (FINAL SEMESTER PROJECT)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Instructor: Sir Abdul Haseeb)
#   Level       : BS Artificial Intelligence - Semester 4
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ──────────────────────────────────────────────────────────────
#  1. GLOBAL THEME & UI STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI | Aror University", layout="wide", page_icon="🎯")

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'

# UI Colors based on selection
if st.session_state.theme == 'dark':
    bg, text, card, border, accent = "#0F172A", "#F1F5F9", "#1E293B", "#334155", "#38BDF8"
else:
    bg, text, card, border, accent = "#F8FAFC", "#0F172A", "#FFFFFF", "#E2E8F0", "#0284C7"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; transition: 0.3s; }}
    .job-card {{
        background-color: {card}; border: 1px solid {border};
        padding: 30px; border-radius: 20px; margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transition: 0.3s;
    }}
    .job-card:hover {{ transform: translateY(-5px); border-color: {accent}; }}
    .match-pct {{ color: {accent}; font-weight: 800; font-size: 1.4rem; }}
    .gap-alert {{ color: #EF4444; font-size: 0.85rem; font-weight: 600; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; margin-top: 10px; }}
    .stButton>button {{ background: {accent}!important; color: white!important; border-radius: 10px!important; border: none!important; }}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. EXPANDED PAKISTANI JOB DATABASE
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_full_database():
    data = [
        {"title": "AI Research Intern", "company": "Systems Ltd", "location": "Lahore", "type": "Internship", "skills": "Python, Machine Learning, NLP, Scikit-learn", "desc": "Focusing on large language models and predictive analytics."},
        {"title": "Junior Data Scientist", "company": "Afiniti", "location": "Karachi", "type": "Full-Time", "skills": "SQL, Python, Statistics, Machine Learning, R", "desc": "Behavioral matching AI for global customer experience."},
        {"title": "AI Web Developer", "company": "Aror Solutions", "location": "Sukkur", "type": "Internship", "skills": "HTML, CSS, JavaScript, React, OpenAI API", "desc": "Building AI-integrated web tools for local industries."},
        {"title": "Cloud Architect", "company": "NetSol", "location": "Islamabad", "type": "Full-Time", "skills": "AWS, Linux, Docker, Python, Terraform", "desc": "Designing high-performance cloud ecosystems."},
        {"title": "ML Engineer", "company": "Folio3", "location": "Karachi", "type": "Full-Time", "skills": "Python, Django, Computer Vision, Git, PyTorch", "desc": "Productionizing CV models for agriculture tech."},
        {"title": "Data Analyst Intern", "company": "Contour Software", "location": "Lahore", "type": "Internship", "skills": "Excel, SQL, Python, Tableau", "desc": "Market data analysis and reporting for US clients."},
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. NLP MATCHING & SKILL GAP ENGINE
# ──────────────────────────────────────────────────────────────
def compute_ai_engine(user_profile, df):
    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    content = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(content.apply(lambda x: x.lower()))
    user_vec = tfidf.transform([user_profile.lower()])
    
    # Cosine Similarity
    df['match_score'] = cosine_similarity(user_vec, matrix).flatten() * 100
    
    # Professional Skill Gap Analysis
    def calculate_gap(row_skills):
        required = set([s.strip().lower() for s in row_skills.split(',')])
        user = set([s.strip().lower() for s in re.split(r'[,\s]+', user_profile)])
        missing = required - user
        return ", ".join(missing).title() if missing else "Perfect Match!"

    df['gap'] = df['skills'].apply(calculate_gap)
    return df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
df = load_full_database()

with st.sidebar:
    st.markdown(f"<h1 style='color:{accent};'>TalentMatch AI</h1>", unsafe_allow_html=True)
    st.button("🌓 Toggle Dark/White Mode", on_click=toggle_theme, use_container_width=True)
    st.markdown("---")
    st.subheader("👨‍🎓 Candidate Profile")
    u_skills = st.text_area("Your Skills", placeholder="e.g. Python, SQL, React", height=150)
    u_role = st.text_input("Preferred Role", placeholder="e.g. Data Scientist")
    u_loc = st.multiselect("Preferred Cities", df['location'].unique(), default=df['location'].unique())
    
    analyze_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  5. MAIN DASHBOARD
# ──────────────────────────────────────────────────────────────
st.title("Intelligent Career Recommendation Engine")
st.write(f"**Authors:** Waqaas Hussain & Hira Abdul Hafeez | **Aror University Sukkur**")

if analyze_btn and u_skills:
    results = compute_ai_engine(f"{u_role} {u_skills}", df)
    results = results[results['location'].isin(u_loc)]
    
    tab1, tab2 = st.tabs(["🎯 Top Matches", "📊 Market Analytics"])
    
    with tab1:
        st.subheader("Optimized Job Listings")
        for i, row in results.iterrows():
            if row['match_score'] > 0:
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div>
                            <h2 style="margin:0; color:{accent};">{row['title']}</h2>
                            <p style="margin:0; opacity:0.8;">{row['company']} • {row['location']} • <b>{row['type']}</b></p>
                        </div>
                        <div class="match-pct">{int(row['match_score'])}% Match</div>
                    </div>
                    <div style="margin-top:20px;">
                        <span style="font-size:0.8rem; font-weight:bold; opacity:0.6;">REQUIRED SKILLS:</span><br>
                        <code>{row['skills']}</code>
                    </div>
                    <div class="gap-alert">
                        ⚠️ SKILL GAP: {row['gap']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Data-Driven Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(df, names='location', title="Job Distribution", hole=0.5), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(results[results['match_score']>0], x='company', y='match_score', title="Match Strength Score"), use_container_width=True)

else:
    st.info("👋 Welcome! Provide your skills and preferred job title in the sidebar to activate the AI Matching Engine.")
    st.plotly_chart(px.histogram(df, x='location', color='type', title="Current Market Availability"), use_container_width=True)

st.markdown("---")
st.caption("BS AI Semester 4 | Instructor: Sir Abdul Haseeb | Aror University Sukkur")

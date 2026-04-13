# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (ENHANCED)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Logic       : TF-IDF + Cosine Similarity + Skill Gap Analysis
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ──────────────────────────────────────────────────────────────
#  1. THEME & SESSION MANAGEMENT
# ──────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# UI Color Logic
if st.session_state.theme == 'dark':
    bg, text, card, border = "#0F172A", "#F1F5F9", "#1E293B", "#334155"
    accent = "#38BDF8"
else:
    bg, text, card, border = "#F8FAFC", "#0F172A", "#FFFFFF", "#E2E8F0"
    accent = "#0284C7"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; transition: 0.3s; }}
    .job-card {{
        background-color: {card}; border: 1px solid {border};
        padding: 25px; border-radius: 16px; margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    .match-pct {{ color: {accent}; font-weight: 800; font-size: 1.2rem; }}
    .gap-text {{ color: #EF4444; font-size: 0.85rem; font-weight: 600; }}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. ENHANCED DATASET (Pakistan Student Market)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_comprehensive_db():
    data = [
        {"title": "AI Intern", "company": "Systems Ltd", "location": "Lahore", "type": "Internship", "skills": "Python, Machine Learning, NLP, Pandas", "desc": "Assisting in fine-tuning LLMs and data cleaning."},
        {"title": "Junior Data Analyst", "company": "Symmetry Group", "location": "Sukkur", "type": "Full-Time", "skills": "SQL, Python, PowerBI, Statistics, Excel", "desc": "Data visualization and reporting for local logistics."},
        {"title": "Web Developer", "company": "10Pearls", "location": "Karachi", "type": "Internship", "skills": "React, JavaScript, CSS, HTML, Git", "desc": "Frontend optimization for high-traffic web apps."},
        {"title": "Cloud Associate", "company": "NetSol", "location": "Islamabad", "type": "Full-Time", "skills": "AWS, Docker, Linux, Python, Bash", "desc": "Automating cloud infrastructure and deployments."},
        {"title": "ML Engineer", "company": "Folio3", "location": "Karachi", "type": "Full-Time", "skills": "Python, PyTorch, Computer Vision, Docker", "desc": "Integrating CV models into mobile applications."}
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. NLP ENGINE & SKILL GAP LOGIC
# ──────────────────────────────────────────────────────────────
def run_matching_engine(user_query, df):
    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    combined_content = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(combined_content.apply(lambda x: x.lower()))
    user_vec = tfidf.transform([user_query.lower()])
    
    # Cosine Similarity
    scores = cosine_similarity(user_vec, matrix).flatten()
    df['match_score'] = scores * 100
    
    # Skill Gap Analysis (Set Theory)
    def find_gap(row_skills):
        req = set([s.strip().lower() for s in row_skills.split(',')])
        user = set([s.strip().lower() for s in re.split(r'[,\s]+', user_query)])
        missing = req - user
        return ", ".join(missing).title() if missing else "None"

    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR & NAVIGATION
# ──────────────────────────────────────────────────────────────
df = load_comprehensive_db()

with st.sidebar:
    st.title("TalentMatch AI")
    st.button("🌓 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.markdown("---")
    st.header("Candidate Profile")
    u_skills = st.text_area("Input Your Skills", placeholder="e.g. Python, SQL, React", height=150)
    u_role = st.text_input("Target Role", placeholder="e.g. Data Scientist")
    u_city = st.multiselect("Preferred Cities", df['location'].unique(), default=df['location'].unique())
    
    run_btn = st.button("Analyze & Match", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────
#  5. MAIN INTERFACE
# ──────────────────────────────────────────────────────────────
st.title("Career Intelligence Dashboard")
st.write(f"**Authors:** Waqaas Hussain & Hira Abdul Hafeez | **Aror University Sukkur**")

if run_btn and u_skills:
    # Processing
    results = run_matching_engine(f"{u_role} {u_skills}", df)
    results = results[results['location'].isin(u_city)]
    
    # Tabs for Organization
    tab1, tab2 = st.tabs(["🎯 Job Matches", "📊 Market Insights"])
    
    with tab1:
        st.subheader("Personalized Recommendations")
        for _, row in results.iterrows():
            if row['match_score'] > 0:
                with st.container():
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div>
                                <h3 style="margin:0;">{row['title']}</h3>
                                <p style="margin:0; opacity:0.7;">{row['company']} • {row['location']}</p>
                            </div>
                            <div class="match-pct">{int(row['match_score'])}% Match</div>
                        </div>
                        <div style="margin-top:15px;">
                            <span style="font-size:0.8rem; font-weight:bold; opacity:0.6;">REQUIRED:</span><br>
                            <code>{row['skills']}</code>
                        </div>
                        <div class="gap-text" style="margin-top:10px;">
                            ⚠️ Missing: {row['gap']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(df, names='location', title="Job Distribution by City", hole=0.4)
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(results[results['match_score']>0], x='company', y='match_score', 
                          title="Match Strength per Company", color_discrete_sequence=[accent])
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👋 Welcome! Input your skills in the sidebar and click 'Analyze' to begin.")
    # Show generic stats when idle
    st.plotly_chart(px.histogram(df, x='location', title="Market Availability Overview"), use_container_width=True)

st.markdown("---")
st.caption("BS AI Semester 4 Final Project | Instructor: Sir Abdul Haseeb")

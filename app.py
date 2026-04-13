# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (OUTLINE ALIGNED)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Sir Abdul Haseeb)
#   CLO Alignment: CLO-1 (Python), CLO-3 (ML & Data Science)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
#  1. GLOBAL THEME & UI (Streamlit Deployment - Week 14)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI Pro", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'

if st.session_state.theme == 'dark':
    bg, text, card, accent = "#0F172A", "#F1F5F9", "#1E293B", "#10B981"
else:
    bg, text, card, accent = "#F8FAFC", "#1E293B", "#FFFFFF", "#059669"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; transition: 0.3s; font-family: 'Inter', sans-serif; }}
    .job-card {{ background: {card}; border: 1px solid #334155; padding: 25px; border-radius: 15px; margin-bottom: 20px; }}
    .hero-stat {{ background: linear-gradient(45deg, #064e3b, #065f46); padding: 20px; border-radius: 12px; text-align: center; color: white; }}
    .match-val {{ color: {accent}; font-weight: 800; font-size: 1.3rem; }}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATASET & CLEANING (Pandas - Week 07 & 08)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_cleaned_data():
    # Demonstrating Week 03: Dictionaries/Lists
    raw_data = [
        {"id": 1, "title": "AI Engineer", "company": "Systems Ltd", "location": "Lahore", "salary": 250000, "skills": "Python, Machine Learning, NLP, Scikit-learn"},
        {"id": 2, "title": "Data Analyst", "company": "Afiniti", "location": "Karachi", "salary": 180000, "skills": "SQL, Python, Statistics, PowerBI"},
        {"id": 3, "title": "Web Developer", "company": "Aror Solutions", "location": "Sukkur", "salary": 90000, "skills": "React, JavaScript, HTML, CSS, Django"},
        {"id": 4, "title": "MLOps Intern", "company": "NetSol", "location": "Islamabad", "salary": 35000, "skills": "Docker, Kubernetes, AWS, Python, Linux"},
        {"id": 5, "title": "Junior AI Dev", "company": "Folio3", "location": "Karachi", "salary": 120000, "skills": "Python, Computer Vision, Git, PyTorch"},
    ]
    df = pd.DataFrame(raw_data)
    # Week 08 L1: Data Cleaning (Handling strings)
    df['skills'] = df['skills'].apply(lambda x: x.lower())
    return df

# ──────────────────────────────────────────────────────────────
#  3. THE ML ENGINE (TF-IDF & Cosine Similarity - Week 10-12)
# ──────────────────────────────────────────────────────────────
class RecommenderEngine: # Week 06 L1: Classes & Objects
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words='english') # Week 07 L2: Sklearn

    def get_recommendations(self, user_profile):
        # Week 10 L1: Similarity Logic
        content = self.df['title'] + " " + self.df['skills']
        matrix = self.vectorizer.fit_transform(content)
        user_vec = self.vectorizer.transform([user_profile.lower()])
        
        scores = cosine_similarity(user_vec, matrix).flatten()
        self.df['match_score'] = scores * 100
        
        # Week 03 L1: Comprehensions (Finding missing skills)
        user_list = set(re.split(r'[,\s]+', user_profile.lower()))
        def get_gap(row_skills):
            req = set([s.strip() for s in row_skills.split(',')])
            return list(req - user_list)
        
        self.df['gap'] = self.df['skills'].apply(get_gap)
        return self.df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION & INPUTS
# ──────────────────────────────────────────────────────────────
df = get_cleaned_data()
engine = RecommenderEngine(df)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.title("TalentMatch AI")
    st.button("🌓 Switch Theme", on_click=toggle_theme)
    st.markdown("---")
    
    # User Profile (CLO-1)
    st.header("Candidate Profile")
    name = st.text_input("Full Name", "Waqaas Hussain")
    u_skills = st.text_area("Your Technical Skills", placeholder="e.g. Python, SQL, Statistics")
    u_city = st.selectbox("Preferred City", ["Any"] + list(df['location'].unique()))
    
    # Week 04 L1: Functions (Triggering Search)
    search_btn = st.button("Generate AI Match", type="primary")

# ──────────────────────────────────────────────────────────────
#  5. MAIN DASHBOARD (Visualization - Week 08 & 09)
# ──────────────────────────────────────────────────────────────
st.title("AI Career Intelligence Hub")
st.caption(f"Aror University Sukkur | Semester 4 | Instructor: Sir Abdul Haseeb")

# Hero Section (Week 08: Matplotlib Visualization)
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="hero-stat"><h5>Market Jobs</h5><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="hero-stat"><h5>Avg Salary (PKR)</h5><h2>136K</h2></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="hero-stat"><h5>Top Domain</h5><h2>AI/ML</h2></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if search_btn and u_skills:
    # Part A: The Results (Week 10-12 Logic)
    results = engine.get_recommendations(u_skills)
    if u_city != "Any":
        results = results[results['location'] == u_city]
    
    tab1, tab2 = st.tabs(["🎯 AI Matches", "📊 Market Analytics"])
    
    with tab1:
        for _, row in results.iterrows():
            if row['match_score'] > 0:
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0;">{row['title']}</h3>
                        <span class="match-val">{int(row['match_score'])}% Compatibility</span>
                    </div>
                    <p style="margin:5px 0; opacity:0.8;"><b>{row['company']}</b> • {row['location']}</p>
                    <div style="margin-top:10px;">
                        <span style="font-size:0.85rem; color:#64748b;"><b>Required Skills:</b> {row['skills']}</span>
                    </div>
                    <div style="margin-top:10px; color:#ef4444; font-size:0.85rem;">
                        ⚠️ <b>Skill Gap:</b> {', '.join(row['gap']).title() if row['gap'] else 'None! Ready to apply.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        # Week 09: Seaborn Visualization
        st.subheader("Data Science Insights")
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Matplotlib Plot
            fig, ax = plt.subplots()
            sns.barplot(data=df, x='location', y='salary', ax=ax, palette='viridis')
            ax.set_title("Salary Distribution by City")
            st.pyplot(fig)
            
        with col_right:
            # Seaborn Heatmap / Pairplot equivalent
            fig2, ax2 = plt.subplots()
            df_numeric = df[['salary']].copy()
            sns.heatmap(df_numeric.corr(), annot=True, ax=ax2)
            st.pyplot(fig2)

else:
    # Week 10 L1: Background Visualization
    st.info("👋 Input your profile skills in the sidebar to run the Recommendation Engine.")
    fig_hist = px.histogram(df, x='location', y='salary', color='company', barmode='group', title="Market Overview")
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("BS AI | Semester 4 Project | Demonstrating Week 02-14 Syllabus Concepts")

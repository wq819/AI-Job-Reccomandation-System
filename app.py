# ============================================================
#   TALENTMATCH AI: ENTERPRISE RECRUITMENT DASHBOARD
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Logic       : TF-IDF Vectorization + Cosine Similarity
#   Semester    : 4th (BS Artificial Intelligence)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ──────────────────────────────────────────────────────────────
#  1. PREMIUM UI CONFIGURATION & CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI Pro", layout="wide", page_icon="🎯")

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'

# UI Colors based on selection
if st.session_state.theme == 'dark':
    bg, text, card, border, accent = "#0F172A", "#F1F5F9", "#1E293B", "#334155", "#10B981"
else:
    bg, text, card, border, accent = "#F8FAFC", "#1E293B", "#FFFFFF", "#E2E8F0", "#059669"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    .stApp {{ background-color: {bg}; color: {text}; transition: 0.3s; }}
    
    /* Hero Stats */
    .stat-card {{
        background: {card}; border: 1px solid {border};
        padding: 20px; border-radius: 15px; text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    .stat-val {{ color: {accent}; font-size: 1.8rem; font-weight: 800; }}
    
    /* Job Card */
    .job-card {{
        background-color: {card}; border: 1px solid {border};
        padding: 25px; border-radius: 20px; margin-bottom: 20px;
        transition: 0.3s; display: flex; align-items: center; gap: 20px;
    }}
    .job-card:hover {{ transform: translateY(-5px); border-color: {accent}; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }}
    
    .company-logo {{
        width: 70px; height: 70px; border-radius: 15px;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, {accent}, #3B82F6); color: white;
        font-size: 1.5rem; font-weight: bold;
    }}
    
    .match-tag {{
        background: rgba(16, 185, 129, 0.1); color: {accent};
        padding: 5px 15px; border-radius: 50px; font-weight: 700; font-size: 0.8rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATASET & AI LOGIC
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_enterprise_db():
    data = [
        {"title": "AI Research Engineer", "company": "Systems Ltd", "location": "Lahore", "icon": "SL", "skills": "Python, NLP, PyTorch, Transformers", "desc": "Leading global AI innovation projects."},
        {"title": "Senior Data Scientist", "company": "Afiniti", "location": "Karachi", "icon": "AF", "skills": "SQL, Statistics, Python, ML, AWS", "desc": "Behavioral matching for enterprise clients."},
        {"title": "Machine Learning Intern", "company": "Folio3", "location": "Sukkur", "icon": "F3", "skills": "Python, Computer Vision, OpenCV, Git", "desc": "Developing AI for AgTech solutions."},
        {"title": "Full Stack Dev (AI)", "company": "10Pearls", "location": "Islamabad", "icon": "10P", "skills": "React, Node.js, Python, OpenAI", "desc": "Building next-gen AI interfaces."},
        {"title": "Cloud Architect", "company": "NetSol", "location": "Lahore", "icon": "NS", "skills": "Docker, Kubernetes, AWS, Terraform", "desc": "Scaling financial tech infrastructure."},
    ]
    return pd.DataFrame(data)

def run_recommendation_engine(user_query, df):
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(corpus.apply(lambda x: x.lower()))
    user_vec = tfidf.transform([user_query.lower()])
    df['match_score'] = cosine_similarity(user_vec, matrix).flatten() * 100
    
    def get_gap(row_skills):
        req = set([s.strip().lower() for s in row_skills.split(',')])
        user = set([s.strip().lower() for s in re.split(r'[,\s]+', user_query)])
        gap = req - user
        return ", ".join(gap).title() if gap else "Ready!"
        
    df['gap'] = df['skills'].apply(get_gap)
    return df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  3. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
df = load_enterprise_db()

with st.sidebar:
    st.markdown(f"<h1 style='color:{accent};'>TalentMatch Pro</h1>", unsafe_allow_html=True)
    st.button("🌓 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.markdown("---")
    st.subheader("👤 User Profile")
    u_skills = st.text_area("Skill Set", placeholder="e.g. Python, SQL, NLP", height=150)
    u_role = st.text_input("Target Position", placeholder="e.g. AI Engineer")
    run_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Developed by Waqaas & Hira\nAror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  4. MAIN DASHBOARD UI
# ──────────────────────────────────────────────────────────────
st.title("AI Career Intelligence Hub")

# Hero Stats Section
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f'<div class="stat-card"><div class="stat-label">Market Openings</div><div class="stat-val">{len(df)}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-card"><div class="stat-label">Top City</div><div class="stat-val">Karachi</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-card"><div class="stat-label">AI Demand</div><div class="stat-val">High</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="stat-card"><div class="stat-label">Avg. Stipend</div><div class="stat-val">35K</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if run_btn and u_skills:
    results = run_recommendation_engine(f"{u_role} {u_skills}", df)
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.subheader("🎯 Best Matched Opportunities")
        for _, row in results.iterrows():
            if row['match_score'] > 0:
                st.markdown(f"""
                <div class="job-card">
                    <div class="company-logo">{row['icon']}</div>
                    <div style="flex-grow: 1;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="margin:0;">{row['title']}</h3>
                            <span class="match-tag">{int(row['match_score'])}% AI Match</span>
                        </div>
                        <p style="margin:2px 0; opacity:0.7; font-weight:600;">{row['company']} • {row['location']}</p>
                        <div style="margin-top:10px; font-size:0.85rem;">
                            <span style="opacity:0.6;">Required:</span> <code>{row['skills']}</code>
                        </div>
                        <div style="margin-top:5px; font-size:0.85rem; color:#EF4444;">
                            <b>Gap:</b> {row['gap']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        st.subheader("📊 Market Distribution")
        st.plotly_chart(px.pie(df, names='location', hole=0.6, color_discrete_sequence=[accent, "#3B82F6", "#6366F1"]), use_container_width=True)
        st.plotly_chart(px.bar(results[results['match_score']>0], x='company', y='match_score', title="Match Accuracy by Company"), use_container_width=True)

else:
    st.info("👋 Input your professional profile on the left to start the AI recommendation engine.")
    # Default Analytics
    st.plotly_chart(px.histogram(df, x='location', color='company', title="Active Tech Hubs Across Pakistan"), use_container_width=True)

st.markdown("---")
st.caption("BS AI Semester 4 | Instructor: Sir Abdul Haseeb | Aror University Sukkur")

# ============================================================
#   TALENTMATCH AI: PREMIUM SOLO CAPSTONE
#   Institution : Aror University Sukkur
#   Student     : Waqaas Hussain (SAP-5000000291)
#   Subject     : Programming for AI (Sir Abdul Haseeb)
#   Semester    : 4th (BS Artificial Intelligence)
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
#  1. PREMIUM UI CONFIGURATION (Week 14 Deployment)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI Pro", layout="wide", page_icon="🎯")

# Emerald & Dark Slate Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #064e3b, #022c22, #000000);
        color: #ecfdf5;
    }
    
    /* Premium Glassmorphism Cards */
    .job-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 24px;
        margin-bottom: 25px;
        transition: 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .job-card:hover {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid #10b981;
        transform: translateY(-8px);
    }
    
    .match-val {
        background: linear-gradient(135deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 1.6rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        width: 100% !important;
    }
    
    code { color: #34d399 !important; background: rgba(16, 185, 129, 0.1) !important; padding: 4px 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. ENHANCED DATASET (Week 07 & 08: Sklearn & Pandas)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = [
        {"title": "AI Research Engineer", "company": "Systems Ltd", "location": "Lahore", "salary": 280000, "skills": "Python, NLP, PyTorch, Transformers, Scikit-learn", "desc": "Leading global AI innovation and neural architecture research."},
        {"title": "Senior Data Scientist", "company": "Afiniti", "location": "Karachi", "salary": 350000, "skills": "SQL, Python, Statistics, Machine Learning, AWS, Big Data", "desc": "Advanced behavioral matching algorithms for enterprise scale."},
        {"title": "AI Developer Intern", "company": "Aror Solutions", "location": "Sukkur", "salary": 25000, "skills": "Python, React, API, Git, Data Structures", "desc": "Developing AI integration tools for local Sindh industries."},
        {"title": "Cloud Architect", "company": "NetSol", "location": "Islamabad", "salary": 310000, "skills": "AWS, Docker, Kubernetes, Linux, Python, Terraform", "desc": "Scaling financial cloud ecosystems globally."},
        {"title": "Junior ML Engineer", "company": "Folio3", "location": "Karachi", "salary": 125000, "skills": "Python, Computer Vision, Git, Django, OpenCV", "desc": "Building Computer Vision models for AgTech solutions."},
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. THE AI ENGINE (Week 10: Similarity Measures)
# ──────────────────────────────────────────────────────────────
def run_ai_logic(input_text, df):
    # Week 08 L1: Data Preprocessing (Cleaning)
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
    # Week 10: TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    # Combining features for a dense profile
    corpus = df['title'] + " " + df['skills'] + " " + df['desc']
    tfidf_matrix = tfidf.fit_transform(corpus.apply(clean))
    
    # Transforming input (Quick Skills or Full CV)
    user_vec = tfidf.transform([clean(input_text)])
    
    # Week 10-11: Evaluation Metrics (Cosine Similarity)
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['score'] = scores * 100
    
    # Week 03: List Comprehensions (Skill Gap Analysis)
    user_words = set(clean(input_text).split())
    def find_gap(row_skills):
        required = set([s.strip().lower() for s in row_skills.split(',')])
        gap = required - user_words
        return ", ".join(list(gap)).title() if gap else "None! You're a perfect match."
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR (Personalized Trigger & Dual-Input Mode)
# ──────────────────────────────────────────────────────────────
df_main = load_data()

with st.sidebar:
    st.markdown("<h1 style='color:#10b981;'>TalentMatch AI</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=70)
    st.markdown("---")
    
    st.subheader("👨‍🎓 Candidate Profile")
    u_name = st.text_input("Candidate Name", "Waqaas Hussain")
    
    # Enhanced Input Mode (CV vs Skills)
    input_mode = st.radio("Input Method", ["Quick Skill Entry", "Full Resume/CV Text"])
    
    if input_mode == "Quick Skill Entry":
        u_data = st.text_input("Technical Skills", placeholder="e.g. Python, SQL, ML")
    else:
        u_data = st.text_area("Paste Your CV Content", placeholder="Copy and paste your entire resume here...", height=250)
    
    u_loc = st.selectbox("Preferred City", ["Any City"] + list(df_main['location'].unique()))
    
    st.markdown("---")
    search_btn = st.button("Generate Recommendations")
    st.caption(f"Project by {u_name} | Aror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. DASHBOARD & EVALUATION (Week 08 & 09 Visualization)
# ──────────────────────────────────────────────────────────────
st.title("AI Career Intelligence Hub")
st.write(f"Instructor: **Sir Abdul Haseeb** | **Semester 4 Final Submission**")

if search_btn and u_data:
    results = run_ai_logic(u_data, df_main)
    
    if u_loc != "Any City":
        results = results[results['location'] == u_loc]
        
    tab_res, tab_vis = st.tabs(["🎯 Top Matches", "📊 Market Analytics"])
    
    with tab_res:
        st.subheader(f"Ranked Matches for {u_name}")
        for _, row in results.iterrows():
            if row['score'] > 2: # Threshold for relevance
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div>
                            <h2 style="margin:0; color:#10b981;">{row['title']}</h2>
                            <p style="margin:0; opacity:0.8; font-weight:600;">{row['company']} • {row['location']}</p>
                        </div>
                        <div class="match-val">{int(row['score'])}% Match</div>
                    </div>
                    <p style="margin-top:15px; font-size:0.95rem; color:#d1d5db; line-height:1.6;">{row['desc']}</p>
                    <div style="margin-top:20px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:15px;">
                        <span style="font-size:0.85rem; color:#94a3b8; font-weight:bold; letter-spacing:1px;">⚠️ MISSING SKILLS:</span><br>
                        <span style="color:#f87171; font-weight:600; font-size:0.9rem;">{row['gap']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_vis:
        # Week 09: Seaborn Visualization
        st.subheader("Industry Insights & Salary Benchmarks")
        col_l, col_r = st.columns(2)
        
        with col_l:
            fig, ax = plt.subplots(facecolor='none')
            sns.barplot(data=df_main, x='location', y='salary', palette='Greens_d', ax=ax)
            ax.set_title("Salary Trends (PKR)", color='white', fontsize=14, weight='bold')
            ax.tick_params(colors='white')
            st.pyplot(fig)
            
        with col_r:
            fig2, ax2 = plt.subplots()
            sns.lineplot(data=df_main, x='location', y='salary', marker='o', color='#10b981', ax=ax2)
            ax2.set_title("Market Demand Density", color='white', fontsize=14, weight='bold')
            st.pyplot(fig2)
else:
    # Default State Visualization
    st.info("👋 Welcome! Use the sidebar to analyze your profile and find AI-based recommendations.")
    st.markdown("### Pakistan Tech Market Snapshot")
    st.bar_chart(df_main.groupby('location')['salary'].mean())

st.markdown("---")
st.caption("BS AI Semester 4 | Aror University Sukkur | Final Solo Project")

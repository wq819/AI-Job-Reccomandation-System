# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (SOLO PRO)
#   Institution : Aror University Sukkur
#   Student     : Waqaas Hussain (SAP-5000000291)
#   Logic       : TF-IDF Vectorization + Cosine Similarity
#   Subject     : Programming for AI (Sir Abdul Haseeb)
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
#  1. GUI CONFIGURATION (Streamlit Week 14)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI Pro", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f8fafc; }
    .job-card {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 6px solid #10B981; margin-bottom: 20px;
    }
    .match-score { color: #10B981; font-weight: 800; font-size: 1.2rem; }
    .stButton>button { background: #10B981!important; color: white!important; border-radius: 10px!important; }
    .cv-highlight { background: #ecfdf5; border: 1px dashed #10B981; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATASET & PREPROCESSING (Week 07 & 08)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean_data():
    data = [
        {"title": "AI Research Engineer", "company": "Systems Ltd", "location": "Lahore", "salary": 250000, "skills": "Python, Machine Learning, NLP, Scikit-learn, TensorFlow"},
        {"title": "Senior Data Scientist", "company": "Afiniti", "location": "Karachi", "salary": 380000, "skills": "SQL, Python, Statistics, Machine Learning, R, Big Data"},
        {"title": "Machine Learning Intern", "company": "Aror Solutions", "location": "Sukkur", "salary": 25000, "skills": "Python, React, API, Git, Data Analysis"},
        {"title": "Cloud Architect", "company": "NetSol", "location": "Islamabad", "salary": 320000, "skills": "AWS, Docker, Kubernetes, Linux, Python, DevOps"},
        {"title": "Junior ML Developer", "company": "Folio3", "location": "Karachi", "salary": 140000, "skills": "Python, Computer Vision, Git, PyTorch, Django"},
        {"title": "Software Engineer (AI)", "company": "Contour Software", "location": "Lahore", "salary": 210000, "skills": "C++, Python, Algorithms, SQL, Data Structures"}
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. AI CORE: TF-IDF & SIMILARITY (Week 10-12)
# ──────────────────────────────────────────────────────────────
def run_recommender(user_profile, df):
    # Preprocessing (Proposal Step 2)
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
    # Feature Extraction (Proposal Step 3)
    tfidf = TfidfVectorizer(stop_words='english')
    job_content = df['title'] + " " + df['skills'] + " " + df['desc'] if 'desc' in df else df['title'] + " " + df['skills']
    tfidf_matrix = tfidf.fit_transform(job_content.apply(clean))
    
    # Matching User Input
    user_vec = tfidf.transform([clean(user_profile)])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['match_score'] = scores * 100
    
    # Skill Gap Detection (Week 03 Logic)
    user_words = set(clean(user_profile).split())
    def find_gap(row_skills):
        req = set([s.strip().lower() for s in row_skills.split(',')])
        gap = req - user_words
        return ", ".join(list(gap)).title() if gap else "Ready to Apply!"
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR & INPUTS (CV OPTION INCLUDED)
# ──────────────────────────────────────────────────────────────
df_main = load_and_clean_data()

with st.sidebar:
    st.markdown("<h2 style='color:#10B981;'>TalentMatch AI</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.markdown("---")
    
    st.subheader("📄 Your Profile Details")
    u_name = st.text_input("Name", placeholder="Waqaas Hussain")
    
    # NEW OPTION: CV/Resume Text Input
    st.markdown("**Option 1: Quick Skills**")
    u_skills = st.text_input("List Skills", placeholder="e.g. Python, SQL")
    
    st.markdown("**Option 2: CV Analyzer**")
    u_cv = st.text_area("Paste Full CV/Resume Text", placeholder="Paste all text from your CV here...", height=150)
    
    # Decision logic for input
    final_input = u_cv if u_cv else u_skills
    
    u_loc = st.selectbox("Preferred City", ["Any"] + list(df_main['location'].unique()))
    
    st.markdown("---")
    search_btn = st.button("Generate AI Match", type="primary", use_container_width=True)
    st.caption("Solo Project: Waqaas Hussain\nAror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. MAIN DASHBOARD (Week 08 & 09 Visualization)
# ──────────────────────────────────────────────────────────────
st.title("AI-Driven Career Intelligence Dashboard")
st.write(f"Instructor: **Sir Abdul Haseeb** | Semester 4 BS AI")

if search_btn and final_input:
    # Processing Recommendations
    results = run_recommender(final_input, df_main)
    if u_loc != "Any":
        results = results[results['location'] == u_loc]
    
    # Results Display
    tab1, tab2 = st.tabs(["🎯 Job Matches", "📊 Market Trends"])
    
    with tab1:
        st.subheader(f"Best Matches for {u_name}")
        if u_cv:
            st.markdown('<div class="cv-highlight">🔍 <b>CV Analysis Mode:</b> Extracting key professional features from your resume...</div>', unsafe_allow_html=True)
        
        for _, row in results.iterrows():
            if row['match_score'] > 0:
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0;">{row['title']}</h3>
                        <span class="match-score">{int(row['match_score'])}% Compatibility</span>
                    </div>
                    <p style="margin:5px 0; color:#64748b;"><b>{row['company']}</b> • {row['location']}</p>
                    <div style="margin-top:10px; color:#ef4444; font-size:0.85rem; font-weight:600;">
                        ⚠️ Skill Gap: {row['gap']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Industry Insights (Week 09: Seaborn)")
        col_a, col_b = st.columns(2)
        with col_a:
            fig1, ax1 = plt.subplots()
            sns.barplot(data=df_main, x='location', y='salary', palette='viridis', ax=ax1)
            ax1.set_title("Salary Benchmarks by City")
            st.pyplot(fig1)
        with col_b:
            fig2, ax2 = plt.subplots()
            sns.lineplot(data=df_main, x='location', y='salary', marker='o', color='#10B981', ax=ax2)
            ax2.set_title("Market Demand Growth")
            st.pyplot(fig2)

else:
    st.info("👋 Welcome! Either list your skills or paste your full CV text in the sidebar to begin.")
    st.markdown("### National Tech Overview")
    st.bar_chart(df_main.groupby('location')['salary'].mean())

st.markdown("---")
st.caption("BS AI Semester 4 | Final Solo Project Submission")

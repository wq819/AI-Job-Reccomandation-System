# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (FINAL)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Sir Abdul Haseeb)
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
#  1. GUI CONFIGURATION (Week 14: Streamlit)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI Pro", layout="wide", page_icon="🎯")

# Professional Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }
    .job-card {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 5px solid #059669; margin-bottom: 20px;
    }
    .match-score { color: #059669; font-weight: 800; font-size: 1.2rem; }
    .stat-box {
        background: #064e3b; color: white; padding: 20px;
        border-radius: 12px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATA COLLECTION & PREPROCESSING (Week 07 & 08)
# ──────────────────────────────────────────────────────────────
class JobDataProcessor: # Week 06: Classes & Objects
    @staticmethod
    @st.cache_data
    def load_and_clean():
        # Step 1: Data Collection (Simulated CSV/Kaggle Dataset)
        data = [
            {"title": "AI Research Intern", "company": "Systems Ltd", "location": "Lahore", "salary": 35000, "skills": "Python, Machine Learning, NLP, Scikit-learn", "desc": "Assisting in model training and data preprocessing."},
            {"title": "Junior Data Scientist", "company": "Afiniti", "location": "Karachi", "salary": 110000, "skills": "SQL, Python, Statistics, Machine Learning, R", "desc": "Behavioral matching AI for global interaction optimization."},
            {"title": "AI Developer Intern", "company": "Aror Solutions", "location": "Sukkur", "id": 3, "salary": 20000, "skills": "Python, React, API Integration, Git", "desc": "Building AI-integrated web tools for regional industry."},
            {"title": "Cloud Solutions Associate", "company": "NetSol", "location": "Islamabad", "salary": 95000, "skills": "AWS, Docker, Kubernetes, Linux, Python", "desc": "Designing scalable cloud architectures for automotive software."},
            {"title": "Junior ML Engineer", "company": "Folio3", "location": "Karachi", "salary": 85000, "skills": "Python, Computer Vision, Git, PyTorch, Django", "desc": "Developing backends for AI-driven mobile applications."}
        ]
        df = pd.DataFrame(data)
        # Step 2: Data Preprocessing (Cleaning & Normalization)
        df['cleaned_skills'] = df['skills'].apply(lambda x: x.lower()) # Lambda/Map Week 04
        return df

# ──────────────────────────────────────────────────────────────
#  3. MODEL DESIGN: TF-IDF & SIMILARITY (Week 10: KNN/Similarity)
# ──────────────────────────────────────────────────────────────
def run_matching_engine(user_query, df):
    # Step 3: Feature Extraction using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    # Combine relevant columns for content-based filtering
    job_content = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(job_content.apply(lambda x: x.lower()))
    
    # Vectorize User Profile
    user_vec = tfidf.transform([user_query.lower()])
    
    # Evaluate System Performance using Cosine Similarity
    scores = cosine_similarity(user_vec, matrix).flatten()
    df['match_score'] = scores * 100
    
    # Logic for Skill Gap Analysis (Comprehensions Week 03)
    user_set = set(re.split(r'[,\s]+', user_query.lower()))
    def find_gap(row_skills):
        req = set([s.strip().lower() for s in row_skills.split(',')])
        return list(req - user_set)
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR & NAVIGATION (Step 4: Implementation)
# ──────────────────────────────────────────────────────────────
df_main = JobDataProcessor.load_and_clean()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.title("TalentMatch AI")
    st.markdown("---")
    
    # User Input Collection
    st.subheader("Your AI Profile")
    u_name = st.text_input("Full Name", placeholder="Alex Johnson")
    u_skills = st.text_area("Technical Skills", placeholder="e.g. Python, SQL, Statistics")
    u_loc = st.selectbox("Target City", ["Any"] + list(df_main['location'].unique()))
    
    search_btn = st.button("Generate Recommendations", type="primary")
    
    st.markdown("---")
    st.write("**Authors:**")
    st.caption("Waqaas Hussain & Hira Abdul Hafeez")
    st.caption("BS AI - 4th Semester")

# ──────────────────────────────────────────────────────────────
#  5. TESTING & EVALUATION (Step 5: Testing)
# ──────────────────────────────────────────────────────────────
st.title("AI Career Intelligence Hub")
st.write(f"Instructor: **Sir Abdul Haseeb** | Aror University Sukkur")

# Dashboard Analytics (Week 08 & 09: Visualization)
c1, c2, c3 = st.columns(3)
with c1: st.markdown(f'<div class="stat-box"><h5>Total Jobs</h5><h2>{len(df_main)}</h2></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><h5>AI Intensity</h5><h2>High</h2></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><h5>Top Region</h5><h2>Sindh/Punjab</h2></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if search_btn and u_skills:
    results = run_matching_engine(u_skills, df_main)
    
    # Filter by Location
    if u_loc != "Any":
        results = results[results['location'] == u_loc]
    
    tab1, tab2 = st.tabs([" Job Reccomandation System ", "📊 Market Trends"])
    
    with tab1:
        st.subheader(f"Recommendations for {u_name}")
        for _, row in results.iterrows():
            if row['match_score'] > 0:
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0;">{row['title']}</h3>
                        <span class="match-score">{int(row['match_score'])}% Match</span>
                    </div>
                    <p style="margin:5px 0; color:#64748b;"><b>{row['company']}</b> • {row['location']}</p>
                    <p style="font-size:0.9rem; margin-top:10px;">{row['desc']}</p>
                    <div style="margin-top:10px; color:#ef4444; font-size:0.85rem; font-weight:600;">
                         Skill Gap: {', '.join(row['gap']).title() if row['gap'] else 'None! Perfect Fit.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        # Week 09: Seaborn and Matplotlib Visualization
        st.subheader("Data Science Metrics")
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig1, ax1 = plt.subplots()
            sns.barplot(data=df_main, x='location', y='salary', palette='Greens_d', ax=ax1)
            ax1.set_title("Salary Benchmarks by City")
            st.pyplot(fig1)
            
        with col_b:
            fig2, ax2 = plt.subplots()
            # Heatmap of correlation (Week 09: Seaborn II)
            sns.heatmap(df_main[['salary']].corr(), annot=True, cmap='Greens', ax=ax2)
            ax2.set_title("Feature Correlation")
            st.pyplot(fig2)

else:
    st.info("👋 Welcome! Fill in your skills in the sidebar to see the Recommendation Engine in action.")
    # Default Visualization
    fig_hist = px.histogram(df_main, x='location', color='company', title="Job Distribution Across Pakistan")
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("BS AI Semester 4 | Programming for AI | Final Project Submission")

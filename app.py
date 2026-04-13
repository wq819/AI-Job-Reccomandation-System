# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM 
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
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
#  1. GUI DESIGN 
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI JOB RECCOMANDATION SYSTEM ", layout="wide", page_icon="")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f8fafc; }
    .job-card {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 6px solid #10B981; margin-bottom: 20px;
        transition: 0.3s ease;
    }
    .job-card:hover { transform: translateY(-5px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
    .match-score { color: #10B981; font-weight: 800; font-size: 1.2rem; }
    .stButton>button { background: #10B981!important; color: white!important; border-radius: 10px!important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATA COLLECTION & PREPROCESSING 
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_job_dataset():
    # Simulated Dataset (Can be replaced with pd.read_csv('jobs.csv'))
    data = [
        {"id": 1, "title": "AI Research Intern", "company": "Systems Ltd", "location": "Lahore", "salary": 35000, "skills": "Python, Machine Learning, NLP, Scikit-learn", "desc": "Assisting in neural network training and data cleaning."},
        {"id": 2, "title": "Junior Data Scientist", "company": "Afiniti", "location": "Karachi", "salary": 110000, "skills": "SQL, Python, Statistics, Machine Learning, R", "desc": "Behavioral matching AI for global customer interaction."},
        {"id": 3, "title": "AI Web Developer", "company": "Aror Solutions", "location": "Sukkur", "salary": 25000, "skills": "Python, React, API Integration, Git, HTML", "desc": "Building AI-integrated web tools for the regional industry in Sindh."},
        {"id": 4, "title": "Cloud Engineer", "company": "NetSol", "location": "Islamabad", "salary": 95000, "skills": "AWS, Docker, Kubernetes, Linux, Python", "desc": "Designing scalable cloud architectures for automotive software."},
        {"id": 5, "title": "Junior ML Engineer", "company": "Folio3", "location": "Karachi", "salary": 85000, "skills": "Python, Computer Vision, Git, PyTorch, Django", "desc": "Developing backends for AI-driven applications."}
    ]
    df = pd.DataFrame(data)
    return df

def clean_text(text):
    # Step 2: Data Preprocessing (Normalization & Cleaning)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# ──────────────────────────────────────────────────────────────
#  3. MODEL DESIGN: TF-IDF & SIMILARITY (Proposal Step 3)
# ──────────────────────────────────────────────────────────────
def run_ai_engine(user_query, df):
    # Step 3: Feature Extraction using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Content-Based Filtering: Combining metadata for matching
    job_content = df['title'] + " " + df['skills'] + " " + df['desc']
    tfidf_matrix = tfidf.fit_transform(job_content.apply(clean_text))
    
    # Vectorize User Input
    user_vec = tfidf.transform([clean_text(user_query)])
    
    # Evaluation: Cosine Similarity Measure
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['match_score'] = scores * 100
    
    # Skill Gap Identification (Additional Intelligence)
    user_skills = set(clean_text(user_query).split())
    def find_gap(row_skills):
        req = set(clean_text(row_skills).split())
        gap = req - user_skills
        return ", ".join(list(gap)).title() if gap else "Ready!"
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. IMPLEMENTATION & GUI 
# ──────────────────────────────────────────────────────────────
df_main = get_job_dataset()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.title("TalentMatch AI")
    st.markdown("---")
    
    st.subheader("Your Candidate Profile")
    u_name = st.text_input("Full Name", placeholder="Waqaas Hussain")
    u_skills = st.text_area("List Your Skills", placeholder="e.g. Python, SQL, Machine Learning")
    u_loc = st.selectbox("Preferred City", ["Any"] + list(df_main['location'].unique()))
    
    search_btn = st.button("Generate Recommendations", type="primary")
    
    st.markdown("---")
    st.caption("Developed by:\nWaqaas Hussain & Hira Abdul Hafeez\nAror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. RESULTS & EVALUATION 
# ──────────────────────────────────────────────────────────────
st.title("AI Career Intelligence Hub")
st.write(f"Course: **Programming for AI** | Instructor: **Sir Abdul Haseeb**")

# Market Analytics (Visualization - Week 09)
c1, c2, c3 = st.columns(3)
with c1: st.info(f"**Total Jobs Analyzed**\n# {len(df_main)}")
with c2: st.success(f"**Top Tech Hub**\nKarachi")
with c3: st.warning(f"**Average Match**\nDynamic")

st.markdown("<br>", unsafe_allow_html=True)

if search_btn and u_skills:
    results = run_ai_engine(u_skills, df_main)
    
    if u_loc != "Any":
        results = results[results['location'] == u_loc]
    
    tab1, tab2 = st.tabs([" AI Job Reccomandation system ", "📊 Market Visualization"])
    
    with tab1:
        st.subheader(f"Best Matches for {u_name}")
        for _, row in results.iterrows():
            if row['match_score'] > 0:
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3 style="margin:0;">{row['title']}</h3>
                        <span class="match-score">{int(row['match_score'])}% Compatibility</span>
                    </div>
                    <p style="margin:5px 0; color:#64748b;"><b>{row['company']}</b> • {row['location']}</p>
                    <p style="font-size:0.9rem; margin-top:10px;">{row['desc']}</p>
                    <div style="margin-top:10px; color:#EF4444; font-size:0.85rem; font-weight:600;">
                         Skill Gap: {row['gap']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("Industry Insights (Week 09: Seaborn/Matplotlib)")
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig1, ax1 = plt.subplots()
            sns.barplot(data=df_main, x='location', y='salary', palette='magma', ax=ax1)
            ax1.set_title("Salary Trends by City")
            st.pyplot(fig1)
            
        with col_b:
            fig2, ax2 = plt.subplots()
            sns.heatmap(df_main[['salary']].corr(), annot=True, cmap='coolwarm', ax=ax2)
            ax2.set_title("Feature Correlation")
            st.pyplot(fig2)
else:
    st.info(" Welcome! Fill in your skills in the sidebar and click the button to see AI recommendations.")
    # Show market map by default
    st.plotly_chart(px.histogram(df_main, x='location', title="Job Distribution Across Tech Hubs"), use_container_width=True)

st.markdown("---")
st.caption("BS AI Semester 4| ")

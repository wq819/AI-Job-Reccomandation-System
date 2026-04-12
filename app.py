# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (ACADEMIC VERSION)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Sir Abdul Haseeb)
#   Algorithm   : TF-IDF Vectorization + Cosine Similarity
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re

# ──────────────────────────────────────────────────────────────
#  1. PAGE CONFIG & BRANDING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Matcher | Aror University", layout="wide", page_icon="🎓")

# Professional Green & Grey Theme (Academic Look)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stButton>button { border-radius: 8px; background-color: #065f46; color: white; width: 100%; }
    
    .job-card {
        background: white; border-radius: 12px; border: 1px solid #e5e7eb;
        padding: 20px; margin-bottom: 20px; transition: 0.3s;
    }
    .job-card:hover { border-color: #059669; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    
    .badge-intern { background: #ecfdf5; color: #065f46; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.75rem; }
    .badge-match { background: #eff6ff; color: #1e40af; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.75rem; }
    .skill-missing { color: #b91c1c; font-size: 0.8rem; font-weight: 600; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATASET (Aligned with Student/Fresh Graduate Market)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_student_data():
    data = [
        {
            "id": 1, "title": "AI Research Intern", "company": "Systems Ltd", 
            "location": "Lahore", "type": "Internship", "salary": "Rs. 30,000",
            "skills": "Python, Machine Learning, Data Cleaning, Scikit-learn", 
            "desc": "Assist the AI team in preprocessing datasets and fine-tuning models.",
            "img": "https://images.unsplash.com/photo-1590059530472-87034f593322?q=80&w=600"
        },
        {
            "id": 2, "title": "Junior Data Analyst", "company": "Symmetry Group", 
            "location": "Sukkur", "type": "Fresh Graduate", "salary": "Rs. 60,000",
            "skills": "SQL, Excel, Python, PowerBI, Statistics", 
            "desc": "Perform exploratory data analysis and create visualization dashboards.",
            "img": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=600"
        },
        {
            "id": 3, "title": "Web Developer Intern", "company": "10Pearls", 
            "location": "Karachi", "type": "Internship", "salary": "Rs. 25,000",
            "skills": "HTML, CSS, JavaScript, React, Git", 
            "desc": "Help develop responsive UI components for international clients.",
            "img": "https://images.unsplash.com/photo-1568205706871-332308933220?q=80&w=600"
        },
        {
            "id": 4, "title": "Associate ML Engineer", "company": "Afiniti", 
            "location": "Islamabad", "type": "Fresh Graduate", "salary": "Rs. 85,000",
            "skills": "Python, PyTorch, Linux, API Integration, Docker", 
            "desc": "Junior role focusing on deploying ML models into production environments.",
            "img": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=600"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. NLP ENGINE (TF-IDF + Cosine Similarity)
# ──────────────────────────────────────────────────────────────
def run_match_engine(user_skills, df):
    if not user_skills.strip(): return pd.DataFrame()
    
    # Preprocessing & Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    combined_text = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(combined_text)
    user_vec = tfidf.transform([user_skills.lower()])
    
    # Calculate Similarity
    df['match_percent'] = cosine_similarity(user_vec, matrix).flatten() * 100
    
    # Simple Skill Gap Analysis
    def check_gap(req):
        user = user_skills.lower()
        missing = [s.strip() for s in req.split(',') if s.strip().lower() not in user]
        return ", ".join(missing) if missing else "Ready to Apply!"
    
    df['missing_skills'] = df['skills'].apply(check_gap)
    return df.sort_values('match_percent', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR & NAVIGATION
# ──────────────────────────────────────────────────────────────
df = load_student_data()

with st.sidebar:
    st.markdown("<h2 style='color:#065f46;'>🎓 TalentMatch AI</h2>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Home", "🔍 Find My Internship", "📄 Project Proposal"])
    st.markdown("---")
    st.write("**Authors:**")
    st.caption("Waqaas Hussain (291)\nHira Abdul Hafeez (314)")
    st.write("**Institution:**")
    st.caption("Aror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. APP SECTIONS
# ──────────────────────────────────────────────────────────────

# --- HOME ---
if page == "🏠 Home":
    st.title("AI-Based Job Recommendation System")
    st.subheader("BS Artificial Intelligence | Programming for AI")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("""
        Welcome to the official project demo. This system was designed to help students 
        at **Aror University** bridge the gap between academic learning and industry 
        requirements. By using **TF-IDF Vectorization**, we analyze the semantic meaning 
        of your skills to find the most relevant internships in Pakistan.
        """)
        st.success("🎯 Goal: Improve job search accuracy via Content-Based Filtering.")
    with col2:
        st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?q=80&w=400", use_container_width=True)

# --- SEARCH ENGINE ---
elif page == "🔍 Find My Internship":
    st.header("Intelligent Matching Engine")
    
    skills_in = st.text_input("Enter your Skills (e.g. Python, SQL, React, Machine Learning)")
    loc_filter = st.selectbox("Filter by City", ["All Cities", "Karachi", "Lahore", "Islamabad", "Sukkur"])
    
    if st.button("Analyze & Match"):
        results = run_match_engine(skills_in, df)
        
        if results.empty:
            st.warning("Please enter your skills to begin the matching process.")
        else:
            if loc_filter != "All Cities":
                results = results[results['location'] == loc_filter]
            
            for _, row in results.iterrows():
                if row['match_percent'] > 0:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div style="display:flex; gap:15px;">
                                <img src="{row['img']}" width="100" style="border-radius:8px; object-fit:cover;">
                                <div>
                                    <h3 style="margin:0;">{row['title']}</h3>
                                    <p style="margin:0; color:#64748b; font-weight:600;">{row['company']} • {row['location']}</p>
                                    <p style="margin-top:8px; font-size:0.9rem;">{row['desc']}</p>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <span class="badge-intern">{row['type']}</span><br>
                                <span class="badge-match" style="display:inline-block; margin-top:8px;">{int(row['match_percent'])}% Match</span>
                            </div>
                        </div>
                        <div style="margin-top:15px; border-top: 1px solid #f3f4f6; padding-top:10px;">
                            <span style="font-size:0.85rem; font-weight:bold; color:#065f46;">Stipend: {row['salary']}</span>
                            <div class="skill-missing">⚠️ Missing Skills: {row['missing_skills']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# --- PROPOSAL DOCUMENTATION ---
elif page == "📄 Project Proposal":
    st.header("Project Proposal Documentation")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Methodology", "Scope"])
    
    with tab1:
        st.markdown(f"""
        **Instructor:** Sir Abdul Haseeb  
        **Problem:** Traditional keyword matching leads to irrelevant job suggestions.  
        **Solution:** Content-based filtering using Machine Learning to analyze user-skill semantic relationships.
        """)
        
    with tab2:
        st.markdown("""
        **Methodology Steps:**
        1. **Data Collection:** Job titles/skills via CSV datasets.
        2. **Preprocessing:** Text cleaning, stop-word removal, normalization.
        3. **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency).
        4. **Similarity:** Cosine Similarity to calculate vector distance.
        """)
        
    with tab3:
        st.markdown("""
        **Included:** Recommender system, Streamlit GUI, Dataset analysis.  
        **Not Included:** Mobile App, Real-time API, Deep Learning.
        """)

st.markdown("---")
st.caption("© 2026 | Aror University Sukkur | Department of Artificial Intelligence")

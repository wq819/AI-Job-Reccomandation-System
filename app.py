# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM
#   Author      : Waqaas Hussain
#   Institution : Aror University Sukkur
#   Subject     : Programming for AI
#   Algorithm   : TF-IDF Vectorization + Cosine Similarity
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
#  1. PAGE CONFIGURATION & UI STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Matcher | Waqaas Hussain", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f4f7f6; }
    .header-container {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 40px; border-radius: 20px; color: white;
        text-align: center; margin-bottom: 30px;
    }
    .job-card {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        border-left: 6px solid #3b82f6; margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .job-card:hover { transform: translateY(-5px); }
    .match-badge {
        background: #dcfce7; color: #166534;
        padding: 5px 15px; border-radius: 20px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATA COLLECTION (Methodology Step 1)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_job_data():
    # Structured dataset representing real-world job listings
    data = [
        {"id": 1, "title": "AI Engineer", "company": "DeepMind", "location": "Remote", "category": "AI/ML", "skills": "Python, TensorFlow, PyTorch, Deep Learning", "desc": "Design and implement neural networks and machine learning models."},
        {"id": 2, "title": "Data Scientist", "company": "DataCorp", "location": "Karachi", "category": "Data Science", "skills": "Python, R, SQL, Machine Learning, Statistics", "desc": "Extract insights from complex datasets and build predictive models."},
        {"id": 3, "title": "Full Stack Developer", "company": "SoftSolutions", "location": "Remote", "category": "Engineering", "skills": "React, Node.js, JavaScript, MongoDB, HTML, CSS", "desc": "Develop and maintain end-to-end web applications."},
        {"id": 4, "title": "Backend Developer", "company": "FinTech Hub", "location": "Lahore", "category": "Engineering", "skills": "Python, Django, PostgreSQL, Docker, REST APIs", "desc": "Build scalable backend architectures and API integrations."},
        {"id": 5, "title": "Cybersecurity Specialist", "company": "SecureNet", "location": "Islamabad", "category": "Security", "skills": "Network Security, Linux, Ethical Hacking, Python", "desc": "Identify vulnerabilities and safeguard corporate digital infrastructure."},
        {"id": 6, "title": "Cloud Architect", "company": "CloudNine", "location": "Remote", "category": "DevOps", "skills": "AWS, Azure, Kubernetes, Terraform, Python", "desc": "Design and manage cloud infrastructure and automation pipelines."},
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. DATA PREPROCESSING & MODEL (Methodology Steps 2 & 3)
# ──────────────────────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@st.cache_resource
def initialize_engine(df):
    # TF-IDF Vectorization for Feature Extraction
    tfidf = TfidfVectorizer(stop_words='english')
    # Combine Title, Skills, and Description for a semantic profile
    content = df['title'] + " " + df['skills'] + " " + df['desc']
    tfidf_matrix = tfidf.fit_transform(content.apply(preprocess_text))
    return tfidf, tfidf_matrix

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
df = load_job_data()
tfidf_vec, matrix = initialize_engine(df)

with st.sidebar:
    st.title("Navigation")
    selection = st.radio("Sections", ["🏠 Home", "🔍 Job Recommendations", "📊 Market Analytics", "📄 Project Proposal"])
    st.markdown("---")
    st.markdown(f"**Developed by:** \nWaqaas Hussain  \n*Aror University Sukkur*")

# ──────────────────────────────────────────────────────────────
#  5. SECTION: HOME
# ──────────────────────────────────────────────────────────────
if selection == "🏠 Home":
    st.markdown("""
    <div class="header-container">
        <h1>AI-Based Job Recommendation System</h1>
        <p>A Content-Based Filtering Approach Using TF-IDF & Cosine Similarity</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Welcome to the Future of Job Hunting")
        st.write("""
        Finding the right job is a complex task. Traditional systems rely on simple keywords, 
        often yielding irrelevant results. This Artificial Intelligence system analyzes the 
        semantic relationship between your skills and job requirements to provide 
        highly accurate recommendations.
        """)
        st.info("Use the sidebar to input your skills and find the best matches!")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3222/3222800.png", width=250)

# ──────────────────────────────────────────────────────────────
#  6. SECTION: JOB RECOMMENDATIONS (Core Engine)
# ──────────────────────────────────────────────────────────────
elif selection == "🔍 Job Recommendations":
    st.header("Search & Match Engine")
    
    left, right = st.columns([1, 2])
    
    with left:
        st.subheader("Your Profile")
        user_input = st.text_area("List Your Skills", placeholder="e.g. Python, SQL, Machine Learning, React", height=150)
        loc_pref = st.selectbox("Preferred Location", ["Any", "Remote", "Karachi", "Lahore", "Islamabad"])
        process_btn = st.button("Generate Recommendations", type="primary")

    with right:
        if process_btn and user_input:
            # Step 4: Cosine Similarity Matching
            user_vector = tfidf_vec.transform([preprocess_text(user_input)])
            scores = cosine_similarity(user_vector, matrix).flatten()
            
            df['match_percentage'] = scores * 100
            results = df.sort_values(by='match_percentage', ascending=False)
            
            if loc_pref != "Any":
                results = results[results['location'] == loc_pref]

            st.subheader("Matched Job Listings")
            for _, row in results.iterrows():
                if row['match_percentage'] > 5:  # Threshold
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.4rem; font-weight:700; color:#1e293b;">{row['title']}</span>
                            <span class="match-badge">{int(row['match_percentage'])}% Match</span>
                        </div>
                        <div style="color:#64748b; margin-top:5px;">🏢 {row['company']} | 📍 {row['location']}</div>
                        <p style="margin-top:15px; color:#475569;">{row['desc']}</p>
                        <div style="margin-top:10px;">
                            <code style="background:#f1f5f9; padding:5px; color:#2563eb;">{row['skills']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        elif process_btn and not user_input:
            st.error("Please provide your skills to analyze.")
        else:
            st.info("Awaiting input... enter your skills in the left panel.")

# ──────────────────────────────────────────────────────────────
#  7. SECTION: ANALYTICS
# ──────────────────────────────────────────────────────────────
elif selection == "📊 Market Analytics":
    st.header("System Statistics & Insights")
    st.write("Visual breakdown of the current job dataset.")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_pie = px.pie(df, names='category', title='Jobs by Category', hole=0.5)
        st.plotly_chart(fig_pie)
    with c2:
        fig_bar = px.bar(df, x='location', color='category', title='Job Distribution by Location')
        st.plotly_chart(fig_bar)

# ──────────────────────────────────────────────────────────────
#  8. SECTION: PROJECT PROPOSAL (Proposal Content)
# ──────────────────────────────────────────────────────────────
elif selection == "📄 Project Proposal":
    st.header("Academic Project Documentation")
    
    with st.expander("Project Introduction & Background", expanded=True):
        st.write("""
        Artificial Intelligence can improve the recruitment process by analyzing user skills and matching them 
        accurately with job requirements. This project focuses on developing a Streamlit-based interface 
        for intelligent job suggestions.
        """)

    with st.expander("Problem Statement"):
        st.write("Traditional systems lack personalization. This AI system addresses keyword-matching inefficiencies.")

    with st.expander("Objectives"):
        st.markdown("""
        - Design a Streamlit-based GUI.
        - Develop a Machine Learning model for matching.
        - Analyze semantic profiles of user skills.
        - Evaluate performance using Cosine Similarity.
        """)

    with st.expander("Proposed Methodology"):
        st.markdown("""
        - **Step 1:** Data Collection (CSV/Manual Datasets).
        - **Step 2:** Preprocessing (Normalization/Stop-word removal).
        - **Step 3:** Feature Extraction (TF-IDF Vectorization).
        - **Step 4:** Implementation (Python & Streamlit).
        """)

# Footer
st.markdown("---")
st.caption("| Aror University Sukkur | Department of Artificial Intelligence")

# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (CORE ENGINE)
#   Institution : Aror University Sukkur
#   Logic       : TF-IDF Vectorization + Cosine Similarity
#   Level       : BS Artificial Intelligence - Semester 4
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ─── 1. ADVANCED DATASET (Mock Database for Vectorization) ───
@st.cache_data
def load_internal_job_db():
    # In a real 4th-sem project, this would be a CSV file.
    data = [
        {"id": 1, "title": "AI Research Intern", "company": "Systems Ltd", "skills": "Python, Machine Learning, NLP, Scikit-learn", "desc": "Assisting in neural network training and data preprocessing."},
        {"id": 2, "title": "Junior Data Scientist", "company": "Afiniti", "skills": "SQL, Python, Statistics, Machine Learning, R", "desc": "Analyzing behavioral patterns using advanced matching algorithms."},
        {"id": 3, "title": "Web Dev Intern", "company": "10Pearls", "skills": "HTML, CSS, JavaScript, React, Git", "desc": "Developing responsive frontend components for scale."},
        {"id": 4, "title": "Cloud Architect", "company": "NetSol", "skills": "AWS, Docker, Linux, Kubernetes, Python", "desc": "Designing scalable cloud-native infrastructures."},
        {"id": 5, "title": "Software Engineer (AI)", "company": "Folio3", "skills": "Python, Django, PostgreSQL, Git, Computer Vision", "desc": "Integrating computer vision models into mobile applications."}
    ]
    return pd.DataFrame(data)

# ─── 2. MATHEMATICAL MATCHING ENGINE (The "AI" Part) ───
def calculate_similarity_scores(user_profile, df):
    """
    Implements Content-Based Filtering using TF-IDF and Cosine Similarity.
    This demonstrates 4th-semester NLP concepts.
    """
    # Preprocessing
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
    # Feature Extraction
    tfidf = TfidfVectorizer(stop_words='english')
    # Combine job metadata into a single string for vectorization
    job_content = df['title'] + " " + df['skills'] + " " + df['desc']
    
    # Generate TF-IDF Matrix
    tfidf_matrix = tfidf.fit_transform(job_content.apply(clean))
    
    # Transform User Profile to same Vector Space
    user_vec = tfidf.transform([clean(user_profile)])
    
    # Compute Cosine Similarity (Math: dot product of normalized vectors)
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    return scores * 100

# ─── 3. PAGE CONFIG & STYLING (KEEPING YOUR DARK UI) ───
st.set_page_config(page_title="TalentMatch AI | Aror University", layout="wide")

# (Inject your CSS here - omitted for brevity, keep your original CSS)

# ─── 4. LOGIC INTEGRATION ───
st.session_state.df = load_internal_job_db()

with st.sidebar:
    st.title("TalentMatch AI")
    st.write("BS AI Semester 4 Project")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    st.subheader("Candidate Profile")
    u_name = st.text_input("Full Name")
    u_skills = st.text_area("Your Skills", placeholder="e.g. Python, Machine Learning, SQL")
    u_goal = st.text_input("Target Role", placeholder="Data Scientist")
    
    match_btn = st.button("Calculate Match & Get Advice", type="primary")

# ─── 5. DUAL-ENGINE PROCESSING ───
# Engine 1: Mathematical Ranking (TF-IDF)
# Engine 2: LLM Insights (Gemini)

if match_btn and u_skills:
    # A. VECTOR MATH RANKING
    with st.spinner("Processing Vectors..."):
        scores = calculate_similarity_scores(f"{u_goal} {u_skills}", st.session_state.df)
        st.session_state.df['ai_match'] = scores
        results = st.session_state.df.sort_values(by='ai_match', ascending=False)

    # B. LLM CAREER ADVICE
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        Analyze this candidate for the role of {u_goal}.
        Skills: {u_skills}.
        Provide a 3-sentence expert career strategy and identify the #1 missing skill.
        """
        response = model.generate_content(prompt).text
        st.info(f"**AI Strategist Advice:** {response}")

    # ─── 6. VISUALIZING RESULTS ───
    st.subheader("Ranked Job Opportunities")
    
    # Image of TF-IDF Vector Space Concept
    
    
    cols = st.columns(2)
    for i, (idx, row) in enumerate(results.iterrows()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#1A1A2E; padding:20px; border-radius:15px; border:1px solid #6C63FF; margin-bottom:15px;">
                <h3 style="margin:0; color:#43E97B;">{row['title']}</h3>
                <p style="color:#9999BB; margin:0;">{row['company']}</p>
                <div style="margin-top:10px;">
                    <span style="background:#6C63FF; padding:4px 10px; border-radius:10px; font-size:0.8em;">
                        {int(row['ai_match'])}% Mathematical Match
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.title("Welcome to TalentMatch AI")
    st.write("Enter your profile details on the left to see the AI Recommendation Engine in action.")

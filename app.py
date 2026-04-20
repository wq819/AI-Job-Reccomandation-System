# ============================================================
#   TALENTMATCH AI PRO (ENTERPRISE EDITION)
#   Student     : Waqaas Hussain (SAP-5000000291)
#   Instructor  : Sir Abdul Haseeb (BS AI - Semester 4)
#   Core Logic  : BERT Transformers + PDF Parsing + Session Auth
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
from sentence_transformers import SentenceTransformer, util # Week 13: ML Overview
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
#  1. SESSION & LOGIN SYSTEM (Requirement 4)
# ──────────────────────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login_system():
    if not st.session_state.logged_in:
        st.markdown("### 🔐 Secure Student Login")
        user = st.text_input("Username (SAP ID)")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if user and pwd: # Simplified for Demo
                st.session_state.logged_in = True
                st.rerun()
        st.stop()

# ──────────────────────────────────────────────────────────────
#  2. UI & TRANSFORMER MODEL INITIALIZATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch Pro | BERT", layout="wide")

@st.cache_resource
def load_bert_model():
    # Week 13: Using Pre-trained Transformer models (BERT)
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_bert_model()

# ──────────────────────────────────────────────────────────────
#  3. RESUME UPLOAD & PDF PARSING (Requirement 3)
# ──────────────────────────────────────────────────────────────
def extract_data_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# ──────────────────────────────────────────────────────────────
#  4. MOCK API DATASET (Requirement 1 & 5)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def fetch_mock_api_jobs():
    # Simulating data that would come from a Real Job API
    data = [
        {"title": "Senior AI Engineer", "company": "Systems Ltd", "loc": "Lahore", "skills": "Python, BERT, PyTorch", "url": "https://systemsltd.com/careers"},
        {"title": "Data Scientist", "company": "Afiniti", "loc": "Karachi", "skills": "SQL, Statistics, R, ML", "url": "https://afiniti.com/careers"},
        {"title": "MLOps Lead", "company": "NetSol", "loc": "Islamabad", "skills": "Docker, AWS, Kubernetes", "url": "https://netsoltech.com/careers"},
        {"title": "AI Intern", "company": "Aror Solutions", "loc": "Sukkur", "skills": "Python, Flask, Git", "url": "#"}
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  5. MAIN APP INTERFACE
# ──────────────────────────────────────────────────────────────
login_system() # Activate Login

st.title("🚀 TalentMatch Pro: AI Recruitment Engine")
st.write(f"Candidate: **{st.session_state.get('user', 'Waqaas Hussain')}** | Aror University Sukkur")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("📤 Upload Resume")
    uploaded_file = st.file_uploader("Choose your PDF Resume", type="pdf")
    manual_skills = st.text_area("Or Paste Skills Manually")
    
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# --- Matching Logic ---
if uploaded_file or manual_skills:
    # Get Resume Text
    resume_text = extract_data_from_pdf(uploaded_file) if uploaded_file else manual_skills
    
    # 1. Use BERT / Transformers (Requirement 2)
    with st.spinner("BERT Model analyzing semantic context..."):
        jobs_df = fetch_mock_api_jobs()
        job_descriptions = jobs_df['title'] + " " + jobs_df['skills']
        
        # Encoding (Transforming text to BERT Vector Space)
        job_embeddings = model.encode(job_descriptions.tolist(), convert_to_tensor=True)
        user_embedding = model.encode(resume_text, convert_to_tensor=True)
        
        # Calculate Cosine Similarity via BERT Tensors
        cosine_scores = util.cos_sim(user_embedding, job_embeddings).flatten()
        jobs_df['match_score'] = cosine_scores.tolist()

    # --- Display Results ---
    st.subheader("🎯 Top Semantic Matches")
    results = jobs_df.sort_values(by='match_score', ascending=False)
    
    for _, row in results.iterrows():
        match_pct = int(row['match_score'] * 100)
        if match_pct > 20: # Display relevant matches
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; border-left:5px solid #10b981;">
                        <h4>{row['title']} @ {row['company']}</h4>
                        <p>📍 {row['loc']} | <b>Skills:</b> {row['skills']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("Match", f"{match_pct}%")
                    # 2. Add job apply button (Requirement 5)
                    st.link_button("Apply Now ↗️", row['url'])
                st.markdown("<br>", unsafe_allow_html=True)

else:
    st.info("Please upload your PDF resume or enter skills to begin the Transformer-based matching.")

st.markdown("---")
st.caption("BS AI Semester 4 | Advanced Implementation: Transformers & PDF Parsing")

# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM
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
st.set_page_config(page_title="AI Job Recommendation System", layout="wide", page_icon="💼")

# THEME: Deep Indigo (#4338ca) and Soft Slate
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        padding: 50px; border-radius: 25px; color: white;
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(67, 56, 202, 0.3);
    }
    
    /* Job Card Styling */
    .job-card {
        background: white; padding: 25px; border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-top: 5px solid #4338ca; margin-bottom: 20px;
        transition: all 0.3s ease;
        position: relative;
    }
    .job-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
    }
    
    /* Badge Styling */
    .match-badge {
        background: #e0e7ff; color: #4338ca;
        padding: 6px 16px; border-radius: 50px; font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Logo Styling */
    .company-logo {
        width: 55px; height: 55px; border-radius: 12px;
        background: #f1f5f9; padding: 5px; margin-right: 15px;
        object-fit: contain;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATA COLLECTION
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_job_data():
    # Added 'logo' links for visual representation
    data = [
        {"id": 1, "title": "AI Engineer", "company": "DeepMind", "location": "Remote", "category": "AI/ML", "skills": "Python, TensorFlow, PyTorch, Deep Learning", "desc": "Design and implement neural networks and machine learning models.", "logo": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"},
        {"id": 2, "title": "Data Scientist", "company": "DataCorp", "location": "Karachi", "category": "Data Science", "skills": "Python, R, SQL, Machine Learning, Statistics", "desc": "Extract insights from complex datasets and build predictive models.", "logo": "https://cdn-icons-png.flaticon.com/512/4248/4248873.png"},
        {"id": 3, "title": "Full Stack Developer", "company": "SoftSolutions", "location": "Remote", "category": "Engineering", "skills": "React, Node.js, JavaScript, MongoDB, HTML, CSS", "desc": "Develop and maintain end-to-end web applications.", "logo": "https://cdn-icons-png.flaticon.com/512/1183/1183672.png"},
        {"id": 4, "title": "Backend Developer", "company": "FinTech Hub", "location": "Lahore", "category": "Engineering", "skills": "Python, Django, PostgreSQL, Docker, REST APIs", "desc": "Build scalable backend architectures and API integrations.", "logo": "https://cdn-icons-png.flaticon.com/512/2721/2721620.png"},
        {"id": 5, "title": "Cybersecurity Specialist", "company": "SecureNet", "location": "Islamabad", "category": "Security", "skills": "Network Security, Linux, Ethical Hacking, Python", "desc": "Identify vulnerabilities and safeguard corporate digital infrastructure.", "logo": "https://cdn-icons-png.flaticon.com/512/1055/1055683.png"},
        {"id": 6, "title": "Cloud Architect", "company": "CloudNine", "location": "Remote", "category": "DevOps", "skills": "AWS, Azure, Kubernetes, Terraform, Python", "desc": "Design and manage cloud infrastructure and automation pipelines.", "logo": "https://cdn-icons-png.flaticon.com/512/1162/1162499.png"},
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. PREPROCESSING & ENGINE
# ──────────────────────────────────────────────────────────────
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@st.cache_resource
def initialize_engine(df):
    tfidf = TfidfVectorizer(stop_words='english')
    content = df['title'] + " " + df['skills'] + " " + df['desc']
    tfidf_matrix = tfidf.fit_transform(content.apply(preprocess_text))
    return tfidf, tfidf_matrix

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR
# ──────────────────────────────────────────────────────────────
df = load_job_data()
tfidf_vec, matrix = initialize_engine(df)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.title("Control Center")
    selection = st.radio("Navigation", ["🏠 Home", "🔍 Job Recommendations", "📊 Market Analytics", "📄 Project Proposal"])
    st.markdown("---")
    st.markdown(f"**Researcher:** \nWaqaas Hussain  \n*Aror University Sukkur*")

# ──────────────────────────────────────────────────────────────
#  5. SECTION: HOME
# ──────────────────────────────────────────────────────────────
if selection == "🏠 Home":
    st.markdown("""
    <div class="header-container">
        <h1>AI-Based Job Recommendation System</h1>
        <p style="font-size: 1.1rem; opacity: 0.9;">Leveraging TF-IDF & Cosine Similarity for Intelligent Career Matching</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Redefining Job Search")
        st.write("""
        Stop searching through irrelevant listings. Our AI analyzes the **semantic DNA** of your skill set and matches it against specific job requirements. 
        Whether you are a Data Scientist or a Cloud Architect, we find where you belong.
        """)
        st.info("💡 **Pro-Tip:** List specific libraries (like PyTorch or React) for a higher match accuracy!")
    with col2:
        st.image("https://cdn-icons-png.flaticon.

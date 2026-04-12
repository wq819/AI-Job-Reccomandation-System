# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (Proposal Implementation)
#   Prepared by: Waqaas Hussain
#   Subject: Programming for AI | Aror University Sukkur
#   Algorithm: TF-IDF + Cosine Similarity
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re
import warnings

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Job Recommendation AI", layout="wide", page_icon="🎯")

# Professional UI Styling
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .header-box { 
        background: linear-gradient(135deg, #1e3a8a, #3b82f6); 
        padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px;
    }
    .card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #3b82f6; margin-bottom: 15px;
    }
    .match-pct { color: #10b981; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  DATASET (Methodology Step 1: Data Collection)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    # As per your scope, we use a structured dataset
    data = [
        {"id": 1, "title": "AI Engineer", "company": "Google", "location": "Remote", "skills": "Python, TensorFlow, PyTorch, Machine Learning", "desc": "Develop AI models and neural networks using deep learning frameworks."},
        {"id": 2, "title": "Data Analyst", "company": "Amazon", "location": "Karachi", "skills": "SQL, Excel, Power BI, Python", "desc": "Analyze business data and create dashboards using SQL and Power BI."},
        {"id": 3, "title": "Web Developer", "company": "Meta", "location": "Remote", "skills": "React, JavaScript, HTML, CSS, Node.js", "desc": "Build responsive web applications using modern frontend and backend tools."},
        {"id": 4, "title": "Python Developer", "company": "Microsoft", "location": "Lahore", "skills": "Python, Django, Flask, PostgreSQL", "desc": "Backend development using Python frameworks and database management."},
        {"id": 5, "title": "Mobile App Developer", "company": "Apple", "location": "Remote", "skills": "Flutter, Dart, Swift, Firebase", "desc": "Creating cross-platform mobile applications for iOS and Android."},
        {"id": 6, "title": "Cybersecurity Analyst", "company": "IBM", "location": "Islamabad", "skills": "Network Security, Linux, Python, Ethical Hacking", "desc": "Protecting systems from threats and monitoring network security."},
        {"id": 7, "title": "MLOps Engineer", "company": "Tesla", "location": "Remote", "skills": "Docker, Kubernetes, Python, AWS", "desc": "Managing machine learning pipelines and deployment in cloud environments."},
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  PREPROCESSING & MODEL (Methodology Step 2 & 3)
# ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@st.cache_resource
def build_vectorizer(df):
    # TF-IDF for Feature Extraction
    tfidf = TfidfVectorizer(stop_words='english')
    # Combine Title and Skills for better matching
    combined_data = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(combined_data.apply(clean_text))
    return tfidf, matrix

# ──────────────────────────────────────────────────────────────
#  GUI & INTERFACE (Methodology Step 4)
# ──────────────────────────────────────────────────────────────

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.title("Navigation")
    menu = st.radio("Go to:", ["🏠 Home", "🚀 Recommendation", "📊 Market Analytics", "📜 Project Proposal"])
    st.markdown("---")
    st.info(f"**Prepared by:**\nWaqaas Hussain\nAror University Sukkur")

df = get_data()
tfidf, matrix = build_vectorizer(df)

# --- 🏠 HOME PAGE ---
if menu == "🏠 Home":
    st.markdown('<div class="header-box"><h1>AI-Based Job Recommendation System</h1><p>Intelligent Matching using Content-Based Filtering</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Welcome Waqaas!")
        st.write("""
        Online job platforms par suitable job dhondna mushkil kaam hai. 
        Ye system AI ke zaryae aapki skills ko analyze karta hai aur 
        aapko personalized suggestions deta hai.
        """)
        if st.button("Start Finding Jobs"):
            st.toast("Sidebar se Recommendation select karein!")
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135697.png", width=250)

# --- 🚀 RECOMMENDATION PAGE (Core Functionality) ---
elif menu == "🚀 Recommendation":
    st.header("🔍 Find Your Dream Job")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Your Profile")
        u_skills = st.text_area("Enter Your Skills (comma separated)", placeholder="e.g. Python, SQL, React")
        u_loc = st.selectbox("Preferred Location", ["Any", "Remote", "Karachi", "Lahore", "Islamabad"])
        search_btn = st.button("Analyze & Match")

    with col2:
        if search_btn and u_skills:
            # Step 3: Algorithm Matching
            user_vec = tfidf.transform([clean_text(u_skills)])
            similarity = cosine_similarity(user_vec, matrix).flatten()
            
            df['match_score'] = similarity
            results = df.sort_values(by='match_score', ascending=False)
            
            # Filter by location if not 'Any'
            if u_loc != "Any":
                results = results[results['location'] == u_loc]

            st.markdown(f"### Top Matches for You:")
            
            for index, row in results.head(5).iterrows():
                score = int(row['match_score'] * 100)
                if score > 0:
                    st.markdown(f"""
                    <div class="card">
                        <div style="display:flex; justify-content:space-between;">
                            <span class="job-title" style="font-size:1.3rem; color:#1e3a8a;">{row['title']}</span>
                            <span class="match-pct">{score}% Match</span>
                        </div>
                        <p><strong>Company:</strong> {row['company']} | <strong>Location:</strong> {row['location']}</p>
                        <p style="font-size:0.9rem; color:#555;">{row['desc']}</p>
                        <div style="margin-top:10px;">
                            <span style="background:#e0f2fe; padding:4px 10px; border-radius:15px; font-size:0.8rem;">{row['skills']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
        elif search_btn and not u_skills:
            st.error("Please enter some skills to see recommendations.")
        else:
            st.info("Sidebar mein apni skills likhen aur Match button dabayen.")

# --- 📊 ANALYTICS PAGE ---
elif menu == "📊 Market Analytics":
    st.header("📈 Job Market Insights")
    
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(df, names='category', title='Job Distribution by Category', hole=0.4)
        st.plotly_chart(fig1)
    
    with c2:
        # Dummy trends based on dataset
        fig2 = px.bar(df, x='title', y='id', color='location', title='Job Availability by Location')
        st.plotly_chart(fig2)

# --- 📜 PROPOSAL PAGE (As per your request) ---
elif menu == "📜 Project Proposal":
    st.header("Project Proposal Details")
    
    with st.expander("4. Introduction / Background", expanded=True):
        st.write("""
        With the increasing use of online job platforms, finding suitable jobs has become a challenging task...
        Artificial Intelligence (AI) can improve this process by analyzing user skills.
        """)
        
    with st.expander("5. Problem Statement"):
        st.error("Current job recommendation systems lack personalization and accuracy. They depend mainly on keyword matching.")

    with st.expander("6. Objectives"):
        st.success("""
        - To design a GUI-based job recommendation system using Streamlit
        - To develop a machine learning model for job matching
        - To analyze user skills and job descriptions for better recommendations
        """)

    with st.expander("9. Proposed Methodology"):
        st.image("https://cdn-icons-png.flaticon.com/512/2620/2620611.png", width=50)
        st.write("""
        **Step 1:** Data Collection (CSV/Kaggle)  
        **Step 2:** Data Preprocessing (Cleaning/Tokenization)  
        **Step 3:** Model Design (TF-IDF & Cosine Similarity)  
        **Step 4:** Implementation (Python & Streamlit)
        """)

# Footer
st.markdown("---")
st.caption("© 2026 Waqaas Hussain | Programming for AI Project | Aror University Sukkur")

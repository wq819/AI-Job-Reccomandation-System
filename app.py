# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (V1.0)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Techniques  : TF-IDF Vectorization & Cosine Similarity
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
#  1. THEME MANAGEMENT (DARK & WHITE MODE)
# ──────────────────────────────────────────────────────────────
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

def toggle_theme():
    st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'

# UI Colors based on selection
if st.session_state.theme == 'dark':
    bg_color = "#0D0D1A"
    text_color = "#E8E8F0"
    card_bg = "#16213E"
    border_color = "rgba(108,99,255,0.3)"
else:
    bg_color = "#F0F2F6"
    text_color = "#1E293B"
    card_bg = "#FFFFFF"
    border_color = "#D1D5DB"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    .job-card {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .match-score {{ color: #43E97B; font-weight: bold; font-family: 'Courier New', monospace; }}
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. LOCAL DATASET (Pakistani Tech Hubs)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = [
        {"title": "AI Research Intern", "company": "Systems Ltd", "location": "Lahore", "skills": "Python, Machine Learning, NLP, Scikit-learn", "desc": "Assisting in neural network training and data preprocessing."},
        {"title": "Junior Data Scientist", "company": "Afiniti", "location": "Karachi", "skills": "SQL, Python, Statistics, Machine Learning", "desc": "Analyzing patterns using behavioral matching algorithms."},
        {"id": 3, "title": "Web Dev Intern", "company": "Aror Solutions", "location": "Sukkur", "skills": "HTML, CSS, JavaScript, React, Git", "desc": "Developing responsive frontend components for local clients."},
        {"title": "Cloud Associate", "company": "NetSol", "location": "Islamabad", "skills": "AWS, Docker, Linux, Python", "desc": "Designing scalable cloud-native infrastructures."},
        {"title": "Junior ML Engineer", "company": "Folio3", "location": "Karachi", "skills": "Python, Django, Computer Vision, Git", "desc": "Integrating CV models into mobile applications."}
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. NLP MATCHING ENGINE (TF-IDF & COSINE SIMILARITY)
# ──────────────────────────────────────────────────────────────
def calculate_recommendations(user_input, df):
    # Data Cleaning
    def clean_text(text):
        return re.sub(r'[^a-z0-9\s]', '', text.lower())

    # Feature Extraction
    tfidf = TfidfVectorizer(stop_words='english')
    # Combine relevant columns into a "bag of words"
    job_content = df['title'] + " " + df['skills'] + " " + df['desc']
    
    # Mathematical Representation (Vector Space Model)
    tfidf_matrix = tfidf.fit_transform(job_content.apply(clean_text))
    user_vec = tfidf.transform([clean_text(user_input)])
    
    # Calculate Similarity Score (Cosine Similarity)
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['match_percent'] = scores * 100
    return df.sort_values(by='match_percent', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. USER INTERFACE
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Settings")
    st.button("🌓 Switch Theme", on_click=toggle_theme)
    st.markdown("---")
    st.header("Candidate Profile")
    u_skills = st.text_area("List your skills", placeholder="e.g. Python, Machine Learning, SQL")
    u_role = st.text_input("Target Job Title", placeholder="e.g. Data Scientist")
    
    find_jobs = st.button("APPLY", type="primary")

# Header Section
st.title("AI JOB RECCOMANDATION SYSTEM")
st.subheader("BS Artificial Intelligence | Semester 4 ")
st.write(f" Developed By  Waqaas Hussain & Hira Abdul Hafeez | Aror University Sukkur ")

if find_jobs and u_skills:
    df = load_data()
    results = calculate_recommendations(f"{u_role} {u_skills}", df)
    
    st.markdown("### Recommended Opportunities")
    
    for i, row in results.iterrows():
        if row['match_percent'] > 0:
            st.markdown(f"""
            <div class="job-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0;">{row['title']}</h3>
                    <span class="match-score">{int(row['match_percent'])}% AI Match</span>
                </div>
                <p style="margin:5px 0; opacity:0.8;">{row['company']} • {row['location']}</p>
                <div style="margin-top:10px;">
                    <code style="background:rgba(108,99,255,0.1); padding:4px 8px; border-radius:5px;">
                        {row['skills']}
                    </code>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info(" Welcome! Please enter your skills in the sidebar to see the recommendation system in action.")

# Technical Logic Explanation (For the VIVA/Presentation)
with st.expander("🔬 View Technical Logic (NLP Pipeline)"):
    st.write("""
    1. **Preprocessing**: Text is lowercased and cleaned using Regular Expressions (Regex).
    2. **Tokenization**: Breaking text into individual words.
    3. **TF-IDF Vectorization**: Converting text into numerical vectors based on word frequency.
    4. **Cosine Similarity**: Calculating the 'angle' between your skills and job requirements to determine match percentage.
    """)

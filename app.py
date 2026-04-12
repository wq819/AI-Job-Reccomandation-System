# ============================================================
#   AI STUDENT CAREER LAUNCHPAD (PAKISTAN)
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

# ──────────────────────────────────────────────────────────────
#  1. THEME & UI (Student-Focused)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Career Launchpad", layout="wide", page_icon="🎓")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stButton>button { border-radius: 10px; height: 3em; background-color: #047857; color: white; border: none; }
    
    .student-card {
        background: white; border-radius: 18px; border: 1px solid #e2e8f0;
        margin-bottom: 25px; overflow: hidden; transition: 0.3s ease;
    }
    .student-card:hover { transform: scale(1.01); border-color: #047857; }
    
    .loc-header { width: 100%; height: 140px; object-fit: cover; }
    .content { padding: 20px; }
    
    .badge-intern { background: #dcfce7; color: #166534; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }
    .badge-match { background: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }
    .skill-gap { color: #dc2626; font-size: 0.85rem; font-weight: 600; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. STUDENT JOB DATABASE (Internships & Entry-Level)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_student_db():
    data = [
        {
            "id": 1, "title": "AI Intern", "company": "Systems Ltd", 
            "location": "Lahore", "type": "Internship", "stipend": "Rs. 25,000",
            "skills": "Python, Basic Machine Learning, Data Cleaning", 
            "desc": "Great for 3rd/4th year students. Work on real-world datasets and model tuning.",
            "loc_pic": "https://images.unsplash.com/photo-1590059530472-87034f593322?q=80&w=600"
        },
        {
            "id": 2, "title": "Junior Data Analyst", "company": "Foodpanda PK", 
            "location": "Karachi", "type": "Fresh Graduate", "stipend": "Rs. 70,000",
            "skills": "SQL, Excel, Python, PowerBI", 
            "desc": "Start your career in data. Analyze delivery patterns and user behavior.",
            "loc_pic": "https://images.unsplash.com/photo-1568205706871-332308933220?q=80&w=600"
        },
        {
            "id": 3, "title": "Web Dev Intern", "company": "TechVantage", 
            "location": "Sukkur", "type": "Internship", "stipend": "Rs. 15,000",
            "skills": "HTML, CSS, JavaScript, React, Bootstrap", 
            "desc": "Perfect for Aror University students looking for local experience in Sindh.",
            "loc_pic": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=600"
        },
        {
            "id": 4, "title": "Associate AI Engineer", "company": "S&P Global", 
            "location": "Islamabad", "type": "Fresh Graduate", "stipend": "Rs. 95,000",
            "skills": "Python, PyTorch, Git, Linux, Fast API", 
            "desc": "Intensive training program for top-performing AI graduates.",
            "loc_pic": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=600"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. MATCHING & SKILL GAP LOGIC
# ──────────────────────────────────────────────────────────────
def analyze_student_match(user_skills, df):
    if not user_skills.strip(): return pd.DataFrame()
    
    # 1. Similarity Score
    tfidf = TfidfVectorizer(stop_words='english')
    combined = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(combined)
    user_vec = tfidf.transform([user_skills.lower()])
    df['match'] = cosine_similarity(user_vec, matrix).flatten() * 100
    
    # 2. Basic Skill Gap Analysis
    def find_gap(required_skills):
        req_list = [s.strip().lower() for s in required_skills.split(',')]
        user_list = user_skills.lower()
        gap = [s for s in req_list if s not in user_list]
        return ", ".join(gap).title() if gap else "None! You're ready."
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values('match', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. STUDENT INTERFACE
# ──────────────────────────────────────────────────────────────
df = get_student_db()

with st.sidebar:
    st.markdown("<h2 style='color:#047857;'>🎓 Student Portal</h2>", unsafe_allow_html=True)
    menu = st.radio("Menu", ["🏠 Student Home", "🔍 Internship Finder", "📊 Skill Insights"])
    st.markdown("---")
    st.write("**Aror University Sukkur**")
    st.caption("AI Programming Project")

if menu == "🏠 Student Home":
    st.title("Your Bridge from University to Industry")
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?q=80&w=1200", caption="Digital Skills for Pakistan")
    st.info("Welcome students of Aror University! Use this tool to find internships and see what skills you need to learn.")

elif menu == "🔍 Internship Finder":
    st.header("Find Internships & Entry-Level Roles")
    
    # User Input
    col1, col2 = st.columns([2, 1])
    with col1:
        u_skills = st.text_input("Enter your current skills (e.g. Python, Java, HTML)")
    with col2:
        u_loc = st.selectbox("Preferred City", ["Any City"] + list(df['location'].unique()))
    
    if st.button("Match Me with Opportunities"):
        results = analyze_student_match(u_skills, df)
        
        if results.empty:
            st.warning("Enter your skills to see where you fit!")
        else:
            if u_loc != "Any City":
                results = results[results['location'] == u_loc]
            
            # Display Cards
            for _, row in results.iterrows():
                if row['match'] > 0:
                    st.markdown(f"""
                    <div class="student-card">
                        <img src="{row['loc_pic']}" class="loc-header">
                        <div class="content">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h3 style="margin:0; color:#0f172a;">{row['title']}</h3>
                                <div>
                                    <span class="badge-intern">{row['type']}</span>
                                    <span class="badge-match">{int(row['match'])}% Match</span>
                                </div>
                            </div>
                            <p style="color:#64748b; font-weight:600; margin:5px 0;">🏢 {row['company']} | 📍 {row['location']}</p>
                            <p style="font-size:0.9rem; margin-top:10px;">{row['desc']}</p>
                            <p style="font-size:0.85rem; font-weight:bold; color:#047857;">Stipend: {row['stipend']}</p>
                            <div class="skill-gap">⚠️ Missing Skills: {row['gap']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

elif menu == "📊 Skill Insights":
    st.header("What Skills are Pakistani Employers looking for?")
    st.plotly_chart(px.bar(df, x='title', y='location', color='type', title="Available Opportunities by City"), use_container_width=True)

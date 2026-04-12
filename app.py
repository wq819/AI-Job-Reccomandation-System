# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (BS AI - SEM 4)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Instructor: Sir Abdul Haseeb)
#   Logic       : NLP / TF-IDF Vectorization / Cosine Similarity
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
#  1. GLOBAL CONFIGURATION & CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI JOB RECCOMANDATION SYSTEM | Aror University", layout="wide", page_icon="🎓")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Professional Card Styling */
    .job-card {
        background: white; border-radius: 20px; border: 1px solid #e2e8f0;
        margin-bottom: 25px; overflow: hidden; transition: 0.4s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .job-card:hover { 
        transform: translateY(-5px); 
        border-color: #059669; 
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); 
    }
    
    .card-banner { width: 100%; height: 160px; object-fit: cover; }
    .card-content { padding: 25px; }
    
    /* Custom Badges */
    .badge { padding: 5px 14px; border-radius: 8px; font-weight: 700; font-size: 0.75rem; display: inline-block; }
    .badge-intern { background: #dcfce7; color: #166534; }
    .badge-fresh { background: #fef3c7; color: #92400e; }
    .badge-match { background: #dbeafe; color: #1e40af; }
    
    /* Skill Gap UI */
    .skill-gap-box {
        background: #fff1f2; border-left: 5px solid #e11d48;
        padding: 12px; margin-top: 15px; border-radius: 8px;
        font-size: 0.85rem; color: #9f1239;
    }
    
    /* Sidebar Branding */
    .sidebar-brand {
        padding: 20px; background: linear-gradient(135deg, #064e3b, #065f46);
        border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. STUDENT-CENTRIC DATASET 
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_project_database():
    data = [
        {
            "id": 1, "title": "AI Research Intern", "company": "Systems Ltd", 
            "location": "Lahore", "type": "Internship", "stipend": "Rs. 30,000",
            "skills": "Python, Machine Learning, NLP, Pandas, Scikit-learn", 
            "desc": "Assist the AI engineering team in building large-scale language models and data pipelines.",
            "img": "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=800"
        },
        {
            "id": 2, "title": "Junior Data Scientist", "company": "Afiniti", 
            "location": "Karachi", "type": "Fresh Graduate", "stipend": "Rs. 95,000",
            "skills": "SQL, Python, Statistics, Machine Learning, R, Spark", 
            "desc": "Join our world-class data science team to optimize behavioral matching algorithms.",
            "img": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800"
        },
        {
            "id": 3, "title": "AI Developer Intern", "company": "Aror Solutions", 
            "location": "Sukkur", "type": "Internship", "stipend": "Rs. 20,000",
            "skills": "Python, JavaScript, React, FastAPI, OpenAI API", 
            "desc": "Focus on developing AI-powered tools for the regional industry in Sindh.",
            "img": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=800"
        },
        {
            "id": 4, "title": "Cloud DevOps Intern", "company": "NetSol", 
            "location": "Islamabad", "type": "Internship", "stipend": "Rs. 25,000",
            "skills": "Linux, AWS, Docker, Jenkins, Kubernetes", 
            "desc": "Work within the DevOps department to manage cloud automation and security patches.",
            "img": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=800"
        },
        {
            "id": 5, "title": "Software Engineer (AI)", "company": "Folio3", 
            "location": "Karachi", "type": "Fresh Graduate", "stipend": "Rs. 85,000",
            "skills": "Python, Django, PostgreSQL, Git, Computer Vision", 
            "desc": "Develop and maintain high-performance backends for AI-driven mobile applications.",
            "img": "https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=800"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. 
# ──────────────────────────────────────────────────────────────
def compute_ai_matching(user_input, df):
    if not user_input.strip(): return pd.DataFrame()
    
    # Text Cleaning Function
    def clean_text(text): return re.sub(r'[^a-z0-9\s]', '', text.lower())
    
    # Vectorization (The 'AI' Part)
    tfidf = TfidfVectorizer(stop_words='english')
    # Combining Title, Skills, and Description for a semantic profile
    corpus = df['title'] + " " + df['skills'] + " " + df['desc']
    tfidf_matrix = tfidf.fit_transform(corpus.apply(clean_text))
    
    # User Profile Transformation
    user_vec = tfidf.transform([clean_text(user_input)])
    
    # Cosine Similarity Calculation
    df['match_score'] = cosine_similarity(user_vec, tfidf_matrix).flatten() * 100
    
    # Skill-Gap logic: Identifying what's missing
    def check_skills(required_str):
        required = [s.strip().lower() for s in required_str.split(',')]
        input_data = user_input.lower()
        gap = [s for s in required if s not in input_data]
        return ", ".join(gap).title() if gap else "Ready to Apply!"
    
    df['skill_gap'] = df['skills'].apply(check_skills)
    return df.sort_values('match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
df = load_project_database()

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><h2>TalentMatch AI</h2><p>AI JOB RECCOMANDATION SYSTEM</p></div>', unsafe_allow_html=True)
    nav_choice = st.radio("System Menu", ["🏠 Dashboard", "🔍 Smart Matcher", "📊 Skill Analytics"])
    st.markdown("---")
    st.write("**Developed By :**")
    st.caption("Waqaas Hussain (SAP-291)")
    st.caption("Hira Abdul Hafeez (SAP-314)")



# ──────────────────────────────────────────────────────────────
#  5. APP PAGES
# ──────────────────────────────────────────────────────────────

# --- PAGE: DASHBOARD ---
if nav_choice == "🏠 Dashboard":
    st.title("AI-Based Job Recommendation System")
    st.image("https://images.unsplash.com/photo-1522202176988-66273c2fd55f?q=80&w=1200", caption="Developing the Future Workforce of Pakistan")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Available Openings", len(df))
    col2.metric("Tech Hubs", "Sindh, Punjab, Federal")
    col3.metric("AI Core", "TF-IDF / Cosine")
    

# --- PAGE: SMART MATCHER ---
elif nav_choice == "🔍 Smart Matcher":
    st.header("AI Talent Matching Engine")
    
    input_col, output_col = st.columns([1, 2.3])
    
    with input_col:
        st.subheader("Your Candidate Profile")
        u_skills = st.text_area("Paste Your CV Text or List Skills", placeholder="e.g. Python, Machine Learning, Data Analyst, React...", height=250)
        u_city = st.multiselect("Select Cities", df['location'].unique(), default=df['location'].unique())
        u_type = st.radio("Employment Type", ["All", "Internship", "Fresh Graduate"])
        match_btn = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with output_col:
        if match_btn and u_skills:
            results = compute_ai_matching(u_skills, df)
            
            # Applying Filters
            results = results[results['location'].isin(u_city)]
            if u_type != "All":
                results = results[results['type'] == u_type]
            
            st.subheader(f"Found {len(results)} Relevant Opportunities")
            
            for _, row in results.iterrows():
                if row['match_score'] > 2: # Show even low matches to show engine working
                    type_badge = "badge-intern" if row['type'] == "Internship" else "badge-fresh"
                    st.markdown(f"""
                    <div class="job-card">
                        <img src="{row['img']}" class="card-banner">
                        <div class="card-content">
                            <div style="display:flex; justify-content:space-between; align-items:start;">
                                <div>
                                    <h3 style="margin:0; color:#0f172a;">{row['title']}</h3>
                                    <p style="margin:0; color:#059669; font-weight:700;">{row['company']} • {row['location']}</p>
                                </div>
                                <div style="text-align:right;">
                                    <span class="badge {type_badge}">{row['type']}</span><br>
                                    <span class="badge badge-match" style="margin-top:8px;">{int(row['match_score'])}% Match</span>
                                </div>
                            </div>
                            <p style="margin-top:15px; font-size:0.9rem; color:#475569;">{row['desc']}</p>
                            <div style="font-weight:bold; font-size:0.85rem; color:#064e3b; margin-top:5px;">Stipend/Salary: {row['stipend']}</div>
                            <div class="skill-gap-box">
                                <b>💡 Skill Gap Detected:</b> {row['skill_gap']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter your skills or paste your resume text on the left to activate the AI Matching Engine.")

# --- PAGE: ANALYTICS ---
elif nav_choice == "📊 Skill Analytics":
    st.header("Pakistani Tech Industry Market Insights")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(px.pie(df, names='location', title='Job Distribution by Geographic Region', hole=0.4), use_container_width=True)
    with col_b:
        st.plotly_chart(px.bar(df, x='company', y='match_score' if 'match_score' in df else None, title="Mock Demand per Organization"), use_container_width=True)

st.markdown("---")
st.caption("Aror University Sukkur | Department of Artificial Intelligence | BS AI Semester 4 Final Project")

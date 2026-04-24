# ============================================================
#   JOB RECCOMANDATION SYSTEM
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Supervisor  : Sir Abdul Haseeb 
#   Project     : Final Semester Project (BS Artificial Intelligence)
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
st.set_page_config(page_title="Job Reccomandation System", layout="wide", page_icon="Aror Logo.jpg")

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
#  2. DATASET (Aligned with Final Semester IT/AI Roles)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_student_data():
    data = [
        {
            "id": 1, "title": "AI Research Engineer", "company": "Systems Ltd", 
            "location": "Lahore", "type": "Full Time", "salary": "Rs. 80,000",
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
            "id": 3, "title": "AI & Software Developer", "company": "10Pearls", 
            "location": "Karachi", "type": "Full Time", "salary": "Rs. 75,000",
            "skills": "Python, HTML, CSS, JavaScript, React, Machine Learning", 
            "desc": "Help develop intelligent web applications and responsive UI components.",
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
    st.image("Aror Logo.jpg", use_container_width=True)
    st.markdown("<h2 style='color:#065f46; font-size: 22px; text-align: center;'>🎓 Job Reccomandation System</h2>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠 Home", "🔍 Find Jobs", "📈 Market Analysis"])
    st.markdown("---")
    st.write("**Project Type:**")
    st.caption("Final Semester Project")
    st.write("**Institution:**")
    st.caption("Aror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. APP SECTIONS
# ──────────────────────────────────────────────────────────────

# --- HOME ---
if page == "🏠 Home":
    st.title("Job Reccomandation System")
    st.subheader("BS Artificial Intelligence | Final Semester Project")
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.write("""
        Welcome to our Final Semester Project demonstration. This system was designed to help students 
        and graduates at **Aror University** bridge the gap between academic learning and industry 
        requirements. By using **TF-IDF Vectorization** and **Cosine Similarity**, we analyze the semantic meaning 
        of your skills to find the most relevant jobs and opportunities in Pakistan.
        """)
        st.success("🎯 Goal: Improve job search accuracy via AI and Content-Based Filtering.")
    with col2:
        st.image("Aror Logo.jpg", use_container_width=True)

# --- SEARCH ENGINE ---
elif page == "🔍 Find Jobs":
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
            
            if results.empty:
                st.info("No jobs found in that specific city. Try another location!")
            
            for _, row in results.iterrows():
                if row['match_percent'] > 0:
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div style="display:flex; gap:15px;">
                                <div style="width: 80px; height: 80px; border-radius: 12px; background: linear-gradient(135deg, #065f46, #059669); color: white; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: bold; flex-shrink: 0; box-shadow: 0 4px 10px rgba(6, 95, 70, 0.2);">
                                    {row['company'][0]}
                                </div>
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
                            <span style="font-size:0.85rem; font-weight:bold; color:#065f46;">Salary Range: {row['salary']}</span>
                            <div class="skill-missing">⚠️ Missing Skills: {row['missing_skills']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# --- MARKET ANALYSIS ---
elif page == "📈 Market Analysis":
    st.header("Job Market Analytics")
    
    st.markdown("Here is an analysis of the current job market trends based on our dataset.")
    
    col1, col2 = st.columns(2)
    
    df_chart = df.copy()
    # Convert 'Rs. 80,000' to numeric for chart
    df_chart['salary_num'] = df_chart['salary'].apply(lambda x: int(re.sub(r'[^0-9]', '', x)))
    
    with col1:
        fig1 = px.bar(df_chart, x='location', y='salary_num', color='location', 
                      title="Average Salary by Location",
                      labels={'salary_num': 'Salary (PKR)', 'location': 'City'})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        fig2 = px.pie(df_chart, names='location', title="Opportunity Distribution by City", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info("💡 **Market Insights:** The analytics above reflect the current demand and compensation metrics for fresh graduates and entry-level professionals in Pakistan's tech sector.")

st.markdown("---")
st.caption("© 2026 | Aror University Sukkur | Department of Artificial Intelligence")

# ============================================================
#   TALENTMATCH AI: ALL-PAKISTAN ENTERPRISE EDITION
#   Institution : Aror University Sukkur
#   Student     : Waqaas Hussain (SAP-5000000291)
#   Subject     : Programming for AI (Sir Abdul Haseeb)
#   Algorithm   : TF-IDF Vectorization + Cosine Similarity
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
#  1. PREMIUM UI CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch PK | AI Recommender", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #064e3b, #022c22, #000000);
        color: #ecfdf5;
    }
    
    /* Premium Glassmorphism Cards */
    .job-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 24px;
        margin-bottom: 25px;
        transition: 0.4s ease;
    }
    .job-card:hover {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid #10b981;
        transform: translateY(-8px);
    }
    
    .match-val {
        background: linear-gradient(135deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 1.6rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        width: 100% !important;
    }
    
    code { color: #34d399 !important; background: rgba(16, 185, 129, 0.1) !important; padding: 4px 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. ALL-PAKISTAN DATASET (Extended Cities)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_pakistan_db():
    data = [
        {"title": "AI Research Engineer", "company": "Systems Ltd", "location": "Lahore", "salary": 280000, "skills": "Python, NLP, PyTorch, Scikit-learn", "desc": "Leading global AI innovation and neural architecture research."},
        {"title": "Senior Data Scientist", "company": "Afiniti", "location": "Karachi", "salary": 350000, "skills": "SQL, Python, Statistics, Machine Learning, AWS", "desc": "Advanced behavioral matching algorithms for enterprise scale."},
        {"title": "AI Developer Intern", "company": "Aror Solutions", "location": "Sukkur", "salary": 25000, "skills": "Python, React, API, Git, Data Structures", "desc": "Developing AI integration tools for local Sindh industries."},
        {"title": "Cloud Architect", "company": "NetSol", "location": "Islamabad", "salary": 310000, "skills": "AWS, Docker, Kubernetes, Linux, Terraform", "desc": "Scaling financial cloud ecosystems globally."},
        {"title": "Junior ML Engineer", "company": "Folio3", "location": "Karachi", "salary": 125000, "skills": "Python, Computer Vision, Git, Django, OpenCV", "desc": "Building Computer Vision models for AgTech solutions."},
        {"title": "Data Analyst", "company": "Contour Software", "location": "Lahore", "salary": 180000, "skills": "SQL, Excel, Python, PowerBI, Statistics", "desc": "Market intelligence and data mining for global clients."},
        {"title": "Web Dev (AI Integrations)", "company": "Symmetry Group", "location": "Karachi", "salary": 140000, "skills": "JavaScript, React, Python, OpenAI API", "desc": "Transforming marketing via generative AI tools."},
        {"title": "DevOps Engineer", "company": "NorthBay Solutions", "location": "Islamabad", "salary": 240000, "skills": "AWS, CI/CD, Jenkins, Python, Linux", "desc": "Modernizing infrastructure for data-intensive apps."},
        {"title": "Junior AI Dev", "company": "TechVantage", "location": "Peshawar", "salary": 90000, "skills": "Python, ML, Flask, SQL, Git", "desc": "Building intelligent automation for local startups."},
        {"title": "Full Stack Dev", "company": "Devsinc", "location": "Faisalabad", "salary": 130000, "skills": "MERN Stack, Python, AWS, Docker", "desc": "Developing scalable web services in central Punjab."}
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. AI ENGINE (TF-IDF & Cosine Similarity)
# ──────────────────────────────────────────────────────────────
def run_ai_logic(input_text, df):
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = df['title'] + " " + df['skills'] + " " + df['desc']
    tfidf_matrix = tfidf.fit_transform(corpus.apply(clean))
    
    user_vec = tfidf.transform([clean(input_text)])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['score'] = scores * 100
    
    user_words = set(clean(input_text).split())
    def find_gap(row_skills):
        required = set([s.strip().lower() for s in row_skills.split(',')])
        gap = required - user_words
        return ", ".join(list(gap)).title() if gap else "Ready!"
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR (All Cities Filter)
# ──────────────────────────────────────────────────────────────
df_main = load_pakistan_db()

with st.sidebar:
    st.markdown("<h1 style='color:#10b981;'>TalentMatch PK</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=70)
    st.markdown("---")
    
    st.subheader("👨‍🎓 Candidate Profile")
    u_name = st.text_input("Candidate Name", "Waqaas Hussain")
    
    # Input Method Toggle
    input_mode = st.radio("Input Method", ["Quick Skill Entry", "Full Resume Analysis"])
    if input_mode == "Quick Skill Entry":
        u_data = st.text_input("Technical Skills", placeholder="e.g. Python, SQL, ML")
    else:
        u_data = st.text_area("Paste Your CV Content", placeholder="Copy and paste resume text here...", height=250)
    
    # All Cities Selection
    u_loc = st.selectbox("Preferred City", ["All Cities"] + sorted(list(df_main['location'].unique())))
    
    st.markdown("---")
    search_btn = st.button("Generate Recommendations")
    st.caption(f"Project by {u_name} | Aror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. DASHBOARD & ANALYTICS
# ──────────────────────────────────────────────────────────────
st.title("Digital Pakistan AI Career Hub")
st.write(f"Instructor: **Sir Abdul Haseeb** | **Semester 4 BS AI Submission**")

if search_btn and u_data:
    results = run_ai_logic(u_data, df_main)
    
    if u_loc != "All Cities":
        results = results[results['location'] == u_loc]
        
    tab_res, tab_vis = st.tabs(["🎯 Top Matches", "📊 Market Analytics"])
    
    with tab_res:
        st.subheader(f"Ranked Matches across {u_loc}")
        for _, row in results.iterrows():
            if row['score'] > 2:
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div>
                            <h2 style="margin:0; color:#10b981;">{row['title']}</h2>
                            <p style="margin:0; opacity:0.8; font-weight:600;">{row['company']} • {row['location']}</p>
                        </div>
                        <div class="match-val">{int(row['score'])}% Match</div>
                    </div>
                    <p style="margin-top:15px; font-size:0.95rem; color:#d1d5db; line-height:1.6;">{row['desc']}</p>
                    <div style="margin-top:20px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:15px;">
                        <span style="font-size:0.85rem; color:#94a3b8; font-weight:bold; letter-spacing:1px;">⚠️ MISSING SKILLS:</span><br>
                        <span style="color:#f87171; font-weight:600; font-size:0.9rem;">{row['gap']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_vis:
        st.subheader("National Tech Industry Insights")
        col_l, col_r = st.columns(2)
        
        with col_l:
            fig, ax = plt.subplots(facecolor='none')
            sns.barplot(data=df_main, x='location', y='salary', palette='Greens_d', ax=ax)
            ax.set_title("Salary Benchmarks by City", color='white', weight='bold')
            ax.tick_params(colors='white', rotation=45)
            st.pyplot(fig)
            
        with col_r:
            fig2, ax2 = plt.subplots(facecolor='none')
            city_counts = df_main['location'].value_counts()
            plt.pie(city_counts, labels=city_counts.index, autopct='%1.1f%%', colors=sns.color_palette('Greens_d'))
            ax2.set_title("Job Density (Market Share)", color='white', weight='bold')
            st.pyplot(fig2)
else:
    st.info("👋 Welcome! Use the sidebar to analyze your profile and see AI recommendations across Pakistan.")
    st.markdown("### Pakistan Tech Market Snapshot")
    st.bar_chart(df_main.groupby('location')['salary'].mean())

st.markdown("---")
st.caption("BS AI Semester 4 | Aror University Sukkur | All-Pakistan Edition")

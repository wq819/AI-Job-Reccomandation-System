# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (PROFESSIONAL VERSION)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Sir Abdul Haseeb)
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
#  1. GLOBAL CONFIGURATION & UI STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI | Student Edition", layout="wide", page_icon="🎓")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f8fafc; }
    
    /* Custom Card Design */
    .job-card {
        background: white; border-radius: 20px; border: 1px solid #e2e8f0;
        margin-bottom: 25px; overflow: hidden; transition: 0.4s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .job-card:hover { transform: translateY(-5px); border-color: #059669; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }
    
    .loc-banner { width: 100%; height: 180px; object-fit: cover; }
    .card-body { padding: 25px; }
    
    /* Badges */
    .badge { padding: 5px 14px; border-radius: 8px; font-weight: 700; font-size: 0.75rem; display: inline-block; }
    .badge-intern { background: #dcfce7; color: #166534; }
    .badge-fresh { background: #fef3c7; color: #92400e; }
    .badge-match { background: #dbeafe; color: #1e40af; }
    
    /* Skill Gap Highlight */
    .gap-alert {
        background: #fff1f2; border-left: 5px solid #e11d48;
        padding: 12px; margin-top: 15px; border-radius: 8px;
        font-size: 0.85rem; color: #9f1239;
    }
    
    .sidebar-branding {
        padding: 20px; background: linear-gradient(135deg, #064e3b, #065f46);
        border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. PAKISTANI STUDENT JOB DATABASE
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_comprehensive_database():
    data = [
        {
            "id": 1, "title": "AI Research Intern", "company": "Systems Ltd", 
            "location": "Lahore", "type": "Internship", "stipend": "Rs. 30,000",
            "skills": "Python, Machine Learning, Data Cleaning, Scikit-learn", 
            "desc": "Join our AI lab to assist in preprocessing large datasets and tuning NLP models.",
            "img": "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=800"
        },
        {
            "id": 2, "title": "Junior Data Scientist", "company": "Afiniti", 
            "location": "Karachi", "type": "Fresh Graduate", "stipend": "Rs. 95,000",
            "skills": "SQL, Python, Statistics, Machine Learning, R", 
            "desc": "Apply behavioral matching AI to solve complex customer interaction challenges.",
            "img": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800"
        },
        {
            "id": 3, "title": "Python Developer (AI/Web)", "company": "Aror Solutions", 
            "location": "Sukkur", "type": "Internship", "stipend": "Rs. 20,000",
            "skills": "Python, FastAPI, HTML, CSS, OpenAI API", 
            "desc": "Help integrate AI models into local web applications for the Sindh region.",
            "img": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=800"
        },
        {
            "id": 4, "title": "Cloud Security Intern", "company": "NetSol", 
            "location": "Islamabad", "type": "Internship", "stipend": "Rs. 25,000",
            "skills": "Linux, AWS, CyberSecurity, Docker, Bash", 
            "desc": "Collaborate with our DevOps team to secure cloud-native infrastructures.",
            "img": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=800"
        },
        {
            "id": 5, "title": "Junior Full Stack Dev", "company": "Folio3", 
            "location": "Karachi", "type": "Fresh Graduate", "stipend": "Rs. 80,000",
            "skills": "React, Node.js, JavaScript, MongoDB, Git", 
            "desc": "Develop and maintain high-performance web applications for healthcare tech.",
            "img": "https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=800"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. AI MATCHING ENGINE LOGIC
# ──────────────────────────────────────────────────────────────
def run_ai_match(user_text, df):
    if not user_text.strip(): return pd.DataFrame()
    
    # Text Normalization
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(corpus.apply(clean))
    
    # Similarity Calculation
    user_vec = tfidf.transform([clean(user_text)])
    df['match_score'] = cosine_similarity(user_vec, matrix).flatten() * 100
    
    # Skill Gap Detection
    def get_gap(required):
        req_list = [s.strip().lower() for s in required.split(',')]
        user_skills = user_text.lower()
        gap = [s for s in req_list if s not in user_skills]
        return ", ".join(gap).title() if gap else "None! You're a perfect match."
    
    df['missing_skills'] = df['skills'].apply(get_gap)
    return df.sort_values('match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
df = get_comprehensive_database()

with st.sidebar:
    st.markdown('<div class="sidebar-branding"><h2>TalentMatch AI</h2><p>Aror University Sukkur</p></div>', unsafe_allow_html=True)
    nav = st.radio("Selection", ["🏠 Dashboard", "🔍 Smart Matcher", "📊 Skill Analytics"])
    st.markdown("---")
    st.write("**Developed By:**")
    st.caption("Waqaas Hussain & Hira Abdul Hafeez")
    st.caption("BS AI - 4th Semester")

# ──────────────────────────────────────────────────────────────
#  5. MAIN APP INTERFACE
# ──────────────────────────────────────────────────────────────

# --- DASHBOARD ---
if nav == "🏠 Dashboard":
    st.title("Connecting Students with Pakistan's Tech Industry")
    st.image("https://images.unsplash.com/photo-1522202176988-66273c2fd55f?q=80&w=1200", caption="BS Artificial Intelligence Project")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Listings", len(df))
    col2.metric("Tech Hubs", "Karachi, Lahore, ISB, Sukkur")
    col3.metric("Matching Algorithm", "TF-IDF + Cosine")
    
    st.markdown("---")
    st.info("💡 **Welcome Student!** Use the 'Smart Matcher' to paste your skills and see which internships fit your profile.")

# --- SMART MATCHER ---
elif nav == "🔍 Smart Matcher":
    st.header("AI-Powered Job Recommendation")
    
    left, right = st.columns([1, 2.3])
    
    with left:
        st.subheader("Your Candidate Profile")
        u_skills = st.text_area("List Your Skills or Bio", placeholder="e.g. Python, Machine Learning, HTML, SQL...", height=200)
        u_city = st.multiselect("Preferred Cities", df['location'].unique(), default=df['location'].unique())
        u_type = st.radio("Opportunity Type", ["All", "Internship", "Fresh Graduate"])
        process = st.button("Generate Recommendations", type="primary", use_container_width=True)

    with right:
        if process and u_skills:
            results = run_ai_match(u_skills, df)
            
            # Applying Filters
            results = results[results['location'].isin(u_city)]
            if u_type != "All":
                results = results[results['type'] == u_type]
            
            st.subheader(f"Found {len(results)} Matching Roles")
            
            for _, row in results.iterrows():
                if row['match_score'] > 2:
                    badge_type = "badge-intern" if row['type'] == "Internship" else "badge-fresh"
                    st.markdown(f"""
                    <div class="job-card">
                        <img src="{row['img']}" class="loc-banner">
                        <div class="card-body">
                            <div style="display:flex; justify-content:space-between; align-items:start;">
                                <div>
                                    <h3 style="margin:0; color:#0f172a;">{row['title']}</h3>
                                    <p style="margin:0; color:#059669; font-weight:700;">{row['company']} • {row['location']}</p>
                                </div>
                                <div style="text-align:right;">
                                    <span class="badge {badge_type}">{row['type']}</span>
                                    <span class="badge badge-match">{int(row['match_score'])}% Match</span>
                                </div>
                            </div>
                            <p style="margin-top:15px; font-size:0.9rem; color:#475569;">{row['desc']}</p>
                            <div style="font-weight:bold; font-size:0.85rem; color:#065f46; margin-top:5px;">Estimated Stipend: {row['stipend']}</div>
                            <div class="gap-alert">
                                <b>💡 Skill Gap:</b> To improve your chances, learn: {row['missing_skills']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Enter your technical skills to begin the AI matching analysis.")

# --- ANALYTICS ---
elif nav == "📊 Skill Analytics":
    st.header("Pakistani Tech Market Statistics")
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='location', title='Job Distribution by City', hole=0.4), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(df, x='company', y='match_score' if 'match_score' in df else None, title="Example Demand by Company (Mockup)"), use_container_width=True)

st.markdown("---")
st.caption("Aror University Sukkur | Department of Artificial Intelligence | Instructor: Sir Abdul Haseeb")

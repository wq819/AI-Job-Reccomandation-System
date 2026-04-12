# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (PROFESSIONAL STUDENT EDITION)
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain & Hira Abdul Hafeez
#   Course      : Programming for AI (Sir Abdul Haseeb)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re

# ──────────────────────────────────────────────────────────────
#  1. GLOBAL CONFIG & UI STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI Pro", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Main Card Design */
    .job-card {
        background: white; border-radius: 16px; border: 1px solid #e2e8f0;
        margin-bottom: 20px; overflow: hidden; transition: 0.3s ease;
        display: flex; flex-direction: column;
    }
    .job-card:hover { border-color: #059669; transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.05); }
    
    /* Image Section */
    .loc-banner { width: 100%; height: 150px; object-fit: cover; }
    
    /* Labels */
    .badge { padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.75rem; }
    .badge-intern { background: #dcfce7; color: #166534; }
    .badge-full { background: #fef3c7; color: #92400e; }
    .badge-match { background: #dbeafe; color: #1e40af; }
    
    /* Skill Gap Styling */
    .gap-box { background: #fff1f2; border-left: 4px solid #e11d48; padding: 10px; margin-top: 10px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. COMPREHENSIVE PAKISTANI JOB DATABASE
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_comprehensive_db():
    data = [
        {
            "id": 1, "title": "AI Research Intern", "company": "Systems Ltd", 
            "location": "Lahore", "category": "AI/ML", "type": "Internship", "stipend": "Rs. 30,000",
            "skills": "Python, Machine Learning, Data Cleaning, Scikit-learn", 
            "desc": "Join the AI Lab to work on enterprise NLP and Computer Vision models.",
            "img": "https://images.unsplash.com/photo-1590059530472-87034f593322?q=80&w=600"
        },
        {
            "id": 2, "title": "Junior Data Analyst", "company": "Symmetry Group", 
            "location": "Sukkur", "category": "Data Science", "type": "Fresh Graduate", "stipend": "Rs. 65,000",
            "skills": "SQL, Excel, Python, PowerBI, Statistics", 
            "desc": "Help local businesses in Sindh transform their operations through data insights.",
            "img": "https://images.unsplash.com/photo-1595905584523-999e4f3a3848?q=80&w=600"
        },
        {
            "id": 3, "title": "Cloud Computing Associate", "company": "NetSol", 
            "location": "Islamabad", "category": "DevOps", "type": "Fresh Graduate", "stipend": "Rs. 90,000",
            "skills": "AWS, Linux, Docker, Python, Bash", 
            "desc": "Manage cloud infrastructure for global automotive software solutions.",
            "img": "https://images.unsplash.com/photo-1627581555541-1979965d1b71?q=80&w=600"
        },
        {
            "id": 4, "title": "Front-End Developer Intern", "company": "10Pearls", 
            "location": "Karachi", "category": "Web Dev", "type": "Internship", "stipend": "Rs. 25,000",
            "skills": "HTML, CSS, JavaScript, React, Tailwind", 
            "desc": "Collaborate with senior developers to build modern, responsive interfaces.",
            "img": "https://images.unsplash.com/photo-1568205706871-332308933220?q=80&w=600"
        },
        {
            "id": 5, "title": "Python Developer", "company": "Folio3", 
            "location": "Karachi", "category": "Backend", "type": "Fresh Graduate", "stipend": "Rs. 75,000",
            "skills": "Python, Django, PostgreSQL, REST API", 
            "desc": "Build scalable backend services for agriculture-tech and healthcare apps.",
            "img": "https://images.unsplash.com/photo-1555066931-4365d14bab8c?q=80&w=600"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. THE CORE AI ENGINE
# ──────────────────────────────────────────────────────────────
def run_ai_engine(user_query, df):
    if not user_query.strip(): return pd.DataFrame()
    
    # Text Cleaning
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
    # Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = df['title'] + " " + df['skills'] + " " + df['desc'] + " " + df['category']
    matrix = tfidf.fit_transform(corpus.apply(clean))
    
    # Matching
    user_vec = tfidf.transform([clean(user_query)])
    df['match_score'] = cosine_similarity(user_vec, matrix).flatten() * 100
    
    # Skill-Gap Analysis
    def get_gap(required):
        req_list = [s.strip().lower() for s in required.split(',')]
        user_input = user_query.lower()
        gap = [s for s in req_list if s not in user_input]
        return ", ".join(gap).title() if gap else "Ready to Apply!"
    
    df['missing'] = df['skills'].apply(get_gap)
    return df.sort_values('match_score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────
df = get_comprehensive_db()

with st.sidebar:
    st.markdown("""<div style='text-align:center; padding:10px; background:#065f46; border-radius:10px; color:white;'>
                <h2 style='margin:0;'>TalentMatch AI</h2><p style='font-size:0.8rem;'>Aror University Edition</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    choice = st.radio("Explore System", ["🏠 Student Dashboard", "🔍 Smart AI Search", "📊 Market Analytics", "📑 My Proposal"])
    st.markdown("---")
    st.markdown(f"**Authors:**\nWaqaas Hussain\nHira Abdul Hafeez")
    st.caption("SAP: 5000000291, 5000000314")

# ──────────────────────────────────────────────────────────────
#  5. MAIN APP INTERFACE
# ──────────────────────────────────────────────────────────────

# --- DASHBOARD ---
if choice == "🏠 Student Dashboard":
    st.title("Connecting Students to the Industry")
    st.image("https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=1200", caption="BS AI 4th Semester Project")
    
    st.markdown("""
    ### Project Overview
    This system utilizes **Natural Language Processing (NLP)** to solve the problem of keyword-matching inefficiencies. 
    By creating a high-dimensional vector space using **TF-IDF**, we can measure the semantic distance between your 
    academic profile and real-world job requirements in Pakistan.
    """)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Roles", len(df))
    c2.metric("Major Cities", "4")
    c3.metric("AI Categories", "5")

# --- SMART SEARCH ---
elif choice == "🔍 Smart AI Search":
    st.header("Personalized Matching Engine")
    
    l, r = st.columns([1, 2.5])
    
    with l:
        st.subheader("Your Skills & Bio")
        u_text = st.text_area("Paste your CV text or skills here", placeholder="e.g. Python, Machine Learning, SQL...", height=200)
        u_city = st.multiselect("Filter by City", df['location'].unique(), default=df['location'].unique())
        u_type = st.radio("Employment Type", ["All", "Internship", "Fresh Graduate"])
        process = st.button("Generate Recommendations", type="primary")

    with r:
        if process and u_text:
            results = run_ai_engine(u_text, df)
            
            # Apply Filters
            results = results[results['location'].isin(u_city)]
            if u_type != "All":
                results = results[results['type'] == u_type]
            
            st.subheader(f"Top {len(results)} Recommendations")
            for _, row in results.iterrows():
                if row['match_score'] > 0:
                    badge_class = "badge-intern" if row['type'] == "Internship" else "badge-full"
                    st.markdown(f"""
                    <div class="job-card">
                        <img src="{row['img']}" class="loc-banner">
                        <div style="padding:20px;">
                            <div style="display:flex; justify-content:space-between; align-items:start;">
                                <div>
                                    <h3 style="margin:0;">{row['title']}</h3>
                                    <p style="margin:0; color:#64748b; font-weight:700;">{row['company']} • {row['location']}</p>
                                </div>
                                <div style="text-align:right;">
                                    <span class="badge {badge_class}">{row['type']}</span>
                                    <span class="badge badge-match">{int(row['match_score'])}% Match</span>
                                </div>
                            </div>
                            <p style="margin-top:15px; font-size:0.9rem; color:#334155;">{row['desc']}</p>
                            <div style="margin-top:10px;">
                                <span style="font-size:0.8rem; font-weight:bold; color:#065f46;">Estimated Stipend: {row['stipend']}</span>
                            </div>
                            <div class="gap-box">
                                <span style="font-size:0.8rem; font-weight:700; color:#e11d48;">💡 SKILL GAP:</span>
                                <span style="font-size:0.8rem; color:#881337;"> {row['missing']}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Please enter your skills in the left panel to trigger the AI analysis.")

# --- ANALYTICS ---
elif choice == "📊 Market Analytics":
    st.header("Pakistani Tech Market Insights")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig1 = px.pie(df, names='category', title='Opportunities by Domain', hole=0.4, color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col_b:
        fig2 = px.bar(df, x='location', color='type', title='Job Density by City', barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

# --- PROPOSAL ---
elif choice == "📑 My Proposal":
    st.title("Project Documentation")
    st.markdown(f"""
    ### 1. Introduction
    Developed by **{st.sidebar.markdown('Waqaas Hussain & Hira Abdul Hafeez')}**, this system addresses the inefficiencies of keyword-based searching.
    
    ### 2. Methodology
    1. **Preprocessing:** Lowercasing, punctuation removal, and tokenization.
    2. **TF-IDF:** Transforms text into numerical vectors based on term importance.
    3. **Cosine Similarity:** Measures the angle between the user vector and the job vector.
    
    ### 3. Expected Outcomes
    - Improved personalization for Aror University students.
    - Identification of skill gaps for professional development.
    """)

st.markdown("---")
st.caption("Aror University Sukkur | Department of AI | Instructor: Sir Abdul Haseeb")

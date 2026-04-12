# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (PROFESSIONAL VER)
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
st.set_page_config(page_title="TalentMatch Pro | Aror University", layout="wide", page_icon="🏢")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .main { background-color: #f8fafc; }
    
    .header-container {
        background: linear-gradient(135deg, #1e1b4b, #4338ca);
        padding: 40px; border-radius: 20px; color: white;
        text-align: center; margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    .job-card {
        background: white; padding: 20px; border-radius: 12px;
        border: 1px solid #e2e8f0; margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .job-card:hover { transform: translateY(-3px); border-color: #4338ca; }
    
    .salary-tag {
        background: #f0fdf4; color: #166534;
        padding: 4px 12px; border-radius: 6px;
        font-weight: 600; font-size: 0.85rem;
    }
    .match-badge {
        background: #e0e7ff; color: #4338ca;
        padding: 5px 12px; border-radius: 20px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. ENHANCED DATASET (With Professional Metadata)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_professional_data():
    data = [
        {
            "id": 1, "title": "Senior AI Engineer", "company": "DeepMind", 
            "location": "Remote", "base_salary": 150000, "currency": "$",
            "category": "AI/ML", "skills": "Python, TensorFlow, PyTorch, Deep Learning", 
            "desc": "Lead neural architecture search and implement production-grade ML models.",
            "logo": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
            "tier": "Tier 1 - Tech Giant"
        },
        {
            "id": 2, "title": "Data Architect", "company": "DataCorp", 
            "location": "Karachi", "base_salary": 250000, "currency": "Rs.",
            "category": "Data Science", "skills": "SQL, Snowflake, Python, Machine Learning, ETL", 
            "desc": "Design enterprise data warehouses and predictive analytics pipelines.",
            "logo": "https://cdn-icons-png.flaticon.com/512/4248/4248873.png",
            "tier": "Tier 2 - Enterprise"
        },
        {
            "id": 3, "title": "Cybersecurity Lead", "company": "SecureNet", 
            "location": "Islamabad", "base_salary": 210000, "currency": "Rs.",
            "category": "Security", "skills": "Network Security, Linux, Ethical Hacking, Python", 
            "desc": "Orchestrate threat hunting and secure critical cloud infrastructure.",
            "logo": "https://cdn-icons-png.flaticon.com/512/1055/1055683.png",
            "tier": "Tier 1 - Security"
        },
        {
            "id": 4, "title": "Cloud Solutions Architect", "company": "CloudNine", 
            "location": "Remote", "base_salary": 135000, "currency": "$",
            "category": "DevOps", "skills": "AWS, Azure, Kubernetes, Terraform, Python", 
            "desc": "Design scalable cloud-native architectures and automation pipelines.",
            "logo": "https://cdn-icons-png.flaticon.com/512/1162/1162499.png",
            "tier": "Tier 2 - SaaS"
        }
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. SYSTEM LOGIC
# ──────────────────────────────────────────────────────────────
def preprocess_text(text):
    return re.sub(r'[^a-z0-9\s]', '', text.lower())

def calculate_dynamic_salary(base, match_score):
    # Higher match score simulates better negotiation leverage
    multiplier = 1.0 + (match_score / 1000) 
    return int(base * multiplier)

@st.cache_resource
def initialize_engine(df):
    tfidf = TfidfVectorizer(stop_words='english')
    content = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(content.apply(preprocess_text))
    return tfidf, matrix

# ──────────────────────────────────────────────────────────────
#  4. APP INTERFACE
# ──────────────────────────────────────────────────────────────
df = load_professional_data()
tfidf_vec, matrix = initialize_engine(df)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
    st.title("TalentMatch Pro")
    selection = st.radio("Menu", ["🏠 Home", "🔍 Smart Search", "📊 Market Insights", "📄 Documentation"])
    st.markdown("---")
    st.markdown("**Developers:**\nWaqaas Hussain & Hira Abdul Hafeez\n*Aror University Sukkur*")

# --- SECTION: HOME ---
if selection == "🏠 Home":
    st.markdown("""
    <div class="header-container">
        <h1>Enterprise AI Job Recommendation</h1>
        <p>Professional Matching via TF-IDF Vectorization & Salary Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Next-Gen Recruitment")
        st.write("Our system goes beyond keywords to understand the professional depth of your skills, providing estimated salary brackets and company tiering for a better career fit.")
    with c2:
        st.image("https://cdn-icons-png.flaticon.com/512/8074/8074470.png", width=250)

# --- SECTION: SMART SEARCH ---
elif selection == "🔍 Smart Search":
    st.title("AI-Driven Opportunity Search")
    l, r = st.columns([1, 2.5])
    
    with l:
        st.subheader("Candidate Profile")
        user_input = st.text_area("List Technical Skills", placeholder="e.g., Python, SQL, AWS...", height=150)
        loc = st.selectbox("Preferred Location", ["Any", "Remote", "Karachi", "Islamabad", "Lahore"])
        btn = st.button("Generate Recommendations", type="primary")

    with r:
        if btn and user_input:
            user_vec = tfidf_vec.transform([preprocess_text(user_input)])
            scores = cosine_similarity(user_vec, matrix).flatten()
            df['match'] = scores * 100
            
            results = df.sort_values('match', ascending=False)
            if loc != "Any":
                results = results[results['location'] == loc]
            
            st.subheader("Top Professional Matches")
            for _, row in results.iterrows():
                if row['match'] > 5:
                    dyn_sal = calculate_dynamic_salary(row['base_salary'], row['match'])
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div style="display:flex; gap:15px;">
                                <img src="{row['logo']}" width="50" style="border-radius:8px;">
                                <div>
                                    <h3 style="margin:0; color:#1e1b4b;">{row['title']}</h3>
                                    <p style="margin:0; color:#6366f1; font-weight:600;">{row['company']} • {row['tier']}</p>
                                    <p style="margin:0; color:#94a3b8; font-size:0.8rem;">📍 {row['location']}</p>
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div class="salary-tag">EST. {row['currency']}{dyn_sal:,}</div>
                                <div style="margin-top:8px;" class="match-badge">{int(row['match'])}% Match</div>
                            </div>
                        </div>
                        <p style="margin-top:15px; color:#475569; font-size:0.9rem;">{row['desc']}</p>
                        <div style="margin-top:10px;">
                            <code style="background:#f1f5f9; color:#4338ca; padding:4px 8px; border-radius:5px; font-size:0.8rem;">{row['skills']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Awaiting input. Please enter your skills in the left panel.")

# --- SECTION: ANALYTICS ---
elif selection == "📊 Market Insights":
    st.header("Talent Landscape Analytics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(df, names='category', title='Job Domains', hole=0.4), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(df, x='company', y='base_salary', color='tier', title='Salary Benchmarks by Company'), use_container_width=True)

# --- SECTION: DOCUMENTATION ---
elif selection == "📄 Documentation":
    st.header("Project Technicalities")
    st.markdown("""
    ### System Architecture
    1. **Vectorization**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to weigh skill importance.
    2. **Matching**: Cosine Similarity calculates the angular distance between candidate vectors and job vectors.
    3. **Professional Layer**: Integrated salary estimators based on match accuracy and company tiering.
    """)

st.markdown("---")
st.caption("© 2026 | Aror University Sukkur | Department of Artificial Intelligence")

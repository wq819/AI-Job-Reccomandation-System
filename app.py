# ============================================================
#   AI-BASED JOB RECOMMENDATION SYSTEM (Pro Version)
#   Author      : Waqaas Hussain
#   Institution : Aror University Sukkur
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
#  1. PAGE CONFIG & UI STYLING
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Job Matcher | Waqaas Hussain", layout="wide", page_icon="🤖")

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .job-card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border-left: 5px solid #2563eb; margin-bottom: 15px;
        transition: 0.3s;
    }
    .job-card:hover { transform: scale(1.01); border-left: 5px solid #059669; }
    .match-badge {
        background: #dcfce7; color: #166534;
        padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 0.9rem;
    }
    .skill-tag {
        background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe;
        padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  2. DATASET (Realistic AI/Tech Jobs)
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_job_data():
    data = [
        {"id": 1, "title": "AI Engineer", "company": "DeepMind", "location": "Remote", "category": "AI/ML", "skills": "Python, TensorFlow, PyTorch, Deep Learning", "desc": "Build neural networks and deploy LLMs."},
        {"id": 2, "title": "Data Scientist", "company": "DataCorp", "location": "Karachi", "category": "Data Science", "skills": "Python, SQL, Machine Learning, Statistics", "desc": "Analyze big data and create predictive models."},
        {"id": 3, "title": "Full Stack Developer", "company": "SoftSolutions", "location": "Remote", "category": "Web Dev", "skills": "React, Node.js, JavaScript, MongoDB, HTML, CSS", "desc": "End-to-end web application development."},
        {"id": 4, "title": "Backend Developer", "company": "FinTech", "location": "Lahore", "category": "Web Dev", "skills": "Python, Django, FastAPI, PostgreSQL, Docker", "desc": "Scalable API and database architecture."},
        {"id": 5, "title": "Cybersecurity Analyst", "company": "SecureNet", "location": "Islamabad", "category": "Security", "skills": "Linux, Ethical Hacking, Network Security, Python", "desc": "Securing digital infrastructure and pentesting."},
        {"id": 6, "title": "Cloud Architect", "company": "CloudNine", "location": "Remote", "category": "DevOps", "skills": "AWS, Azure, Kubernetes, Terraform", "desc": "Managing cloud infrastructure and CI/CD."},
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. AI ENGINE (Preprocessing & TF-IDF)
# ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@st.cache_resource
def build_vectorizer(df):
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    # Combine features for better semantic understanding
    corpus = df['title'] + " " + df['skills'] + " " + df['desc']
    matrix = tfidf.fit_transform(corpus.apply(clean_text))
    return tfidf, matrix

# ──────────────────────────────────────────────────────────────
#  4. APP LOGIC
# ──────────────────────────────────────────────────────────────
df = load_job_data()
tfidf_vec, job_matrix = build_vectorizer(df)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3858/3858684.png", width=80)
    st.title("AI Matcher v2.0")
    menu = st.radio("Navigation", ["🏠 Dashboard", "🔍 Match My Skills", "📊 Market Insights", "📖 Documentation"])
    st.markdown("---")
    st.write(f"**Dev:** Waqaas Hussain\n**Uni:** Aror University")

# --- DASHBOARD ---
if menu == "🏠 Dashboard":
    st.title("Next-Gen Job Recommendation System")
    st.info("Welcome, Waqaas! Use the AI engine to find jobs that match your technical profile.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Jobs", len(df))
    col2.metric("Algorithm", "TF-IDF")
    col3.metric("Similarity", "Cosine")
    
    st.markdown("### How it works?")
    st.write("Our AI analyzes the **semantic distance** between your skills and job descriptions using vector space modeling.")

# --- MATCH ENGINE ---
elif menu == "🔍 Match My Skills":
    st.header("AI Matching Engine")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        user_input = st.text_area("Enter your Skills/Bio", placeholder="e.g. I am a Python expert with knowledge of Machine Learning and SQL...", height=200)
        loc_filter = st.selectbox("Location Preference", ["All"] + list(df['location'].unique()))
        match_threshold = st.slider("Similarity Threshold (%)", 0, 100, 10)
        btn = st.button("Generate Recommendations", type="primary")

    with c2:
        if btn and user_input:
            # Transform user input to vector
            user_vec = tfidf_vec.transform([clean_text(user_input)])
            # Calculate Similarity
            sim_scores = cosine_similarity(user_vec, job_matrix).flatten()
            df['match_score'] = sim_scores * 100
            
            # Filter & Sort
            results = df[df['match_score'] >= match_threshold].sort_values(by='match_score', ascending=False)
            if loc_filter != "All":
                results = results[results['location'] == loc_filter]

            if not results.empty:
                st.success(f"Found {len(results)} matches!")
                for _, row in results.iterrows():
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:1.2rem; font-weight:700;">{row['title']}</span>
                            <span class="match-badge">{int(row['match_score'])}% Match</span>
                        </div>
                        <div style="color:#64748b; margin-bottom:10px;">🏢 {row['company']} | 📍 {row['location']}</div>
                        <p style="font-size:0.9rem;">{row['desc']}</p>
                        <div>{" ".join([f'<span class="skill-tag">{s.strip()}</span>' for s in row['skills'].split(',')])}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No matches found. Try adding more skills.")
        else:
            st.info("Waiting for input... Fill in your skills on the left.")

# --- ANALYTICS ---
elif menu == "📊 Market Insights":
    st.header("Dataset Analytics")
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.pie(df, names='category', title="Job Distribution", hole=0.4)
        st.plotly_chart(fig1)
    with col_b:
        fig2 = px.bar(df, x='location', color='category', title="Jobs by City")
        st.plotly_chart(fig2)

# --- DOCUMENTATION ---
elif menu == "📖 Documentation":
    st.header("Technical Overview")
    st.markdown("""
    ### 1. Mathematical Approach
    We use **Cosine Similarity** to measure the angle between two vectors in a multidimensional space:
    """)
    st.latex(r"Similarity(A, B) = \frac{A \cdot B}{\|A\| \|B\|}")
    
    st.markdown("""
    ### 2. Feature Extraction
    **TF-IDF** (Term Frequency-Inverse Document Frequency) is used to give weight to unique skills (like 'PyTorch') while ignoring common words (like 'the', 'is').
    
    ### 3. Tech Stack
    - **Frontend:** Streamlit
    - **Data Handling:** Pandas / Numpy
    - **ML Engine:** Scikit-Learn (Linear Models)
    """)

st.markdown("---")
st.caption("Developed by Waqaas Hussain | Aror University Sukkur | Department of AI")

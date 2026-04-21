# ============================================================
#   Job Recommendation System
#   Institution : Aror University Sukkur
#   Student     : Waqaas Hussain (SAP-5000000291)
#   Instructor  : Sir Abdul Haseeb (BS AI - Semester 4)
#   Core Logic  : NLP / Sentence Transformers & PDF Parsing
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import pdfplumber
import humanize
from typing import Set
from sentence_transformers import SentenceTransformer, util

class JobRecommendationEngine:
    """
    An official implementation of a Job Recommendation Engine using Semantic Search.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = self._load_model()

    @staticmethod
    @st.cache_resource
    def _load_model() -> SentenceTransformer:
        """Loads and caches the Sentence Transformer model."""
        return SentenceTransformer('all-MiniLM-L6-v2')

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans input text by removing special characters and lowering case."""
        return re.sub(r'[^a-z0-9\s]', '', text.lower())

    def calculate_fit(self, input_text: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the semantic fit between the candidate's CV and job descriptions.
        """
        # Combine Job Title and Skills for Context
        corpus = (df['title'] + " " + df['skills']).tolist()
        
        # Semantic Search via Sentence-Transformers
        job_embeddings = self.model.encode(corpus, convert_to_tensor=True)
        user_embedding = self.model.encode(input_text, convert_to_tensor=True)
        
        # Cosine Similarity Tensor
        cosine_scores = util.cos_sim(user_embedding, job_embeddings)[0]
        
        # Convert PyTorch tensor to numpy
        scores = cosine_scores.cpu().numpy()
        df['score'] = np.clip(scores * 100, 0, 100) # Normalize to 0-100%
        
        # Logic: Finding Skill Gaps
        user_tokens: Set[str] = set(self.clean_text(input_text).split())
        
        def find_gap(row_skills: str) -> str:
            required = set([s.strip().lower() for s in row_skills.split(',')])
            gap = required - user_tokens
            return ", ".join(list(gap)).title() if gap else "Ready!"
        
        df['gap'] = df['skills'].apply(find_gap)
        return df.sort_values(by='score', ascending=False)

@st.cache_data
def get_national_db() -> pd.DataFrame:
    """Returns the official dummy database of national jobs."""
    data = [
        {"title": "AI Research Scientist", "company": "Systems Ltd", "location": "Lahore", "salary": 280000, "skills": "Python, PyTorch, NLP, Scikit-learn, Research, Deep Learning"},
        {"title": "Senior Data Architect", "company": "Afiniti", "location": "Karachi", "salary": 350000, "skills": "SQL, Python, Statistics, Machine Learning, AWS, ETL, Big Data"},
        {"title": "ML Engineer (Vision)", "company": "Folio3", "location": "Karachi", "salary": 140000, "skills": "Python, Computer Vision, OpenCV, Git, Django, PyTorch"},
        {"title": "AI Web Developer", "company": "Aror Solutions", "location": "Sukkur", "salary": 125000, "skills": "JavaScript, React, API, Python, Tailwind, Fastapi"},
        {"title": "Cloud Security Expert", "company": "NetSol", "location": "Islamabad", "salary": 310000, "skills": "AWS, Docker, Kubernetes, Linux, Python, CI/CD, Cyber Security"},
        {"title": "Junior Data Analyst", "company": "Contour Software", "location": "Lahore", "salary": 160000, "skills": "SQL, Excel, Python, PowerBI, Statistics, Tableau"},
        {"title": "NLP Engineer", "company": "Aror Solutions", "location": "Sukkur", "salary": 180000, "skills": "Python, NLP, Transformers, Huggingface, Spacy, ML"},
        {"title": "Data Scientist", "company": "Jazz", "location": "Islamabad", "salary": 250000, "skills": "Python, Machine Learning, Data Visualization, SQL, Pandas, Scikit-learn"},
        {"title": "Prompt Engineer", "company": "TenPearls", "location": "Karachi", "salary": 150000, "skills": "LLM, Prompt Engineering, OpenAI, GPT, Python, Communication"},
        {"title": "Backend Developer", "company": "Systems Ltd", "location": "Lahore", "salary": 200000, "skills": "Python, Django, PostgreSQL, REST APIs, Git, Docker"},
        {"title": "GenAI Architect", "company": "Afiniti", "location": "Islamabad", "salary": 450000, "skills": "Python, LLM, LangChain, Vector Databases, RAG, Cloud Computing"}
    ]
    return pd.DataFrame(data)

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from an uploaded PDF file."""
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text

# ──────────────────────────────────────────────────────────────
#  GUI SETUP
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentMatch AI | Pro Edition", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    
    .stApp {
        background: radial-gradient(circle at top right, #064e3b, #022c22, #000000);
        color: #ecfdf5;
    }
    
    /* Premium Glass-Card */
    .job-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 35px;
        border-radius: 28px;
        margin-bottom: 25px;
        transition: 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .job-card:hover {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid #10b981;
        transform: translateY(-10px);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    .match-val {
        background: linear-gradient(135deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 1.8rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        height: 3.5em !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  MAIN APPLICATION FLOW
# ──────────────────────────────────────────────────────────────
engine = JobRecommendationEngine()
df_main = get_national_db()

with st.sidebar:
    st.markdown("<h1 style='color:#10b981;'>TalentMatch AI</h1>", unsafe_allow_html=True)
    # Using an official crest/badge icon
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055644.png", width=70)
    st.markdown("---")
    
    st.subheader("👨‍💻 Professional Profile")
    u_name = st.text_input("Candidate Name", "Waqaas Hussain")
    
    cv_pdf = st.file_uploader("Upload Official CV (PDF)", type=["pdf"])
    extracted_text = ""
    
    if cv_pdf is not None:
        extracted_text = extract_text_from_pdf(cv_pdf)
        st.success("Official Document Parsed Successfully!")
        
    u_input = st.text_area("Or Paste Full CV / Resume Content", value=extracted_text, placeholder="e.g. Python Developer with experience in ML...", height=200)
    u_loc = st.selectbox("Market Focus", ["All Pakistan"] + sorted(list(df_main['location'].unique())))
    
    st.markdown("---")
    trigger = st.button("Generate Opportunities")
    st.caption(f"Official Project by {u_name}\nAror University Sukkur")

st.title("Job Recommendation System 🏛️")
st.write(f"Instructor: **Sir Abdul Haseeb** | **BS AI Semester 4 Final Project**")

# Hero Stats
m1, m2, m3, m4 = st.columns(4)
m1.metric("Available Opportunities", humanize.intword(len(df_main)))
m2.metric("Top Corporate Hub", "Karachi")
m3.metric("AI Market Demand", "High")
# Using humanize for average salary formatting
avg_salary = df_main['salary'].mean()
m4.metric("Avg Salary", f"PKR {humanize.intcomma(int(avg_salary))}")

st.markdown("<br>", unsafe_allow_html=True)

if trigger and u_input:
    results = engine.calculate_fit(u_input, df_main)
    if u_loc != "All Pakistan":
        results = results[results['location'] == u_loc]
        
    res_tab, vis_tab = st.tabs(["🎯 Top Matched Roles", "📊 Market Analytics"])
    
    with res_tab:
        st.subheader(f"Ranked Corporate Recommendations for {u_name}")
        for _, row in results.iterrows():
            if row['score'] > 2:
                # Format salary using humanize for readability
                formatted_salary = f"PKR {humanize.intcomma(row['salary'])}"
                
                st.markdown(f"""
                <div class="job-card">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div>
                            <h2 style="margin:0; color:#10b981;">{row['title']}</h2>
                            <p style="margin:0; opacity:0.8; font-weight:600;">{row['company']} • {row['location']} • {formatted_salary}</p>
                        </div>
                        <div class="match-val">{int(row['score'])}% Match</div>
                    </div>
                    <div style="margin-top:20px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:15px;">
                        <span style="font-size:0.85rem; color:#94a3b8; font-weight:bold;">💡 SKILL GAP ANALYSIS:</span><br>
                        <span style="color:#f87171; font-weight:600; font-size:0.95rem;">{row['gap']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with vis_tab:
        st.subheader("National Official Market Analytics")
        cl, cr = st.columns(2)
        
        salary_data = df_main.groupby('location', as_index=False)['salary'].mean()
        job_counts = df_main['location'].value_counts().reset_index()
        job_counts.columns = ['location', 'count']

        with cl:
            fig = px.bar(salary_data, x='location', y='salary', 
                         title="Average Corporate Compensation by Region",
                         color='salary', color_continuous_scale='Greens',
                         labels={'location': 'Region', 'salary': 'Average Salary (PKR)'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            
        with cr:
            fig2 = px.pie(job_counts, names='location', values='count', 
                          title="Market Opportunity Distribution",
                          color_discrete_sequence=px.colors.sequential.Greens_r)
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)

with st.expander(" Official Algorithm Documentation "):
    st.write("""
    **Official Pipeline Architecture:**
    1. **Data Ingestion & Cleaning:** Extracts unstructured resume data and cleanses special characters to maintain data integrity.
    2. **Semantic Vectorization:** Leverages `all-MiniLM-L6-v2` via Sentence Transformers to map candidate skills into a high-dimensional vector space.
    3. **Cosine Similarity Computation:** Computes the mathematical angle between the candidate's embedding and market opportunities to determine semantic alignment.
    4. **Recommendation Engine:** Ranks the output in descending order of similarity, providing granular skill-gap analysis.
    """)
    
st.markdown("---")
st.caption("Official BS AI Semester 4 Submission | Aror University Sukkur |")

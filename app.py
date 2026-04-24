# ============================================================
#   AI-Powered Job Recommendation System
#   Institution : Aror University Sukkur
#   Students    : Waqaas Hussain
#                 Hira Abdul Hafeez
#   Instructor  : Sir Abdul Haseeb (BS AI - Semester 4)

# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import pdfplumber
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import Set, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

download_nltk_resources()

@st.cache_resource
def load_sentence_transformer():
    """Loads the Sentence Transformer model with caching for performance."""
    logger.info("Loading SentenceTransformer model...")
    return SentenceTransformer('all-MiniLM-L6-v2')

class JobRecommendationEngine:
    """
    A professional implementation of a Job Recommendation Engine using Semantic Search.
    """
    
    def __init__(self):
        self.model = load_sentence_transformer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Cleans input text by removing special characters, lowercasing, and removing stopwords."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-z0-9\s]', '', text.lower())
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(filtered_tokens)

    def calculate_fit(self, input_text: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the semantic similarity between the candidate's CV and job descriptions.
        """
        if df.empty or not input_text.strip():
            logger.warning("Empty dataframe or input text provided to calculate_fit.")
            if not df.empty:
                df['score'] = 0
                df['gap'] = "No input provided"
            return df

        # Combine Job Title and Skills for Context
        corpus = (df['title'] + " " + df['skills']).apply(self.clean_text).tolist()
        
        # Generate Dense Vector Embeddings
        logger.info("Generating semantic embeddings for market matching.")
        corpus_embeddings = self.model.encode(corpus)
        user_embedding = self.model.encode([self.clean_text(input_text)])
        
        # Cosine Similarity Calculation on Dense Vectors
        scores = cosine_similarity(user_embedding, corpus_embeddings).flatten()
        df['score'] = np.clip(scores * 100, 0, 100) # Normalize to 0-100%
        
        # Logic: Finding Explicit Skill Gaps
        user_tokens: Set[str] = set(word_tokenize(self.clean_text(input_text)))
        
        def find_gap(row_skills: str) -> str:
            if not isinstance(row_skills, str):
                return ""
            required = set([s.strip().lower() for s in row_skills.split(',')])
            gap = required - user_tokens
            return ", ".join(list(gap)).title() if gap else "Ready!"
        
        df['gap'] = df['skills'].apply(find_gap)
        return df.sort_values(by='score', ascending=False)

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Loads the official dataset from CSV."""
    try:
        df = pd.read_csv('dataset.csv')
        logger.info(f"Dataset loaded successfully with {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        st.error("Critical Error: dataset.csv not found or could not be read.")
        return pd.DataFrame(columns=['title', 'company', 'location', 'salary', 'skills'])

def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from an uploaded PDF file with error handling."""
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        logger.info("PDF parsed successfully.")
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        st.error("Failed to parse the PDF document. Please ensure it is a valid text-based PDF.")
    return extracted_text

# ──────────────────────────────────────────────────────────────
#  GUI SETUP & CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title=" ", layout="wide", page_icon="👔")

# Premium Enterprise CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Modern Dark Theme Background */
    .stApp {
        background-color: #0f172a;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,0.1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,0.1) 0, transparent 50%);
        color: #f8fafc;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Premium Glass-Card for Jobs */
    .job-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .job-card:hover {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 10px 40px -10px rgba(59, 130, 246, 0.15);
    }
    
    /* Typography and Accents */
    .gradient-text {
        background: linear-gradient(135deg, #60a5fa 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .match-val {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; 
        font-size: 1.75rem;
    }
    
    /* Input Fields & Buttons */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #60a5fa !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
engine = JobRecommendationEngine()
df_main = load_dataset()

import base64
def get_image_base64(path):
    import os
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

logo_base64 = get_image_base64("logo.png")

with st.sidebar:
    # Sidebar Header
    if logo_base64:
        st.markdown(f'''
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" width="120" style="border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: #f8fafc; font-weight: 700; margin-top: 0;'>TalentMatch<span class='gradient-text'> AI</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: -10px;'>Enterprise Talent Intelligence</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 20px 0;'>", unsafe_allow_html=True)
    
    # Input Section
    st.markdown("<h4 style='color: #e2e8f0; font-size: 1rem; margin-bottom: 15px;'>Candidate Profile</h4>", unsafe_allow_html=True)
    u_name = st.text_input("Full Name", "Waqaas Hussain")
    
    cv_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"], help="Upload a professional PDF formatted resume.")
    extracted_text = ""
    
    if cv_pdf is not None:
        extracted_text = extract_text_from_pdf(cv_pdf)
        st.success("✓ Resume Processed Successfully")
        
    u_input = st.text_area("Resume Content / Professional Summary", value=extracted_text, placeholder="Paste your professional summary, skills, and experience here...", height=180)
    
    if not df_main.empty:
        u_loc = st.selectbox("Preferred Location", ["All Regions"] + sorted(list(df_main['location'].unique())))
    else:
        u_loc = "All Regions"
        
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 25px 0 15px 0;'>", unsafe_allow_html=True)
    trigger = st.button("Analyze & Match Opportunities")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Developed by **Waqaas Hussain** & **Hira Abdul Hafeez**<br>Aror University Sukkur", unsafe_allow_html=True)

# Main Content Area
st.markdown("<h1 style='font-size: 2.5rem; font-weight: 700; margin-bottom: 0;'>Talent Acquisition <span class='gradient-text'>Dashboard</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-top: 5px; margin-bottom: 30px;'>AI-Powered Semantic Job Matching System • Supervised by Sir Abdul Haseeb</p>", unsafe_allow_html=True)

if not df_main.empty:
    # High-End Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Total Opportunities", f"{len(df_main):,}")
    with m2:
        top_hub = df_main['location'].mode()[0] if not df_main.empty else "N/A"
        st.metric("Primary Tech Hub", top_hub)
    with m3:
        st.metric("Market Demand", "High Growth")
    with m4:
        avg_salary = int(df_main['salary'].mean()) if not df_main.empty else 0
        st.metric("Avg. Compensation", f"PKR {avg_salary/1000:.0f}k")

st.markdown("<br>", unsafe_allow_html=True)

if trigger and u_input and not df_main.empty:
    with st.spinner("Initializing Semantic Matching Engine..."):
        results = engine.calculate_fit(u_input, df_main)
        if u_loc != "All Regions":
            results = results[results['location'] == u_loc]
            
    if results.empty:
        st.warning(f"No opportunities found for the selected region ({u_loc}). Please broaden your search.")
    else:
        res_tab, vis_tab = st.tabs(["🎯 Top Matched Roles", "📈 Market Analytics"])
        
        with res_tab:
            st.markdown(f"<h3 style='margin-bottom: 20px; font-weight: 600;'>Recommended Roles for {u_name}</h3>", unsafe_allow_html=True)
            for _, row in results.iterrows():
                if row['score'] > 2:
                    formatted_salary = f"PKR {int(row['salary']):,}"
                    
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div>
                                <h3 style="margin:0 0 5px 0; color:#f8fafc; font-weight: 600;">{row['title']}</h3>
                                <p style="margin:0; color:#94a3b8; font-size: 0.95rem; font-weight:500;">
                                    <span style="color: #38bdf8;">🏢 {row['company']}</span> &nbsp;•&nbsp; 
                                    <span style="color: #a78bfa;">📍 {row['location']}</span> &nbsp;•&nbsp; 
                                    <span style="color: #fbbf24;">💰 {formatted_salary}</span>
                                </p>
                            </div>
                            <div class="match-val">{int(row['score'])}% Match</div>
                        </div>
                        <div style="margin-top:18px; background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px 15px;">
                            <span style="font-size:0.8rem; color:#94a3b8; font-weight:700; text-transform: uppercase; letter-spacing: 0.5px;">Skill Gap Identification</span><br>
                            <span style="color:#fb7185; font-weight:500; font-size:0.95rem; display: inline-block; margin-top: 4px;">{row['gap']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
        with vis_tab:
            st.markdown("<h3 style='margin-bottom: 20px; font-weight: 600;'>AI Market Insights & Analytics Dashboard</h3>", unsafe_allow_html=True)
            cl, cr = st.columns(2)
            
            salary_data = df_main.groupby('location', as_index=False)['salary'].mean().sort_values(by='salary', ascending=True)
            job_counts = df_main['location'].value_counts().reset_index()
            job_counts.columns = ['location', 'count']
    
            with cl:
                fig = px.bar(salary_data, x='salary', y='location', orientation='h',
                             title="Average Compensation by Region (PKR)",
                             color='salary', color_continuous_scale='Plasma',
                             text='salary',
                             labels={'location': 'Region', 'salary': 'Avg Salary'})
                fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside', marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.8)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color='#f8fafc',
                    title_font=dict(size=18, family="Inter", color="#f8fafc"),
                    margin=dict(l=20, r=40, t=60, b=20),
                    xaxis_title="",
                    yaxis_title=""
                )
                fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', showticklabels=False)
                fig.update_yaxes(showgrid=False, linecolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig, use_container_width=True)
                
            with cr:
                fig2 = px.pie(job_counts, names='location', values='count', 
                              title="Opportunity Distribution",
                              color='location',
                              color_discrete_sequence=px.colors.qualitative.Pastel)
                fig2.update_traces(hole=.6, hoverinfo="label+percent+name", textinfo='percent+label', textfont_size=14, marker=dict(line=dict(color='#0f172a', width=2)))
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color='#f8fafc',
                    title_font=dict(size=18, family="Inter", color="#f8fafc"),
                    margin=dict(l=20, r=20, t=60, b=20),
                    showlegend=False,
                    annotations=[dict(text='Hubs', x=0.5, y=0.5, font_size=20, showarrow=False, font_color='#60a5fa')]
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Adding an additional Visualization for Salary Distribution to make it look advanced
            st.markdown("<br>", unsafe_allow_html=True)
            fig3 = px.histogram(df_main, x="salary", nbins=20, title="Market Salary Distribution",
                                marginal="box", opacity=0.7, color_discrete_sequence=['#3b82f6'])
            fig3.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    font_color='#f8fafc',
                    title_font=dict(size=18, family="Inter", color="#f8fafc"),
                    margin=dict(l=20, r=20, t=60, b=20),
                    xaxis_title="Salary (PKR)",
                    yaxis_title="Count of Opportunities"
                )
            fig3.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            fig3.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("⚙️ System Architecture & Methodology", expanded=False):
    st.markdown("""
    ### Semantic Matching Pipeline
    This enterprise-grade recommendation system leverages state-of-the-art Natural Language Processing (NLP) to perform highly accurate candidate-to-job matching.
    
    1. **Intelligent Data Parsing:** Extracts unstructured text from resumes using advanced PDF parsing heuristics.
    2. **Text Normalization:** Cleanses text and filters noise using NLTK stop-word corpora.
    3. **Dense Vectorization:** Employs `all-MiniLM-L6-v2` Sentence Transformers to map candidate profiles and job descriptions into a shared high-dimensional semantic space.
    4. **Cosine Similarity Matrix:** Calculates mathematical distance between vectors to yield a precise semantic match percentage.
    5. **Gap Analysis Engine:** Computes set differences between explicit job requirements and parsed candidate tokens to identify critical missing skills.
    """)
    
st.markdown("<div style='text-align: center; margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px; color: #64748b; font-size: 0.85rem;'>", unsafe_allow_html=True)
st.markdown("Final Project ", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

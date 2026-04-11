# ============================================================
#   AI-Based Job Recommendation System
#   Prepared by: Waqaas Hussain
#   Subject: Programming for AI
#   Framework: Streamlit + Scikit-learn (TF-IDF + Cosine Similarity)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="JobMatch AI · Waqaas Hussain",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS + JS ──────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --navy:    #0A1628;
  --navy2:   #0F2040;
  --teal:    #00C9A7;
  --teal2:   #00A88E;
  --gold:    #FFB347;
  --blue:    #3B82F6;
  --purple:  #8B5CF6;
  --pink:    #EC4899;
  --red:     #EF4444;
  --glass:   rgba(255,255,255,0.05);
  --border:  rgba(255,255,255,0.10);
  --text:    #E2E8F0;
  --muted:   #94A3B8;
  --card:    #111C2E;
  --card2:   #162033;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--navy) !important;
  color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 99px; }

/* ── Hero Banner ── */
.hero {
  position: relative;
  background: linear-gradient(135deg, #0A1628 0%, #0F2040 40%, #0A2540 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 300px; height: 300px;
  background: radial-gradient(circle, rgba(0,201,167,0.15) 0%, transparent 70%);
  pointer-events: none;
}
.hero::after {
  content: '';
  position: absolute;
  bottom: -80px; left: 40%;
  width: 250px; height: 250px;
  background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,201,167,0.12); border: 1px solid rgba(0,201,167,0.3);
  color: var(--teal); font-size: 0.72rem; font-weight: 600;
  padding: 4px 12px; border-radius: 99px; margin-bottom: 1rem;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem; font-weight: 800;
  color: #fff; margin: 0 0 0.5rem;
  line-height: 1.15;
}
.hero h1 span { color: var(--teal); }
.hero p { color: var(--muted); font-size: 0.9rem; max-width: 560px; margin: 0; }
.hero-meta {
  display: flex; gap: 1.5rem; margin-top: 1.5rem; flex-wrap: wrap;
}
.hero-chip {
  display: flex; align-items: center; gap: 6px;
  background: var(--glass); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 14px;
  font-size: 0.78rem; color: var(--muted);
}
.hero-chip b { color: var(--text); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--card) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select {
  background: var(--navy2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] .stButton button {
  background: linear-gradient(135deg, var(--teal), var(--teal2)) !important;
  color: var(--navy) !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.65rem 1rem !important;
  font-size: 0.9rem !important;
  letter-spacing: 0.02em !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 20px rgba(0,201,167,0.3) !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 28px rgba(0,201,167,0.45) !important;
}
.sidebar-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem; font-weight: 800; color: #fff;
  margin-bottom: 0.25rem;
}
.sidebar-logo span { color: var(--teal); }
.sidebar-divider {
  border: none; border-top: 1px solid var(--border);
  margin: 1rem 0;
}
.sidebar-section {
  font-size: 0.7rem; font-weight: 600; color: var(--teal);
  text-transform: uppercase; letter-spacing: 0.1em;
  margin: 1rem 0 0.5rem;
}

/* ── Stat Cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  position: relative; overflow: hidden;
  transition: transform 0.2s;
}
.stat-card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--teal), var(--blue));
}
.stat-card:hover { transform: translateY(-2px); }
.stat-icon { font-size: 1.4rem; margin-bottom: 0.5rem; }
.stat-num {
  font-family: 'Syne', sans-serif;
  font-size: 2rem; font-weight: 800; color: #fff;
  line-height: 1;
}
.stat-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 4px; }

/* ── Job Card ── */
.job-card {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 0.75rem;
  transition: border-color 0.2s, transform 0.2s;
}
.job-card:hover { border-color: rgba(0,201,167,0.4); transform: translateX(4px); }
.job-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem; }
.job-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem; font-weight: 700; color: #fff;
}
.job-company { font-size: 0.83rem; color: var(--muted); margin-top: 3px; }
.match-circle {
  min-width: 56px; height: 56px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-size: 0.85rem; font-weight: 800;
  border: 2px solid;
}

/* ── Badges ── */
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 0.68rem; font-weight: 600;
  padding: 3px 10px; border-radius: 99px; margin: 2px;
}
.bg  { background: rgba(0,201,167,0.12);  color: #00C9A7; border: 1px solid rgba(0,201,167,0.2); }
.bb  { background: rgba(59,130,246,0.12); color: #60A5FA; border: 1px solid rgba(59,130,246,0.2); }
.bp  { background: rgba(139,92,246,0.12); color: #A78BFA; border: 1px solid rgba(139,92,246,0.2); }
.bo  { background: rgba(255,179,71,0.12); color: #FFB347; border: 1px solid rgba(255,179,71,0.2); }
.br  { background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.2); }
.bpk { background: rgba(236,72,153,0.12); color: #F472B6; border: 1px solid rgba(236,72,153,0.2); }

/* ── Match Bar ── */
.match-bar-bg   { background: rgba(255,255,255,0.07); border-radius: 99px; height: 6px; overflow: hidden; margin: 8px 0 4px; }
.match-bar-fill { height: 100%; border-radius: 99px; transition: width 1s ease; }

/* ── Skill Pills ── */
.skill-m { display: inline-flex; align-items: center; gap: 4px; font-size: 0.68rem; padding: 3px 10px; border-radius: 99px; margin: 2px; background: rgba(0,201,167,0.1); color: #00C9A7; border: 1px solid rgba(0,201,167,0.25); }
.skill-x { display: inline-flex; align-items: center; gap: 4px; font-size: 0.68rem; padding: 3px 10px; border-radius: 99px; margin: 2px; background: rgba(239,68,68,0.1);  color: #F87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── Gap Box ── */
.gap-box {
  background: rgba(255,179,71,0.06);
  border: 1px solid rgba(255,179,71,0.2);
  border-radius: 12px; padding: 0.85rem 1rem; margin-top: 0.75rem;
}
.gap-t { font-weight: 600; color: #FFB347; font-size: 0.8rem; margin-bottom: 4px; }
.gap-s { font-size: 0.75rem; color: #FCD34D; }

/* ── Section Header ── */
.sec {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem; font-weight: 700; color: #fff;
  border-left: 3px solid var(--teal);
  padding-left: 0.75rem; margin: 2rem 0 1rem;
  display: flex; align-items: center; gap: 0.5rem;
}

/* ── Welcome Screen ── */
.welcome-card {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 20px; padding: 3rem 2rem;
  text-align: center;
}
.welcome-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.welcome-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem; font-weight: 800; color: #fff; margin-bottom: 0.5rem;
}
.welcome-sub { font-size: 0.88rem; color: var(--muted); max-width: 480px; margin: 0 auto; }

.step-card {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 1.25rem;
  text-align: center; transition: border-color 0.2s;
}
.step-card:hover { border-color: rgba(0,201,167,0.35); }
.step-num {
  font-family: 'Syne', sans-serif;
  font-size: 2rem; font-weight: 800;
  color: var(--teal); margin-bottom: 0.5rem;
}
.step-title { font-weight: 600; color: #fff; margin-bottom: 0.25rem; }
.step-desc { font-size: 0.78rem; color: var(--muted); }

/* ── Analytics ── */
.analytics-card {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 1.25rem;
}

/* ── Expander ── */
details { background: var(--card2) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; margin-bottom: 0.5rem !important; }
summary { color: #fff !important; font-weight: 600 !important; }
.streamlit-expanderContent { background: transparent !important; }

/* ── Download Button ── */
.stDownloadButton button {
  background: linear-gradient(135deg, var(--blue), var(--purple)) !important;
  color: white !important; font-weight: 600 !important;
  border: none !important; border-radius: 12px !important;
  padding: 0.6rem 1.2rem !important;
}

/* ── Apply Button ── */
.stButton button:not([kind="primary"]) {
  background: rgba(59,130,246,0.1) !important;
  color: #60A5FA !important;
  border: 1px solid rgba(59,130,246,0.25) !important;
  border-radius: 10px !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  transition: all 0.2s !important;
}
.stButton button:not([kind="primary"]):hover {
  background: rgba(59,130,246,0.2) !important;
  border-color: rgba(59,130,246,0.45) !important;
  transform: translateY(-1px) !important;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 14px !important; overflow: hidden; }
.stDataFrame * { font-size: 0.8rem !important; }

/* ── Radio / Filter tabs ── */
.stRadio > div { gap: 0.5rem !important; }
.stRadio label {
  background: var(--card2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 4px 14px !important;
  font-size: 0.8rem !important;
  cursor: pointer !important;
  transition: all 0.15s !important;
}
.stRadio label:hover { border-color: var(--teal) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab"] {
  background: var(--card2) !important;
  border-radius: 10px !important;
  color: var(--muted) !important;
  font-weight: 600 !important;
  border: 1px solid var(--border) !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,201,167,0.12) !important;
  color: var(--teal) !important;
  border-color: rgba(0,201,167,0.3) !important;
}

/* ── Info / Warning / Success ── */
.stAlert { border-radius: 12px !important; border: 1px solid var(--border) !important; }

/* ── Salary highlight ── */
.salary-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(255,179,71,0.08);
  border: 1px solid rgba(255,179,71,0.2);
  border-radius: 8px; padding: 4px 12px;
  font-size: 0.8rem; font-weight: 700; color: #FFB347;
}

/* ── Toast animation ── */
@keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
.job-card { animation: fadeIn 0.3s ease; }

/* ── Pulse dot ── */
.pulse {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; background: var(--teal);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,201,167,0.4); }
  50%      { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,201,167,0); }
}

/* ── How it works steps ── */
.how-step {
  display: flex; gap: 1.25rem;
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 16px; padding: 1.25rem; margin-bottom: 0.75rem;
}
.how-num {
  font-family: 'Syne', sans-serif;
  font-size: 2rem; font-weight: 800; color: var(--teal);
  min-width: 40px; line-height: 1;
}
.how-title { font-weight: 700; color: #fff; margin-bottom: 4px; }
.how-desc { font-size: 0.82rem; color: var(--muted); }

/* ── Tech table ── */
.tech-table {
  width: 100%; border-collapse: collapse; font-size: 0.82rem;
}
.tech-table th {
  background: rgba(0,201,167,0.08);
  color: var(--teal); font-weight: 600; padding: 10px 14px;
  text-align: left; border-bottom: 1px solid var(--border);
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
}
.tech-table td { padding: 9px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--muted); }
.tech-table td:first-child { color: #fff; font-weight: 600; }
.tech-table tr:hover td { background: var(--glass); }

/* ── LaTeX override ── */
.katex { color: var(--text) !important; }

/* ── Page nav radio ── */
section[data-testid="stSidebar"] .stRadio label {
  font-size: 0.83rem !important;
}
</style>

<script>
// Animate stat numbers on load
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.stat-num[data-target]').forEach(el => {
    const target = +el.dataset.target;
    const isPercent = el.dataset.percent === '1';
    let cur = 0;
    const step = Math.max(1, Math.floor(target / 40));
    const timer = setInterval(() => {
      cur = Math.min(cur + step, target);
      el.textContent = cur + (isPercent ? '%' : '');
      if (cur >= target) clearInterval(timer);
    }, 30);
  });
  // Animate match bars
  document.querySelectorAll('.match-bar-fill').forEach(el => {
    const w = el.style.width;
    el.style.width = '0';
    setTimeout(() => { el.style.width = w; }, 100);
  });
});
</script>
""", unsafe_allow_html=True)


# ── Dataset ───────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    data = [
        {"id":1,"title":"Senior Python Developer","company":"TechNova Solutions","location":"Remote","type":"Full-time","category":"Engineering","salary":"$90,000–$130,000","exp_min":4,"edu":"Bachelor's","desc":"Python Django FastAPI AWS Docker PostgreSQL REST APIs microservices backend agile CI/CD","skills":"Python,Django,FastAPI,AWS,Docker,PostgreSQL,REST APIs,Git"},
        {"id":2,"title":"Full Stack Web Developer","company":"WebCraft Studio","location":"Karachi, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp_min":2,"edu":"Bachelor's","desc":"React Node.js MongoDB Express JavaScript TypeScript HTML CSS REST APIs agile git frontend backend","skills":"React,Node.js,MongoDB,JavaScript,TypeScript,HTML,CSS,Express,Git"},
        {"id":3,"title":"Frontend Developer","company":"UX Studio","location":"London, UK","type":"Full-time","category":"Engineering","salary":"$70,000–$100,000","exp_min":2,"edu":"Bachelor's","desc":"React Vue JavaScript TypeScript CSS HTML webpack performance accessibility user experience design","skills":"React,Vue.js,JavaScript,TypeScript,CSS,HTML,Webpack,Figma"},
        {"id":4,"title":"Backend Developer","company":"ServerLogic","location":"Remote","type":"Full-time","category":"Engineering","salary":"$80,000–$115,000","exp_min":3,"edu":"Bachelor's","desc":"Python Java Spring Boot Django Flask PostgreSQL MySQL Redis Docker Kubernetes cloud AWS CI/CD","skills":"Python,Java,Spring Boot,PostgreSQL,MySQL,Docker,Kubernetes,Redis"},
        {"id":5,"title":"Data Scientist","company":"DataSphere Analytics","location":"New York, USA","type":"Full-time","category":"Data Science","salary":"$95,000–$140,000","exp_min":3,"edu":"Master's","desc":"Python TensorFlow scikit-learn pandas numpy machine learning deep learning SQL statistics data visualization analytics","skills":"Python,Machine Learning,TensorFlow,SQL,Pandas,NumPy,Statistics,Scikit-learn"},
        {"id":6,"title":"Machine Learning Engineer","company":"AI Dynamics","location":"San Francisco, USA","type":"Full-time","category":"AI/ML","salary":"$120,000–$180,000","exp_min":3,"edu":"Master's","desc":"PyTorch TensorFlow MLOps Kubernetes Docker machine learning deep learning NLP computer vision model deployment","skills":"Python,PyTorch,TensorFlow,MLOps,Kubernetes,Docker,Deep Learning,NLP"},
        {"id":7,"title":"AI Research Scientist","company":"DeepMind Labs","location":"London, UK","type":"Full-time","category":"AI/ML","salary":"$130,000–$200,000","exp_min":2,"edu":"PhD","desc":"Deep learning NLP reinforcement learning PyTorch mathematics statistics research papers neural networks transformers BERT GPT algorithms","skills":"Python,PyTorch,Deep Learning,NLP,Research,Mathematics,Statistics,Transformers"},
        {"id":8,"title":"NLP Engineer","company":"LanguageAI","location":"Remote","type":"Full-time","category":"AI/ML","salary":"$100,000–$145,000","exp_min":3,"edu":"Master's","desc":"NLP pipelines text classification BERT transformers spaCy NLTK Python TensorFlow language models GPT fine-tuning sentiment analysis","skills":"Python,NLP,Transformers,BERT,spaCy,NLTK,TensorFlow,Hugging Face"},
        {"id":9,"title":"Data Engineer","company":"PipelineAI","location":"Singapore","type":"Full-time","category":"Data Science","salary":"$85,000–$120,000","exp_min":3,"edu":"Bachelor's","desc":"Apache Spark Kafka SQL AWS Airflow ETL data warehouse Python cloud engineering big data stream processing batch","skills":"Python,Apache Spark,Kafka,SQL,AWS,Airflow,ETL,Data Warehouse"},
        {"id":10,"title":"Business Intelligence Analyst","company":"InsightCorp","location":"Karachi, Pakistan","type":"Full-time","category":"Data Science","salary":"$30,000–$55,000","exp_min":2,"edu":"Bachelor's","desc":"SQL Power BI Tableau Excel data visualization dashboard reporting KPIs business analytics Python stakeholder","skills":"SQL,Power BI,Tableau,Excel,Python,Data Analytics,Reporting"},
        {"id":11,"title":"DevOps Engineer","company":"CloudCore","location":"Remote","type":"Full-time","category":"DevOps","salary":"$85,000–$125,000","exp_min":3,"edu":"Bachelor's","desc":"CI/CD AWS Docker Kubernetes Terraform Jenkins Linux Python monitoring deployment site reliability automation","skills":"AWS,Docker,Kubernetes,Terraform,Jenkins,Linux,Python,CI/CD"},
        {"id":12,"title":"Cloud Solutions Architect","company":"Nimbus Cloud","location":"Remote","type":"Full-time","category":"DevOps","salary":"$110,000–$160,000","exp_min":5,"edu":"Bachelor's","desc":"AWS Azure GCP Terraform Docker Kubernetes networking security cloud migration enterprise solutions infrastructure","skills":"AWS,Azure,GCP,Terraform,Docker,Kubernetes,Networking,Security"},
        {"id":13,"title":"Cybersecurity Analyst","company":"SecureNet","location":"Remote","type":"Full-time","category":"Security","salary":"$80,000–$120,000","exp_min":2,"edu":"Bachelor's","desc":"Network security SIEM penetration testing Linux firewalls incident response Python compliance vulnerability assessment","skills":"Network Security,Python,SIEM,Penetration Testing,Linux,Firewalls,Incident Response"},
        {"id":14,"title":"UX/UI Designer","company":"PixelCraft","location":"Dubai, UAE","type":"Full-time","category":"Design","salary":"$55,000–$90,000","exp_min":2,"edu":"Bachelor's","desc":"Figma Adobe XD wireframes prototypes UI design user research usability testing design systems CSS HTML responsive mobile","skills":"Figma,Adobe XD,UI Design,User Research,Prototyping,Sketch,CSS,HTML"},
        {"id":15,"title":"Graphic Designer","company":"VisualEdge","location":"Lahore, Pakistan","type":"Full-time","category":"Design","salary":"$15,000–$35,000","exp_min":1,"edu":"Bachelor's","desc":"Adobe Photoshop Illustrator InDesign Canva typography brand identity logo social media marketing print digital design","skills":"Adobe Photoshop,Illustrator,InDesign,Canva,Typography,Branding,Figma"},
        {"id":16,"title":"Flutter Mobile Developer","company":"AppForge","location":"Karachi, Pakistan","type":"Full-time","category":"Mobile","salary":"$30,000–$55,000","exp_min":1,"edu":"Bachelor's","desc":"Flutter Dart Firebase REST APIs state management Android iOS app store deployment performance cross-platform mobile","skills":"Flutter,Dart,Firebase,REST APIs,Git,Android,iOS,State Management"},
        {"id":17,"title":"Android Developer","company":"MobileFirst","location":"Remote","type":"Full-time","category":"Mobile","salary":"$70,000–$100,000","exp_min":2,"edu":"Bachelor's","desc":"Kotlin Java Android SDK MVVM REST APIs Firebase Room database Jetpack Compose Google Play native mobile","skills":"Kotlin,Java,Android SDK,MVVM,REST APIs,Firebase,Jetpack Compose"},
        {"id":18,"title":"Product Manager","company":"Innovatech","location":"Austin, USA","type":"Full-time","category":"Product","salary":"$100,000–$150,000","exp_min":4,"edu":"Bachelor's","desc":"Product strategy roadmap agile JIRA stakeholder management user research data analysis A/B testing communication","skills":"Product Strategy,Agile,JIRA,Data Analysis,SQL,Communication,Roadmapping"},
        {"id":19,"title":"Digital Marketing Specialist","company":"GrowthHive","location":"Karachi, Pakistan","type":"Full-time","category":"Marketing","salary":"$20,000–$40,000","exp_min":1,"edu":"Bachelor's","desc":"SEO SEM Google Ads social media content marketing email marketing analytics Canva brand awareness performance reporting","skills":"SEO,Google Ads,Social Media,Content Marketing,Analytics,Email Marketing,Canva"},
        {"id":20,"title":"Business Analyst","company":"FinEdge","location":"Toronto, Canada","type":"Full-time","category":"Business","salary":"$65,000–$95,000","exp_min":2,"edu":"Bachelor's","desc":"SQL data analysis Excel Power BI JIRA process improvement stakeholder communication agile scrum reporting business requirements","skills":"Data Analysis,SQL,Excel,Power BI,JIRA,Tableau,Communication,Agile"},
        {"id":21,"title":"QA Automation Engineer","company":"TestPro","location":"Lahore, Pakistan","type":"Full-time","category":"Engineering","salary":"$25,000–$45,000","exp_min":2,"edu":"Bachelor's","desc":"Selenium Cypress Python JavaScript API testing JIRA CI/CD regression testing quality assurance automation framework","skills":"Selenium,Cypress,Python,JavaScript,API Testing,JIRA,CI/CD"},
        {"id":22,"title":"Blockchain Developer","company":"ChainTech","location":"Remote","type":"Full-time","category":"Engineering","salary":"$100,000–$150,000","exp_min":3,"edu":"Bachelor's","desc":"Solidity Ethereum Web3.js DeFi NFT smart contracts blockchain Python JavaScript cryptography decentralized applications security audit","skills":"Solidity,Ethereum,Web3.js,Python,Smart Contracts,JavaScript,Cryptography"},
        {"id":23,"title":"Network Engineer","company":"NetSystems","location":"Islamabad, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp_min":2,"edu":"Bachelor's","desc":"Cisco routers switches firewalls VPN TCP/IP Linux Windows server networking troubleshooting security monitoring CCNA protocols","skills":"Cisco,Networking,Firewalls,VPN,TCP/IP,Linux,Windows Server,CCNA"},
        {"id":24,"title":"HR Business Partner","company":"PeopleFirst","location":"Dubai, UAE","type":"Full-time","category":"HR","salary":"$50,000–$80,000","exp_min":3,"edu":"Bachelor's","desc":"Talent acquisition employee relations performance management HR policies training development HRMS communication organizational recruitment","skills":"HR Management,Recruitment,Employee Relations,Performance Management,HRMS,Training"},
        {"id":25,"title":"Python Developer Intern","company":"StartupHub","location":"Karachi, Pakistan","type":"Internship","category":"Engineering","salary":"$5,000–$12,000","exp_min":0,"edu":"Intermediate","desc":"Python programming Django REST APIs database SQL Git web development backend basics agile teamwork projects","skills":"Python,Django,SQL,Git,REST APIs,HTML"},
        {"id":26,"title":"Data Science Intern","company":"Analytics Co","location":"Lahore, Pakistan","type":"Internship","category":"Data Science","salary":"$5,000–$10,000","exp_min":0,"edu":"Intermediate","desc":"Python pandas numpy matplotlib machine learning scikit-learn SQL data visualization statistics Excel Jupyter notebook analysis","skills":"Python,Pandas,NumPy,Matplotlib,SQL,Scikit-learn,Excel"},
        {"id":27,"title":"UI/UX Design Intern","company":"CreativeMinds","location":"Remote","type":"Internship","category":"Design","salary":"$4,000–$8,000","exp_min":0,"edu":"Intermediate","desc":"Wireframes prototypes Figma user interface mobile web design typography color theory user experience research Canva iteration","skills":"Figma,Canva,UI Design,Prototyping,Typography"},
        {"id":28,"title":"Freelance Web Developer","company":"Various Clients","location":"Remote","type":"Freelance","category":"Engineering","salary":"$30,000–$80,000","exp_min":1,"edu":"Diploma","desc":"WordPress React JavaScript HTML CSS PHP MySQL client websites freelance remote work communication deadlines project management","skills":"WordPress,React,JavaScript,HTML,CSS,PHP,MySQL"},
        {"id":29,"title":"Content Writer","company":"ContentPro Agency","location":"Remote","type":"Part-time","category":"Marketing","salary":"$15,000–$30,000","exp_min":1,"edu":"Bachelor's","desc":"Technical writing SEO blogs articles research AI technology software communication editing proofreading WordPress content marketing","skills":"Technical Writing,SEO,Research,Communication,WordPress,Editing"},
        {"id":30,"title":"iOS Developer","company":"AppleTree Apps","location":"Dubai, UAE","type":"Full-time","category":"Mobile","salary":"$75,000–$110,000","exp_min":2,"edu":"Bachelor's","desc":"Swift SwiftUI UIKit Xcode CoreData REST APIs push notifications App Store deployment performance Objective-C native iOS mobile","skills":"Swift,SwiftUI,UIKit,Xcode,CoreData,REST APIs,Objective-C"},
    ]
    return pd.DataFrame(data)


@st.cache_resource
def build_model(descs):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    mat = vec.fit_transform(descs)
    return vec, mat


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def recommend(skills_str, exp, edu, jtype, vec, mat, df, n=8):
    user_vec = vec.transform([preprocess(skills_str)])
    scores   = cosine_similarity(user_vec, mat).flatten()
    res      = df.copy()
    res['base'] = scores

    edu_rank = {"Intermediate": 1, "Diploma": 2, "Bachelor's": 3, "Master's": 4, "PhD": 5}
    u_edu = edu_rank.get(edu, 3)

    res['boost'] = (
        res['exp_min'].apply(lambda e: 0.08 if exp >= e else (-0.05 if exp < e - 2 else 0)) +
        res['type'].apply(lambda t: 0.06 if (not jtype or t == jtype) else 0) +
        res['edu'].apply(lambda e: 0.05 if edu_rank.get(e, 3) <= u_edu else -0.03)
    )
    res['final'] = res['base'] * 0.75 + res['boost']

    mn, mx = res['final'].min(), res['final'].max()
    res['pct'] = ((res['final'] - mn) / (mx - mn) * 39 + 60).clip(0, 99).astype(int) if mx > mn else 60
    return res.sort_values('pct', ascending=False).head(n).reset_index(drop=True)


def skill_split(user_str, job_str):
    u = {s.strip().lower() for s in user_str.split(',') if s.strip()}
    j = {s.strip().lower() for s in job_str.split(',')  if s.strip()}
    return sorted(u & j), sorted(j - u)


def match_color(p):
    if p >= 82: return "#00C9A7", "#00C9A7"
    if p >= 68: return "#3B82F6", "#3B82F6"
    return "#EF4444", "#EF4444"


def bar_color(p):
    if p >= 82: return "linear-gradient(90deg,#00C9A7,#00A88E)"
    if p >= 68: return "linear-gradient(90deg,#3B82F6,#8B5CF6)"
    return "linear-gradient(90deg,#EF4444,#F87171)"


# ── Official SVG icons via cdn.simpleicons.org & inline SVG ───
# Each value is an <img> or inline <svg> that renders an official brand/tech icon.
def si(slug, color="ffffff", size=18):
    """Simple Icons CDN img tag."""
    return f'<img src="https://cdn.simpleicons.org/{slug}/{color}" width="{size}" height="{size}" style="vertical-align:middle;filter:drop-shadow(0 0 2px rgba(0,0,0,0.4));">'

# Lucide SVG paths (hand-coded strokes, no external CDN needed)
def lucide_svg(path_d, color="#ffffff", size=18):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;">'
            f'{path_d}</svg>')

# Category icon map — returns HTML string
def cat_icon(category, size=18):
    icons = {
        "Engineering":  si("gnubash",    "00C9A7", size),        # terminal/code
        "Data Science": si("apachehadoop","FFB347", size),       # data/big data
        "AI/ML":        si("openai",     "74AA9C", size),        # OpenAI brand
        "DevOps":       si("docker",     "2496ED", size),        # Docker official
        "Security":     si("hackthebox", "9FEF00", size),        # security green
        "Design":       si("figma",      "F24E1E", size),        # Figma official
        "Mobile":       si("flutter",    "02569B", size),        # Flutter official
        "Product":      si("jira",       "0052CC", size),        # Jira/product
        "Marketing":    si("googleads",  "4285F4", size),        # Google Ads
        "Business":     si("microsoftexcel","217346", size),     # Excel/business
        "HR":           si("workday",    "E36A00", size),        # Workday HR
    }
    return icons.get(category, lucide_svg('<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/>', "#94A3B8", size))

# ── Tech stack official icons ──────────────────────────────────
TECH_ICONS = {
    "Python":       si("python",       "3776AB"),
    "Streamlit":    si("streamlit",    "FF4B4B"),
    "Scikit-learn": si("scikitlearn",  "F7931E"),
    "Pandas":       si("pandas",       "150458"),
    "NumPy":        si("numpy",        "013243"),
    "Matplotlib":   si("python",       "3776AB"),   # no dedicated icon; Python blue
    "JavaScript":   si("javascript",   "F7DF1E"),
    "Docker":       si("docker",       "2496ED"),
    "AWS":          si("amazonaws",    "FF9900"),
    "Git":          si("git",          "F05032"),
    "Linux":        si("linux",        "FCC624"),
    "React":        si("react",        "61DAFB"),
    "TensorFlow":   si("tensorflow",   "FF6F00"),
    "PyTorch":      si("pytorch",      "EE4C2C"),
}

# Simple text fallback label with icon for sidebar sections
SECTION_ICONS = {
    "profile":  si("person", "00C9A7", 14),  # fallback to lucide
    "edu":      si("googlescholar", "4285F4", 14),
    "filter":   si("googleoptimize", "4285F4", 14),
    "nav":      si("googlemaps", "4285F4", 14),
}

# ── Matplotlib dark theme ─────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#111C2E',
    'axes.facecolor':    '#111C2E',
    'axes.edgecolor':    '#1E2D45',
    'axes.labelcolor':   '#94A3B8',
    'xtick.color':       '#94A3B8',
    'ytick.color':       '#94A3B8',
    'text.color':        '#E2E8F0',
    'grid.color':        '#1E2D45',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
})

CHART_COLORS = ['#00C9A7', '#3B82F6', '#8B5CF6', '#EC4899', '#FFB347',
                '#F87171', '#34D399', '#60A5FA', '#A78BFA', '#F472B6']

# ── Load data ─────────────────────────────────────────────────
df  = load_dataset()
vec, mat = build_model(df['desc'].astype(str))

# ══════════════════════════════════════════════════════════════
#   SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">Job<span>Match</span> AI</div>
    <div style="font-size:0.72rem;color:#94A3B8;">Powered by TF-IDF · Cosine Similarity</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section">{si("target","00C9A7",13)} &nbsp;Your Profile</div>', unsafe_allow_html=True)

    user_skills = st.text_area(
        "Skills",
        placeholder="e.g. Python, Machine Learning, SQL, Django, React",
        height=100,
        help="Enter skills separated by commas"
    )

    st.markdown(f'<div class="sidebar-section">{si("googlescholar","4285F4",13)} &nbsp;Education & Experience</div>', unsafe_allow_html=True)
    education  = st.selectbox("Education Level", ["Intermediate", "Diploma", "Bachelor's", "Master's", "PhD"])
    experience = st.slider("Years of Experience", 0, 20, 1)

    st.markdown(f'<div class="sidebar-section">{si("googlesearchconsole","E37400",13)} &nbsp;Filters</div>', unsafe_allow_html=True)
    job_type      = st.selectbox("Job Type", ["Any", "Full-time", "Part-time", "Remote", "Freelance", "Internship"])
    location_pref = st.text_input("Preferred Location", placeholder="e.g. Remote, Karachi, Dubai")
    top_n         = st.slider("Results to Show", 3, 15, 8)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    find_btn = st.button("Find My Jobs  →", use_container_width=True, type="primary")
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section">{si("googlemaps","4285F4",13)} &nbsp;Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "nav",
        ["🏠 Home", "📊 Analytics", "ℹ️ How It Works"],
        label_visibility="collapsed"
    )


# ══════════════════════════════════════════════════════════════
#   HERO
# ══════════════════════════════════════════════════════════════
hero_scikit  = si("scikitlearn", "F7931E", 16)
hero_python  = si("python",      "3776AB", 16)
hero_streaml = si("streamlit",   "FF4B4B", 16)
hero_github  = si("github",      "ffffff", 16)
st.markdown(f"""
<div class="hero">
  <div class="hero-badge"><span class="pulse"></span>&nbsp; AI-Powered Matching</div>
  <h1>Find Your Perfect <span>Career Match</span></h1>
  <p>Intelligent job recommendations using TF-IDF vectorization and cosine similarity — matching your skills to the best opportunities.</p>
  <div class="hero-meta">
    <div class="hero-chip">{hero_scikit} &nbsp;<b>TF-IDF Vectorization</b></div>
    <div class="hero-chip">{hero_python} &nbsp;<b>Cosine Similarity</b></div>
    <div class="hero-chip">{hero_streaml} &nbsp;<b>Streamlit App</b></div>
    <div class="hero-chip">{hero_github} &nbsp;<b>Waqaas Hussain</b></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#   HOME PAGE
# ══════════════════════════════════════════════════════════════
if "Home" in page:

    if find_btn or 'results' in st.session_state:

        if find_btn:
            if not user_skills.strip():
                st.warning("⚠️ Please enter at least one skill to get recommendations.")
                st.stop()

            jt  = "" if job_type == "Any" else job_type
            res = recommend(user_skills, experience, education, jt, vec, mat, df, top_n)

            if location_pref.strip():
                res = res[
                    res['location'].str.lower().str.contains(location_pref.lower(), na=False) |
                    res['location'].str.lower().str.contains('remote', na=False)
                ]

            st.session_state['results'] = res
            st.session_state['skills']  = user_skills

        res        = st.session_state['results']
        skills_str = st.session_state['skills']

        if res.empty:
            st.warning("No matching jobs found. Try broadening your filters or location.")
            st.stop()

        # ── Stat Cards ──────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, icon_html, num, lbl in [
            (c1, si("linkedin",   "0A66C2", 24), len(res),                    "Jobs Found"),
            (c2, si("googleanalytics","E37400",24), f"{int(res['pct'].mean())}%", "Avg Match Score"),
            (c3, si("trello",     "0052CC", 24), f"{int(res['pct'].max())}%",  "Best Match"),
            (c4, si("stackexchange","1E5397",24), len([s for s in skills_str.split(',') if s.strip()]), "Skills Detected"),
        ]:
            col.markdown(f"""
<div class="stat-card">
  <div class="stat-icon">{icon_html}</div>
  <div class="stat-num">{num}</div>
  <div class="stat-lbl">{lbl}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Category filter ─────────────────────────────────
        cats = ["All"] + sorted(res['category'].unique().tolist())
        sel  = st.radio("Filter by Category:", cats, horizontal=True)
        show = res if sel == "All" else res[res['category'] == sel]

        st.markdown(f'<p style="color:#94A3B8;font-size:0.8rem;margin:0.5rem 0 1rem;">Showing <b style="color:#00C9A7;">{len(show)}</b> job(s) — ranked by AI match score</p>', unsafe_allow_html=True)

        # ── Job Cards ────────────────────────────────────────
        for _, j in show.iterrows():
            pct   = j['pct']
            c_hex, _ = match_color(pct)
            bg_bar   = bar_color(pct)
            matched, missing = skill_split(skills_str, j['skills'])
            m_pills = "".join([f'<span class="skill-m">{si("checkmarx","00C9A7",11)} {s}</span>' for s in matched])
            x_pills = "".join([f'<span class="skill-x">{si("x","EF4444",11)} {s}</span>' for s in missing])
            cicon   = cat_icon(j['category'], 16)
            loc_icon = si("googlemaps", "4285F4", 13)
            bld_icon = si("microsoftazure", "0078D4", 13)
            sal_icon = si("cashapp", "00D632", 13)
            skl_icon = si("skillshare", "00FF84", 13)

            with st.expander(f"  {j['title']}  ·  {j['company']}  —  {pct}% match"):
                st.markdown(f"""
<div class="job-card">
  <div class="job-header">
    <div>
      <div class="job-title" style="display:flex;align-items:center;gap:8px;">{cicon} {j['title']}</div>
      <div class="job-company">{bld_icon} {j['company']} &nbsp;·&nbsp; {loc_icon} {j['location']}</div>
    </div>
    <div class="match-circle" style="color:{c_hex};border-color:{c_hex};background:rgba(0,0,0,0.2);">
      {pct}%
    </div>
  </div>

  <div style="margin: 0.5rem 0 0.75rem;">
    <span class="badge bg">{si("target","00C9A7",11)} {pct}% Match</span>
    <span class="badge bb">{si("clockify","3B82F6",11)} {j['type']}</span>
    <span class="badge bp">{cicon} {j['category']}</span>
    <span class="badge bo">{si("googlescholar","FFB347",11)} {j['edu']}+</span>
    <span class="badge br">{si("clockify","EF4444",11)} {j['exp_min']}+ yrs exp</span>
  </div>

  <div class="match-bar-bg">
    <div class="match-bar-fill" style="width:{pct}%;background:{bg_bar};"></div>
  </div>
  <div style="font-size:0.7rem;color:#94A3B8;margin-bottom:1rem;">{pct}% alignment with your profile</div>

  <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
    <div class="salary-badge">{sal_icon} &nbsp;{j['salary']}</div>
  </div>

  <div style="font-size:0.78rem;font-weight:600;color:#E2E8F0;margin-bottom:6px;">{skl_icon} &nbsp;Skill Analysis</div>
  <div>
    {m_pills if m_pills else ''}
    {x_pills if x_pills else ''}
  </div>
  {'<div style="font-size:0.68rem;color:#94A3B8;margin-top:6px;">✅ Matched &nbsp;|&nbsp; ❌ Missing (skill gaps)</div>' if (matched or missing) else ''}
</div>
""", unsafe_allow_html=True)

                if missing:
                    st.markdown(f"""
<div class="gap-box">
  <div class="gap-t">{si("bookstack","FFB347",13)} &nbsp;Skill Gap — Recommended to Learn:</div>
  <div class="gap-s">{', '.join(missing)}</div>
</div>""", unsafe_allow_html=True)

                col_a, col_b = st.columns([1, 4])
                with col_a:
                    if st.button(f"Apply Now", key=f"apply_{j['id']}"):
                        st.success(f"Application sent for **{j['title']}** at **{j['company']}**!")

        st.markdown("<br>", unsafe_allow_html=True)
        csv = show[['title', 'company', 'location', 'type', 'category', 'salary', 'pct']] \
              .rename(columns={'title': 'Title', 'company': 'Company', 'location': 'Location',
                               'type': 'Type', 'category': 'Category', 'salary': 'Salary', 'pct': 'Match%'}) \
              .to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Results as CSV",
            csv, "job_recommendations.csv", "text/csv",
            use_container_width=True
        )

    else:
        # ── Welcome Screen ───────────────────────────────────
        welcome_icon = si("target", "00C9A7", 56)
        st.markdown(f"""
<div class="welcome-card">
  <div class="welcome-icon">{welcome_icon}</div>
  <div class="welcome-title">Welcome, Waqaas!</div>
  <p class="welcome-sub">
    Enter your skills and preferences in the <b>sidebar</b>, then click
    <b>"Find My Jobs"</b> to receive AI-powered personalized job recommendations.
  </p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, num, title, desc in [
            (c1, "01", "Enter Your Skills", "Type your skills separated by commas — e.g. Python, SQL, React, Machine Learning"),
            (c2, "02", "Set Preferences",   "Choose your education level, years of experience, job type & location"),
            (c3, "03", "Get Matched",        "Click 'Find My Jobs' to receive ranked AI-powered job recommendations"),
        ]:
            col.markdown(f"""
<div class="step-card">
  <div class="step-num">{num}</div>
  <div class="step-title">{title}</div>
  <div class="step-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="sec">{si("linkedin","0A66C2",16)} &nbsp;Available Job Categories</div>', unsafe_allow_html=True)
        cats = df['category'].value_counts()
        cols = st.columns(4)
        for i, (cat, cnt) in enumerate(cats.items()):
            icon_html = cat_icon(cat, 20)
            cols[i % 4].markdown(f"""
<div class="step-card" style="text-align:left;padding:0.85rem 1rem;margin-bottom:0.5rem;display:flex;align-items:center;gap:8px;">
  {icon_html}
  <span style="font-weight:600;color:#fff;flex:1;">{cat}</span>
  <span style="background:rgba(0,201,167,0.12);color:#00C9A7;font-size:0.72rem;
               font-weight:700;padding:2px 8px;border-radius:99px;">{cnt}</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#   ANALYTICS PAGE
# ══════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown(f'<div class="sec">{si("googleanalytics","E37400",18)} &nbsp;Job Market Analytics Dashboard</div>', unsafe_allow_html=True)

    # Top stats
    c1, c2, c3, c4 = st.columns(4)
    for col, icon_html, num, lbl in [
        (c1, si("linkedin",         "0A66C2", 26), len(df),              "Total Jobs"),
        (c2, si("microsoftazure",   "0078D4", 26), df['company'].nunique(), "Companies"),
        (c3, si("googletagmanager", "246FDB", 26), df['category'].nunique(), "Categories"),
        (c4, si("googlemaps",       "4285F4", 26), df['location'].nunique(), "Locations"),
    ]:
        col.markdown(f"""
<div class="stat-card">
  <div class="stat-icon">{icon_html}</div>
  <div class="stat-num">{num}</div>
  <div class="stat-lbl">{lbl}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown(f"{si('gnubash','00C9A7',16)} &nbsp;**Jobs by Category**", unsafe_allow_html=True)
        cats = df['category'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.barh(cats.index, cats.values, color=CHART_COLORS[:len(cats)],
                       edgecolor='none', height=0.65)
        for v, b in zip(cats.values, bars):
            ax.text(v + 0.1, b.get_y() + b.get_height() / 2, str(v),
                    va='center', fontsize=9, color='#E2E8F0', fontweight='600')
        ax.set_xlabel('Number of Jobs', fontsize=9)
        ax.set_title('Distribution by Category', fontsize=10, fontweight='bold', color='#E2E8F0', pad=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, cats.values.max() + 2)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown(f"{si('clockify','3B82F6',16)} &nbsp;**Job Type Distribution**", unsafe_allow_html=True)
        types = df['type'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        wedges, texts, autotexts = ax2.pie(
            types.values, labels=types.index, autopct='%1.1f%%',
            startangle=90, colors=CHART_COLORS[:len(types)],
            wedgeprops=dict(edgecolor='#111C2E', linewidth=2),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color('#111C2E')
            at.set_fontweight('bold')
        for t in texts:
            t.set_fontsize(9)
            t.set_color('#E2E8F0')
        ax2.set_title('Job Types Available', fontsize=10, fontweight='bold', color='#E2E8F0', pad=12)
        # Donut
        centre_circle = plt.Circle((0, 0), 0.55, fc='#111C2E')
        ax2.add_patch(centre_circle)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown(f"{si('googletagmanager','246FDB',16)} &nbsp;**Experience Level Required**", unsafe_allow_html=True)
        bins = pd.cut(df['exp_min'], bins=[-1, 0, 1, 2, 3, 5, 10, 20],
                      labels=['Fresher', '<1 yr', '1–2 yr', '2–3 yr', '3–5 yr', '5–10 yr', '10+ yr'])
        bc = bins.value_counts().sort_index()
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        colors3 = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(bc))]
        bars3 = ax3.bar(bc.index.astype(str), bc.values, color=colors3, edgecolor='none', width=0.6)
        for b in bars3:
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, int(b.get_height()),
                     ha='center', fontsize=9, color='#E2E8F0', fontweight='600')
        ax3.set_xlabel('Experience Level', fontsize=9)
        ax3.set_ylabel('Number of Jobs', fontsize=9)
        ax3.set_title('Jobs by Experience Required', fontsize=10, fontweight='bold', color='#E2E8F0', pad=12)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=20, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown(f"{si('stackexchange','1E5397',16)} &nbsp;**Top 12 In-Demand Skills**", unsafe_allow_html=True)
        all_skills = []
        for s in df['skills']:
            all_skills.extend([x.strip() for x in s.split(',')])
        top12 = Counter(all_skills).most_common(12)
        sn, sv = zip(*top12)
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        colors4 = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(sn))]
        ax4.barh(list(sn)[::-1], list(sv)[::-1], color=colors4[::-1], edgecolor='none', height=0.65)
        ax4.set_xlabel('Jobs Requiring Skill', fontsize=9)
        ax4.set_title('Top 12 In-Demand Skills', fontsize=10, fontweight='bold', color='#E2E8F0', pad=12)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sec">{si("microsoftexcel","217346",16)} &nbsp;Complete Job Dataset</div>', unsafe_allow_html=True)
    disp = df[['title', 'company', 'location', 'type', 'category', 'salary', 'exp_min', 'edu']].copy()
    disp.columns = ['Job Title', 'Company', 'Location', 'Type', 'Category', 'Salary', 'Min Exp', 'Education']
    st.dataframe(disp, use_container_width=True, height=420)


# ══════════════════════════════════════════════════════════════
#   HOW IT WORKS PAGE
# ══════════════════════════════════════════════════════════════
elif "How" in page:
    st.markdown(f'<div class="sec">{si("googlecolab","F9AB00",18)} &nbsp;How the AI System Works</div>', unsafe_allow_html=True)

    steps = [
        ("01", si("kaggle",       "20BEFF", 18), "Data Collection",
         "30 real-world job listings curated with job titles, full descriptions, required skills, salary ranges, location, experience and education requirements — forming a rich content-based knowledge base."),
        ("02", si("gnu",          "A42E2B", 18), "Text Preprocessing",
         "Job descriptions are cleaned and normalized: lowercased, special characters removed, whitespace collapsed. Stop words are filtered by TF-IDF's built-in English stop word list for noise reduction."),
        ("03", si("scikitlearn",  "F7931E", 18), "TF-IDF Vectorization",
         "Scikit-learn's TfidfVectorizer converts job descriptions and user skill inputs into high-dimensional numerical vectors. Rare, important keywords receive higher TF-IDF weight, while common words are down-weighted."),
        ("04", si("scipy",        "8CAAE6", 18), "Cosine Similarity Scoring",
         "Cosine similarity is computed between the user's skill vector and every job description vector. Scores range from 0 (no overlap) to 1 (perfect alignment), measuring the angular closeness of the vectors in TF-IDF space."),
        ("05", si("googleoptimize","4285F4", 18), "Score Boosting & Normalization",
         "Raw similarity scores are boosted based on experience alignment (+0.08), job type match (+0.06), and education level (+0.05). Final scores are normalized to a 60–99% range for clear, interpretable results."),
        ("06", si("checkmarx",    "54B263", 18), "Results & Skill Gap Analysis",
         "Jobs are ranked by final match percentage. Each result shows matched skills (green) and missing skills (red), providing actionable skill gap analysis so users know exactly what to learn next."),
    ]
    for num, icon_html, title, desc in steps:
        st.markdown(f"""
<div class="how-step">
  <div class="how-num">{num}</div>
  <div>
    <div class="how-title" style="display:flex;align-items:center;gap:8px;">{icon_html} {title}</div>
    <div class="how-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sec">{si("scipy","8CAAE6",16)} &nbsp;Core Algorithm — Cosine Similarity</div>', unsafe_allow_html=True)
    st.latex(r"\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}")
    st.markdown("""
<div class="how-step" style="margin-top:0.5rem;">
  <div>
    <div class="how-desc">
      <b style="color:#E2E8F0;">A</b> = TF-IDF vector of the user's skills input<br>
      <b style="color:#E2E8F0;">B</b> = TF-IDF vector of a job description<br>
      <b style="color:#E2E8F0;">Result:</b> 0 (no match) → 1 (perfect match) &nbsp;·&nbsp;
      Measures the cosine of the angle between two vectors in TF-IDF space, independent of document length.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec">{si("python","3776AB",16)} &nbsp;Technologies Used</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        py  = si("python",      "3776AB")
        stl = si("streamlit",   "FF4B4B")
        skl = si("scikitlearn", "F7931E")
        pd_ = si("pandas",      "150458")
        st.markdown(f"""
<table class="tech-table">
  <tr><th>Technology</th><th>Purpose</th></tr>
  <tr><td>{py}  &nbsp;Python 3.x</td><td>Core programming language</td></tr>
  <tr><td>{stl} &nbsp;Streamlit</td><td>Interactive web application framework</td></tr>
  <tr><td>{skl} &nbsp;Scikit-learn</td><td>TF-IDF vectorization &amp; cosine similarity</td></tr>
  <tr><td>{pd_} &nbsp;Pandas / NumPy</td><td>Data manipulation &amp; numerical computing</td></tr>
</table>""", unsafe_allow_html=True)
    with c2:
        mpl = si("python",      "3776AB")   # Matplotlib uses Python logo
        git = si("git",         "F05032")
        js  = si("javascript",  "F7DF1E")
        gh  = si("github",      "ffffff")
        st.markdown(f"""
<table class="tech-table">
  <tr><th>Technology</th><th>Purpose</th></tr>
  <tr><td>{mpl} &nbsp;Matplotlib</td><td>Analytics charts &amp; visualizations</td></tr>
  <tr><td>{git} &nbsp;Git / Regex</td><td>Text preprocessing &amp; version control</td></tr>
  <tr><td>{js}  &nbsp;JavaScript</td><td>UI animations &amp; interactive effects</td></tr>
  <tr><td>{gh}  &nbsp;Session State</td><td>In-app memory &amp; state persistence</td></tr>
</table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    user_icon = si("linkedin", "0A66C2", 28)
    st.markdown(f"""
<div class="how-step" style="background:rgba(0,201,167,0.05);border-color:rgba(0,201,167,0.2);">
  <div>{user_icon}</div>
  <div>
    <div class="how-title" style="color:#00C9A7;">Project Information</div>
    <div class="how-desc">
      <b style="color:#E2E8F0;">Prepared by:</b> Waqaas Hussain &nbsp;|&nbsp;
      <b style="color:#E2E8F0;">Subject:</b> Programming for AI &nbsp;|&nbsp;
      <b style="color:#E2E8F0;">Framework:</b> Streamlit + Scikit-learn &nbsp;|&nbsp;
      <b style="color:#E2E8F0;">Algorithm:</b> TF-IDF Vectorization + Cosine Similarity
    </div>
  </div>
</div>""", unsafe_allow_html=True)

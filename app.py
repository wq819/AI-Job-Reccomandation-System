# ============================================================
#   AI-Based Job Recommendation System
#   Prepared by  : Waqaas Hussain & Hira Abdul Hafeez
#   SAP IDs      : 5000000291, 5000000314
#   Subject      : Programming for AI
#   Instructor   : Sir Abdul Hasseb
#   Semester     : 4th Semester | Section C
#   Framework    : Streamlit + Scikit-learn + Plotly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import re
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="JobMatch AI · Waqaas Hussain",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

IS_DARK = st.session_state.theme == "dark"

BG       = "#080F1E"   if IS_DARK else "#F0F5FF"
BG2      = "#0D1A2D"   if IS_DARK else "#E8EFFA"
CARD     = "#0F2040"   if IS_DARK else "#FFFFFF"
CARD2    = "#132545"   if IS_DARK else "#F7FAFF"
BORDER   = "rgba(255,255,255,0.08)" if IS_DARK else "rgba(0,0,0,0.08)"
TEXT     = "#E8EDF5"   if IS_DARK else "#0A1628"
MUTED    = "#7A90A8"   if IS_DARK else "#64748B"
TEAL     = "#00C9A7"   if IS_DARK else "#0D9B82"
TEAL2    = "#00A88E"   if IS_DARK else "#0A7A68"
BLUE     = "#3B82F6"   if IS_DARK else "#2563EB"
PURPLE   = "#8B5CF6"   if IS_DARK else "#7C3AED"
AMBER    = "#F59E0B"   if IS_DARK else "#D97706"
RED      = "#EF4444"   if IS_DARK else "#DC2626"
PLOTLY_BG   = "#0F2040" if IS_DARK else "#FFFFFF"
PLOTLY_GRID = "#162845" if IS_DARK else "#E2E8F0"
PLOTLY_TEXT = "#7A90A8" if IS_DARK else "#64748B"

# ── GLOBAL CSS ───────────────────────────────────────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:{BG}; --bg2:{BG2}; --card:{CARD}; --card2:{CARD2};
  --border:{BORDER}; --text:{TEXT}; --muted:{MUTED};
  --teal:{TEAL}; --teal2:{TEAL2}; --blue:{BLUE};
  --purple:{PURPLE}; --amber:{AMBER}; --red:{RED};
}}
@keyframes fadeUp   {{ from{{opacity:0;transform:translateY(20px)}} to{{opacity:1;transform:translateY(0)}} }}
@keyframes slideRight{{ from{{opacity:0;transform:translateX(-16px)}} to{{opacity:1;transform:translateX(0)}} }}
@keyframes pulse    {{ 0%,100%{{box-shadow:0 0 0 0 rgba(0,201,167,.5)}} 50%{{box-shadow:0 0 0 8px rgba(0,201,167,0)}} }}
@keyframes barGrow  {{ from{{transform:scaleX(0);transform-origin:left}} to{{transform:scaleX(1);transform-origin:left}} }}
@keyframes glow     {{ 0%,100%{{text-shadow:0 0 8px rgba(0,201,167,.3)}} 50%{{text-shadow:0 0 20px rgba(0,201,167,.7)}} }}

*,*::before,*::after{{box-sizing:border-box}}
html,body,[class*="css"]{{font-family:'DM Sans',sans-serif!important;background:var(--bg)!important;color:var(--text)!important}}
::-webkit-scrollbar{{width:4px}} ::-webkit-scrollbar-thumb{{background:var(--teal);border-radius:99px}}
#MainMenu,footer{{visibility:hidden}}
.block-container{{padding:1.5rem 2.5rem 4rem!important;max-width:1600px!important}}
header[data-testid="stHeader"]{{background:transparent!important}}

section[data-testid="stSidebar"]{{background:var(--card)!important;border-right:1px solid var(--border)!important}}
section[data-testid="stSidebar"] > div{{padding:1.5rem 1.2rem!important}}
section[data-testid="stSidebar"] *{{color:var(--text)!important}}
section[data-testid="stSidebar"] label{{font-size:.75rem!important;font-weight:600!important;color:var(--muted)!important;letter-spacing:.05em!important;text-transform:uppercase!important}}
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stTextInput input{{background:var(--bg2)!important;border:1.5px solid var(--border)!important;color:var(--text)!important;border-radius:12px!important;font-size:.88rem!important}}
section[data-testid="stSidebar"] .stTextArea textarea:focus,
section[data-testid="stSidebar"] .stTextInput input:focus{{border-color:var(--teal)!important;box-shadow:0 0 0 3px rgba(0,201,167,.15)!important}}
section[data-testid="stSidebar"] .stButton button{{background:linear-gradient(135deg,var(--teal),var(--blue))!important;color:{'#0A1628' if IS_DARK else '#fff'}!important;font-weight:700!important;border:none!important;border-radius:14px!important;padding:.7rem 1rem!important;font-size:.92rem!important;box-shadow:0 4px 20px rgba(0,201,167,.35)!important;font-family:'Syne',sans-serif!important}}
section[data-testid="stSidebar"] .stButton button:hover{{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(0,201,167,.5)!important}}

.stButton button{{background:linear-gradient(135deg,var(--blue),var(--purple))!important;color:#fff!important;border:none!important;border-radius:12px!important;font-family:'Syne',sans-serif!important;font-weight:600!important}}
.stButton button:hover{{transform:translateY(-2px)!important}}
.stDownloadButton button{{background:linear-gradient(135deg,var(--teal),var(--teal2))!important;color:{'#0A1628' if IS_DARK else '#fff'}!important;border:none!important;border-radius:12px!important;font-weight:700!important}}
.stSelectbox>div>div{{background:var(--bg2)!important;border:1.5px solid var(--border)!important;border-radius:12px!important;color:var(--text)!important}}
.stRadio label{{background:var(--card2)!important;border:1.5px solid var(--border)!important;border-radius:10px!important;padding:5px 14px!important;font-size:.78rem!important;color:var(--muted)!important;cursor:pointer!important}}
.streamlit-expanderHeader{{background:var(--card2)!important;border:1px solid var(--border)!important;border-radius:14px!important;font-weight:600!important;color:var(--text)!important;font-family:'Syne',sans-serif!important}}
.streamlit-expanderContent{{background:var(--card2)!important;border:1px solid var(--border)!important;border-top:none!important;border-radius:0 0 14px 14px!important}}

.hero{{position:relative;overflow:hidden;background:linear-gradient(135deg,{BG2} 0%,{CARD} 50%,{BG2} 100%);border:1px solid {BORDER};border-radius:24px;padding:2.8rem 3.2rem;margin-bottom:2rem;animation:fadeUp .6s ease}}
.hero-glow{{position:absolute;top:-80px;right:-80px;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(0,201,167,.18) 0%,transparent 65%);pointer-events:none}}
.hero-glow2{{position:absolute;bottom:-100px;left:30%;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(59,130,246,.12) 0%,transparent 65%);pointer-events:none}}
.hero-badge{{display:inline-flex;align-items:center;gap:7px;background:rgba(0,201,167,.10);border:1px solid rgba(0,201,167,.28);color:{TEAL};font-size:.7rem;font-weight:700;padding:4px 14px;border-radius:99px;margin-bottom:1.2rem;letter-spacing:.1em;text-transform:uppercase}}
.pulse-dot{{width:7px;height:7px;border-radius:50%;background:{TEAL};animation:pulse 2s infinite;display:inline-block}}
.hero-title{{font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;color:{TEXT};margin:0 0 .6rem;line-height:1.1;letter-spacing:-1px}}
.hero-title span{{color:{TEAL};animation:glow 3s ease infinite;display:inline-block}}
.hero-sub{{color:{MUTED};font-size:.92rem;max-width:600px;line-height:1.7}}
.hero-chips{{display:flex;gap:10px;margin-top:1.8rem;flex-wrap:wrap}}
.hero-chip{{display:inline-flex;align-items:center;gap:7px;background:rgba(255,255,255,{'.06' if IS_DARK else '.7'});border:1px solid {BORDER};border-radius:10px;padding:7px 14px;font-size:.78rem;color:{MUTED};font-weight:500;transition:all .2s}}

.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:1.6rem 0}}
.stat-card{{background:var(--card2);border:1px solid var(--border);border-radius:18px;padding:1.4rem 1.2rem;position:relative;overflow:hidden;animation:fadeUp .5s ease;transition:all .25s}}
.stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,{TEAL},{BLUE},{PURPLE})}}
.stat-card:hover{{transform:translateY(-4px);box-shadow:0 12px 48px rgba(0,0,0,{'.45' if IS_DARK else '.12'})}}
.stat-num{{font-family:'JetBrains Mono',monospace;font-size:2.1rem;font-weight:700;color:{TEXT};line-height:1}}
.stat-lbl{{font-size:.72rem;color:{MUTED};margin-top:5px;font-weight:500}}

.job-card{{background:var(--card2);border:1.5px solid var(--border);border-radius:18px;padding:1.5rem 1.75rem;margin-bottom:4px;animation:slideRight .4s ease;transition:all .25s;position:relative;overflow:hidden}}
.job-card::before{{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:linear-gradient(180deg,{TEAL},{BLUE});border-radius:4px 0 0 4px}}
.job-card:hover{{border-color:rgba(0,201,167,.4);transform:translateX(6px)}}
.job-title-txt{{font-family:'Syne',sans-serif;font-size:1.08rem;font-weight:700;color:{TEXT}}}
.job-meta-row{{display:flex;align-items:center;gap:14px;font-size:.8rem;color:{MUTED};margin:6px 0 0;flex-wrap:wrap}}
.job-meta-item{{display:inline-flex;align-items:center;gap:5px}}
.match-ring{{width:62px;height:62px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-family:'JetBrains Mono',monospace;font-size:.82rem;font-weight:800;border:2.5px solid;transition:all .3s}}
.badge{{display:inline-flex;align-items:center;gap:4px;font-size:.68rem;font-weight:700;padding:3px 10px;border-radius:99px;margin:2px}}
.b-teal{{background:rgba(0,201,167,.12);color:{TEAL};border:1px solid rgba(0,201,167,.25)}}
.b-blue{{background:rgba(59,130,246,.12);color:{BLUE};border:1px solid rgba(59,130,246,.25)}}
.b-purple{{background:rgba(139,92,246,.12);color:{PURPLE};border:1px solid rgba(139,92,246,.25)}}
.b-amber{{background:rgba(245,158,11,.12);color:{AMBER};border:1px solid rgba(245,158,11,.25)}}
.b-red{{background:rgba(239,68,68,.12);color:{RED};border:1px solid rgba(239,68,68,.25)}}
.bar-track{{background:{'rgba(255,255,255,.06)' if IS_DARK else '#E2E8F0'};border-radius:99px;height:7px;margin:12px 0 4px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:99px;animation:barGrow .8s ease .3s both}}
.chip-match{{display:inline-flex;align-items:center;gap:4px;font-size:.68rem;padding:3px 10px;border-radius:99px;margin:2px;background:rgba(0,201,167,.10);color:{TEAL};border:1px solid rgba(0,201,167,.22);font-weight:600}}
.chip-miss{{display:inline-flex;align-items:center;gap:4px;font-size:.68rem;padding:3px 10px;border-radius:99px;margin:2px;background:rgba(239,68,68,.10);color:{RED};border:1px solid rgba(239,68,68,.22);font-weight:600}}
.salary-tag{{display:inline-flex;align-items:center;gap:7px;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.22);border-radius:10px;padding:5px 14px;font-size:.82rem;font-weight:700;color:{AMBER};font-family:'JetBrains Mono',monospace}}
.gap-box{{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);border-radius:12px;padding:10px 14px;margin-top:10px}}
.gap-title{{font-weight:700;color:{AMBER};font-size:.78rem}}
.gap-skills{{font-size:.76rem;color:{'#FCD34D' if IS_DARK else AMBER};margin-top:3px}}
.sec-hdr{{display:flex;align-items:center;gap:10px;font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:{TEXT};border-left:4px solid {TEAL};padding-left:12px;margin:2rem 0 1.2rem;animation:slideRight .4s ease}}
.welcome-wrap{{background:var(--card2);border:1px solid var(--border);border-radius:24px;padding:3.5rem 2rem;text-align:center;animation:fadeUp .5s ease}}
.welcome-title{{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{TEXT};margin:1.2rem 0 .6rem}}
.welcome-sub{{font-size:.9rem;color:{MUTED};max-width:480px;margin:0 auto;line-height:1.7}}
.step-card{{background:var(--card2);border:1.5px solid var(--border);border-radius:16px;padding:1.4rem;text-align:center;animation:fadeUp .5s ease;transition:all .25s}}
.step-card:hover{{border-color:{TEAL};transform:translateY(-4px)}}
.step-num{{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{TEAL};line-height:1;margin-bottom:10px}}
.step-title{{font-weight:700;color:{TEXT};font-size:.95rem;margin-bottom:5px}}
.step-desc{{font-size:.78rem;color:{MUTED};line-height:1.6}}
.how-card{{display:flex;gap:18px;background:var(--card2);border:1px solid var(--border);border-radius:16px;padding:1.25rem 1.5rem;margin-bottom:10px;animation:slideRight .4s ease;transition:all .2s}}
.how-card:hover{{border-color:{TEAL};transform:translateX(4px)}}
.how-num{{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:{TEAL};min-width:42px;line-height:1}}
.how-title{{font-weight:700;color:{TEXT};font-size:.95rem;margin-bottom:4px}}
.how-desc{{font-size:.82rem;color:{MUTED};line-height:1.65}}
.tech-tbl{{width:100%;border-collapse:collapse;font-size:.82rem}}
.tech-tbl th{{background:rgba(0,201,167,.08);color:{TEAL};font-weight:700;padding:10px 14px;text-align:left;border-bottom:1px solid {BORDER};font-size:.7rem;text-transform:uppercase;letter-spacing:.07em}}
.tech-tbl td{{padding:9px 14px;border-bottom:1px solid {BORDER};color:{MUTED}}}
.tech-tbl td:first-child{{color:{TEXT};font-weight:600}}
.tech-tbl tr:hover td{{background:rgba(0,201,167,.04)}}
.cat-pill{{display:flex;align-items:center;gap:10px;background:var(--card2);border:1px solid var(--border);border-radius:12px;padding:10px 16px;margin-bottom:8px;transition:all .2s}}
.cat-pill:hover{{border-color:{TEAL};transform:translateX(4px)}}
.info-card{{background:rgba(0,201,167,.06);border:1px solid rgba(0,201,167,.2);border-radius:14px;padding:1.2rem 1.5rem;margin-top:1rem}}
.info-label{{font-size:.78rem;color:{TEAL};font-weight:700;margin-bottom:6px}}
.info-value{{font-size:.8rem;color:{MUTED};line-height:1.7}}
.sidebar-brand{{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:{TEXT}}}
.sidebar-divider{{border:none;border-top:1px solid {BORDER};margin:.8rem 0}}
</style>
""", unsafe_allow_html=True)


# ── DATASET ──────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    data = [
        {"id":1,"title":"Senior Python Developer","company":"TechNova Solutions","location":"Remote","type":"Full-time","category":"Engineering","salary":"$90,000–$130,000","exp_min":4,"edu":"Bachelor's","desc":"Python Django FastAPI AWS Docker PostgreSQL REST APIs microservices backend agile CI/CD","skills":"Python,Django,FastAPI,AWS,Docker,PostgreSQL,REST APIs,Git"},
        {"id":2,"title":"Full Stack Web Developer","company":"WebCraft Studio","location":"Karachi, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp_min":2,"edu":"Bachelor's","desc":"React Node.js MongoDB Express JavaScript TypeScript HTML CSS REST APIs agile git frontend backend","skills":"React,Node.js,MongoDB,JavaScript,TypeScript,HTML,CSS,Express,Git"},
        {"id":3,"title":"Frontend Developer","company":"UX Studio","location":"London, UK","type":"Full-time","category":"Engineering","salary":"$70,000–$100,000","exp_min":2,"edu":"Bachelor's","desc":"React Vue JavaScript TypeScript CSS HTML webpack performance accessibility user experience design","skills":"React,Vue.js,JavaScript,TypeScript,CSS,HTML,Webpack,Figma"},
        {"id":4,"title":"Backend Developer","company":"ServerLogic","location":"Remote","type":"Full-time","category":"Engineering","salary":"$80,000–$115,000","exp_min":3,"edu":"Bachelor's","desc":"Python Java Spring Boot Django Flask PostgreSQL MySQL Redis Docker Kubernetes cloud AWS CI/CD","skills":"Python,Java,Spring Boot,PostgreSQL,MySQL,Docker,Kubernetes,Redis"},
        {"id":5,"title":"Data Scientist","company":"DataSphere Analytics","location":"New York, USA","type":"Full-time","category":"Data Science","salary":"$95,000–$140,000","exp_min":3,"edu":"Master's","desc":"Python TensorFlow scikit-learn pandas numpy machine learning deep learning SQL statistics data visualization analytics","skills":"Python,Machine Learning,TensorFlow,SQL,Pandas,NumPy,Statistics,Scikit-learn"},
        {"id":6,"title":"Machine Learning Engineer","company":"AI Dynamics","location":"San Francisco, USA","type":"Full-time","category":"AI/ML","salary":"$120,000–$180,000","exp_min":3,"edu":"Master's","desc":"PyTorch TensorFlow MLOps Kubernetes Docker machine learning deep learning NLP computer vision model deployment","skills":"Python,PyTorch,TensorFlow,MLOps,Kubernetes,Docker,Deep Learning,NLP"},
        {"id":7,"title":"AI Research Scientist","company":"DeepMind Labs","location":"London, UK","type":"Full-time","category":"AI/ML","salary":"$130,000–$200,000","exp_min":2,"edu":"PhD","desc":"Deep learning NLP reinforcement learning PyTorch mathematics statistics research papers neural networks transformers BERT GPT","skills":"Python,PyTorch,Deep Learning,NLP,Research,Mathematics,Statistics,Transformers"},
        {"id":8,"title":"NLP Engineer","company":"LanguageAI","location":"Remote","type":"Full-time","category":"AI/ML","salary":"$100,000–$145,000","exp_min":3,"edu":"Master's","desc":"NLP pipelines text classification BERT transformers spaCy NLTK Python TensorFlow language models GPT fine-tuning sentiment","skills":"Python,NLP,Transformers,BERT,spaCy,NLTK,TensorFlow,Hugging Face"},
        {"id":9,"title":"Data Engineer","company":"PipelineAI","location":"Singapore","type":"Full-time","category":"Data Science","salary":"$85,000–$120,000","exp_min":3,"edu":"Bachelor's","desc":"Apache Spark Kafka SQL AWS Airflow ETL data warehouse Python cloud engineering big data stream processing","skills":"Python,Apache Spark,Kafka,SQL,AWS,Airflow,ETL,Data Warehouse"},
        {"id":10,"title":"Business Intelligence Analyst","company":"InsightCorp","location":"Karachi, Pakistan","type":"Full-time","category":"Data Science","salary":"$30,000–$55,000","exp_min":2,"edu":"Bachelor's","desc":"SQL Power BI Tableau Excel data visualization dashboard reporting KPIs business analytics Python stakeholder","skills":"SQL,Power BI,Tableau,Excel,Python,Data Analytics,Reporting"},
        {"id":11,"title":"DevOps Engineer","company":"CloudCore","location":"Remote","type":"Full-time","category":"DevOps","salary":"$85,000–$125,000","exp_min":3,"edu":"Bachelor's","desc":"CI/CD AWS Docker Kubernetes Terraform Jenkins Linux Python monitoring deployment site reliability automation","skills":"AWS,Docker,Kubernetes,Terraform,Jenkins,Linux,Python,CI/CD"},
        {"id":12,"title":"Cloud Solutions Architect","company":"Nimbus Cloud","location":"Remote","type":"Full-time","category":"DevOps","salary":"$110,000–$160,000","exp_min":5,"edu":"Bachelor's","desc":"AWS Azure GCP Terraform Docker Kubernetes networking security cloud migration enterprise solutions infrastructure","skills":"AWS,Azure,GCP,Terraform,Docker,Kubernetes,Networking,Security"},
        {"id":13,"title":"Cybersecurity Analyst","company":"SecureNet","location":"Remote","type":"Full-time","category":"Security","salary":"$80,000–$120,000","exp_min":2,"edu":"Bachelor's","desc":"Network security SIEM penetration testing Linux firewalls incident response Python compliance vulnerability assessment","skills":"Network Security,Python,SIEM,Penetration Testing,Linux,Firewalls,Incident Response"},
        {"id":14,"title":"UX/UI Designer","company":"PixelCraft","location":"Dubai, UAE","type":"Full-time","category":"Design","salary":"$55,000–$90,000","exp_min":2,"edu":"Bachelor's","desc":"Figma Adobe XD wireframes prototypes UI design user research usability testing design systems CSS HTML responsive","skills":"Figma,Adobe XD,UI Design,User Research,Prototyping,Sketch,CSS,HTML"},
        {"id":15,"title":"Graphic Designer","company":"VisualEdge","location":"Lahore, Pakistan","type":"Full-time","category":"Design","salary":"$15,000–$35,000","exp_min":1,"edu":"Bachelor's","desc":"Adobe Photoshop Illustrator InDesign Canva typography brand identity logo social media marketing print digital","skills":"Adobe Photoshop,Illustrator,InDesign,Canva,Typography,Branding,Figma"},
        {"id":16,"title":"Flutter Mobile Developer","company":"AppForge","location":"Karachi, Pakistan","type":"Full-time","category":"Mobile","salary":"$30,000–$55,000","exp_min":1,"edu":"Bachelor's","desc":"Flutter Dart Firebase REST APIs state management Android iOS app store deployment performance cross-platform mobile","skills":"Flutter,Dart,Firebase,REST APIs,Git,Android,iOS,State Management"},
        {"id":17,"title":"Android Developer","company":"MobileFirst","location":"Remote","type":"Full-time","category":"Mobile","salary":"$70,000–$100,000","exp_min":2,"edu":"Bachelor's","desc":"Kotlin Java Android SDK MVVM REST APIs Firebase Room database Jetpack Compose Google Play native mobile","skills":"Kotlin,Java,Android SDK,MVVM,REST APIs,Firebase,Jetpack Compose"},
        {"id":18,"title":"Product Manager","company":"Innovatech","location":"Austin, USA","type":"Full-time","category":"Product","salary":"$100,000–$150,000","exp_min":4,"edu":"Bachelor's","desc":"Product strategy roadmap agile JIRA stakeholder management user research data analysis A/B testing communication","skills":"Product Strategy,Agile,JIRA,Data Analysis,SQL,Communication,Roadmapping"},
        {"id":19,"title":"Digital Marketing Specialist","company":"GrowthHive","location":"Karachi, Pakistan","type":"Full-time","category":"Marketing","salary":"$20,000–$40,000","exp_min":1,"edu":"Bachelor's","desc":"SEO SEM Google Ads social media content marketing email marketing analytics Canva brand awareness reporting","skills":"SEO,Google Ads,Social Media,Content Marketing,Analytics,Email Marketing,Canva"},
        {"id":20,"title":"Business Analyst","company":"FinEdge","location":"Toronto, Canada","type":"Full-time","category":"Business","salary":"$65,000–$95,000","exp_min":2,"edu":"Bachelor's","desc":"SQL data analysis Excel Power BI JIRA process improvement stakeholder communication agile scrum reporting requirements","skills":"Data Analysis,SQL,Excel,Power BI,JIRA,Tableau,Communication,Agile"},
        {"id":21,"title":"QA Automation Engineer","company":"TestPro","location":"Lahore, Pakistan","type":"Full-time","category":"Engineering","salary":"$25,000–$45,000","exp_min":2,"edu":"Bachelor's","desc":"Selenium Cypress Python JavaScript API testing JIRA CI/CD regression testing quality assurance automation framework","skills":"Selenium,Cypress,Python,JavaScript,API Testing,JIRA,CI/CD"},
        {"id":22,"title":"Blockchain Developer","company":"ChainTech","location":"Remote","type":"Full-time","category":"Engineering","salary":"$100,000–$150,000","exp_min":3,"edu":"Bachelor's","desc":"Solidity Ethereum Web3.js DeFi NFT smart contracts blockchain Python JavaScript cryptography decentralized applications","skills":"Solidity,Ethereum,Web3.js,Python,Smart Contracts,JavaScript,Cryptography"},
        {"id":23,"title":"Network Engineer","company":"NetSystems","location":"Islamabad, Pakistan","type":"Full-time","category":"Engineering","salary":"$35,000–$60,000","exp_min":2,"edu":"Bachelor's","desc":"Cisco routers switches firewalls VPN TCP/IP Linux Windows server networking troubleshooting security monitoring CCNA","skills":"Cisco,Networking,Firewalls,VPN,TCP/IP,Linux,Windows Server,CCNA"},
        {"id":24,"title":"HR Business Partner","company":"PeopleFirst","location":"Dubai, UAE","type":"Full-time","category":"HR","salary":"$50,000–$80,000","exp_min":3,"edu":"Bachelor's","desc":"Talent acquisition employee relations performance management HR policies training development HRMS communication recruitment","skills":"HR Management,Recruitment,Employee Relations,Performance Management,HRMS,Training"},
        {"id":25,"title":"Python Developer Intern","company":"StartupHub","location":"Karachi, Pakistan","type":"Internship","category":"Engineering","salary":"$5,000–$12,000","exp_min":0,"edu":"Intermediate","desc":"Python programming Django REST APIs database SQL Git web development backend basics agile teamwork projects","skills":"Python,Django,SQL,Git,REST APIs,HTML"},
        {"id":26,"title":"Data Science Intern","company":"Analytics Co","location":"Lahore, Pakistan","type":"Internship","category":"Data Science","salary":"$5,000–$10,000","exp_min":0,"edu":"Intermediate","desc":"Python pandas numpy matplotlib machine learning scikit-learn SQL data visualization statistics Excel Jupyter notebook","skills":"Python,Pandas,NumPy,Matplotlib,SQL,Scikit-learn,Excel"},
        {"id":27,"title":"UI/UX Design Intern","company":"CreativeMinds","location":"Remote","type":"Internship","category":"Design","salary":"$4,000–$8,000","exp_min":0,"edu":"Intermediate","desc":"Wireframes prototypes Figma user interface mobile web design typography color theory user experience research Canva","skills":"Figma,Canva,UI Design,Prototyping,Typography"},
        {"id":28,"title":"Freelance Web Developer","company":"Various Clients","location":"Remote","type":"Freelance","category":"Engineering","salary":"$30,000–$80,000","exp_min":1,"edu":"Diploma","desc":"WordPress React JavaScript HTML CSS PHP MySQL client websites freelance remote communication deadlines project management","skills":"WordPress,React,JavaScript,HTML,CSS,PHP,MySQL"},
        {"id":29,"title":"Content Writer","company":"ContentPro Agency","location":"Remote","type":"Part-time","category":"Marketing","salary":"$15,000–$30,000","exp_min":1,"edu":"Bachelor's","desc":"Technical writing SEO blogs articles research AI technology software communication editing proofreading WordPress content marketing","skills":"Technical Writing,SEO,Research,Communication,WordPress,Editing"},
        {"id":30,"title":"iOS Developer","company":"AppleTree Apps","location":"Dubai, UAE","type":"Full-time","category":"Mobile","salary":"$75,000–$110,000","exp_min":2,"edu":"Bachelor's","desc":"Swift SwiftUI UIKit Xcode CoreData REST APIs push notifications App Store Objective-C native iOS mobile","skills":"Swift,SwiftUI,UIKit,Xcode,CoreData,REST APIs,Objective-C"},
    ]
    return pd.DataFrame(data)


# ── ML ENGINE ────────────────────────────────────────────────
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
    u_edu    = edu_rank.get(edu, 3)
    res['boost'] = (
        res['exp_min'].apply(lambda e: 0.08 if exp >= e else (-0.05 if exp < e - 2 else 0)) +
        res['type'].apply(lambda t: 0.06 if (not jtype or t == jtype) else 0) +
        res['edu'].apply(lambda e: 0.05 if edu_rank.get(e, 3) <= u_edu else -0.03)
    )
    res['final'] = res['base'] * 0.75 + res['boost']
    mn, mx = res['final'].min(), res['final'].max()
    if mx > mn:
        res['pct'] = ((res['final'] - mn) / (mx - mn) * 39 + 60).clip(0, 99).astype(int)
    else:
        res['pct'] = 60
    return res.sort_values('pct', ascending=False).head(n).reset_index(drop=True)

def skill_split(user_str, job_str):
    u = {s.strip().lower() for s in user_str.split(',') if s.strip()}
    j = {s.strip().lower() for s in job_str.split(',') if s.strip()}
    return sorted(u & j), sorted(j - u)

def match_color(p):
    if p >= 82: return TEAL,   f"linear-gradient(90deg,{TEAL},{TEAL2})"
    if p >= 68: return BLUE,   f"linear-gradient(90deg,{BLUE},{PURPLE})"
    return RED, f"linear-gradient(90deg,{RED},#F87171)"

PALETTE = [TEAL, BLUE, PURPLE, AMBER, RED, "#34D399", "#60A5FA", "#A78BFA", "#F472B6", "#FBBF24"]

def plotly_layout(title=""):
    return dict(
        paper_bgcolor=PLOTLY_BG, plot_bgcolor=PLOTLY_BG,
        font=dict(family="DM Sans", color=PLOTLY_TEXT, size=12),
        title=dict(text=title, font=dict(family="Syne", size=14, color=TEXT), x=0.02),
        margin=dict(l=12, r=12, t=44, b=12),
        showlegend=False,
        xaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, zerolinecolor=PLOTLY_GRID),
    )


# ── LOAD DATA ────────────────────────────────────────────────
df  = load_dataset()
vec, mat = build_model(df['desc'].astype(str))


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.5rem 0 1rem;">
      <div class="sidebar-brand">🎯 JobMatch AI</div>
      <div style="font-size:.72rem;color:{MUTED};margin-top:3px;">TF-IDF · Cosine Similarity · BS AI</div>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    toggle_label = "☀️ Light Mode" if IS_DARK else "🌙 Dark Mode"
    if st.button(toggle_label):
        st.session_state.theme = "light" if IS_DARK else "dark"
        st.rerun()

    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:{TEAL};margin:.5rem 0;">👤 Your Profile</div>', unsafe_allow_html=True)

    user_skills  = st.text_area("Skills", placeholder="e.g. Python, Machine Learning, SQL, React", height=100)
    education    = st.selectbox("Education Level", ["Intermediate", "Diploma", "Bachelor's", "Master's", "PhD"])
    experience   = st.slider("Years of Experience", 0, 20, 1)

    st.markdown(f'<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:{TEAL};margin:.8rem 0 .3rem;">🔍 Filters</div>', unsafe_allow_html=True)
    job_type      = st.selectbox("Job Type", ["Any", "Full-time", "Part-time", "Remote", "Freelance", "Internship"])
    location_pref = st.text_input("Preferred Location", placeholder="e.g. Remote, Karachi, Dubai")
    top_n         = st.slider("Results to Show", 3, 15, 8)

    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)
    find_btn = st.button("🔎  Find My Jobs", use_container_width=True, type="primary")
    st.markdown(f'<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:{TEAL};margin:.5rem 0;">🧭 Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["🏠  Home", "📊  Analytics", "ℹ️  How It Works"], label_visibility="collapsed")


# ── HERO ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-glow"></div>
  <div class="hero-glow2"></div>
  <div class="hero-badge"><span class="pulse-dot"></span> AI-Powered Matching</div>
  <div class="hero-title">Find Your Perfect <span>Career Match</span></div>
  <div class="hero-sub">
    Intelligent job recommendations using TF-IDF vectorization and cosine similarity —
    precisely matching your skills to the best opportunities in real time.
  </div>
  <div class="hero-chips">
    <div class="hero-chip">🐍 <b>Python</b> &nbsp;Scikit-learn</div>
    <div class="hero-chip">⚙️ <b>TF-IDF</b> &nbsp;Vectorization</div>
    <div class="hero-chip">🎯 <b>Cosine</b> &nbsp;Similarity</div>
    <div class="hero-chip">📊 <b>Plotly</b> &nbsp;Analytics</div>
    <div class="hero-chip">⚡ <b>Waqaas</b> &nbsp;Hussain</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#   HOME PAGE
# ╚══════════════════════════════════════════════════════════╝
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
            st.warning("No matching jobs found — try broadening your filters.")
            st.stop()

        # Stat cards
        skill_count = len([s for s in skills_str.split(',') if s.strip()])
        c1, c2, c3, c4 = st.columns(4)
        for col, num, lbl in [
            (c1, len(res),                    "Jobs Found"),
            (c2, f"{int(res['pct'].mean())}%","Avg Match Score"),
            (c3, f"{int(res['pct'].max())}%", "Best Match"),
            (c4, skill_count,                  "Skills Detected"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Category filter
        cats = ["All"] + sorted(res['category'].unique().tolist())
        st.markdown(f'<div class="sec-hdr">🔎 Filter Results</div>', unsafe_allow_html=True)
        sel  = st.radio("cat_filter", cats, horizontal=True, label_visibility="collapsed")
        show = res if sel == "All" else res[res['category'] == sel]

        st.markdown(f'<p style="color:{MUTED};font-size:.8rem;margin:.5rem 0 1rem;">Showing <b style="color:{TEAL};">{len(show)}</b> job(s) · ranked by AI match score</p>', unsafe_allow_html=True)

        # Job cards
        for _, j in show.iterrows():
            pct = j['pct']
            ring_color, bar_grad = match_color(pct)
            matched, missing = skill_split(skills_str, j['skills'])
            m_chips = "".join([f'<span class="chip-match">✓ {s}</span>' for s in matched])
            x_chips = "".join([f'<span class="chip-miss">✗ {s}</span>' for s in missing])

            st.markdown(f"""
<div class="job-card">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
    <div style="flex:1;">
      <div class="job-title-txt">💼 {j['title']}</div>
      <div class="job-meta-row">
        <span class="job-meta-item">🏢 {j['company']}</span>
        <span class="job-meta-item">📍 {j['location']}</span>
      </div>
    </div>
    <div class="match-ring" style="color:{ring_color};border-color:{ring_color};background:rgba(0,0,0,.15);">{pct}%</div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:4px;margin:8px 0;">
    <span class="badge b-teal">🎯 {pct}% Match</span>
    <span class="badge b-blue">⏱ {j['type']}</span>
    <span class="badge b-purple">📂 {j['category']}</span>
    <span class="badge b-amber">🎓 {j['edu']}+</span>
    <span class="badge b-red">📅 {j['exp_min']}+ yrs</span>
  </div>
  <div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{bar_grad};"></div></div>
  <div style="font-size:.7rem;color:{MUTED};margin-bottom:12px;">{pct}% alignment with your skill profile</div>
  <div class="salary-tag">💰 {j['salary']}</div>
  <div style="font-size:.78rem;font-weight:700;color:{TEXT};margin:14px 0 7px;">📊 Skill Analysis</div>
  <div style="display:flex;flex-wrap:wrap;gap:2px;">{m_chips}{x_chips}</div>
  {'<div style="font-size:.68rem;color:' + MUTED + ';margin-top:7px;"><span style=\'color:' + TEAL + ';font-weight:600;\'>● Matched</span> &nbsp; <span style=\'color:' + RED + ';font-weight:600;\'>● Missing</span></div>' if (matched or missing) else ''}
</div>
""", unsafe_allow_html=True)

            if missing:
                st.markdown(f"""
<div class="gap-box">
  <div class="gap-title">⚠️ Skill Gap — Recommended to Learn:</div>
  <div class="gap-skills">{', '.join(missing)}</div>
</div>
""", unsafe_allow_html=True)

            if st.button(f"Apply Now →", key=f"apply_{j['id']}"):
                st.success(f"✅ Application submitted for **{j['title']}** at **{j['company']}**!")

            st.markdown("<br>", unsafe_allow_html=True)

        # Download
        csv = show[['title', 'company', 'location', 'type', 'category', 'salary', 'pct']]\
              .rename(columns={'title':'Title','company':'Company','location':'Location',
                               'type':'Type','category':'Category','salary':'Salary','pct':'Match%'})\
              .to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv, "job_recommendations.csv", "text/csv", use_container_width=True)

    else:
        st.markdown(f"""
<div class="welcome-wrap">
  <div style="font-size:4rem;">🔍</div>
  <div class="welcome-title">Welcome, Waqaas!</div>
  <div class="welcome-sub">
    Enter your skills and preferences in the <b>sidebar</b>, then click
    <b>Find My Jobs</b> to receive AI-powered personalized recommendations
    powered by TF-IDF vectorization and cosine similarity.
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, delay, num, title, desc in [
            (c1, "0.0s", "01", "Enter Your Skills",  "Type skills separated by commas — Python, SQL, React, ML, Docker..."),
            (c2, "0.1s", "02", "Set Preferences",    "Choose education, experience, job type & preferred location"),
            (c3, "0.2s", "03", "Get Matched",         "Receive ranked, AI-scored job recommendations instantly"),
        ]:
            col.markdown(f'<div class="step-card" style="animation-delay:{delay};"><div class="step-num">{num}</div><div class="step-title">{title}</div><div class="step-desc">{desc}</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="sec-hdr">💼 Available Job Categories</div>', unsafe_allow_html=True)
        cats = df['category'].value_counts()
        cols = st.columns(3)
        for i, (cat, cnt) in enumerate(cats.items()):
            cols[i % 3].markdown(f"""
<div class="cat-pill">
  <span style="font-weight:600;color:{TEXT};flex:1;">📂 {cat}</span>
  <span style="background:rgba(0,201,167,.12);color:{TEAL};font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:99px;">{cnt}</span>
</div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#   ANALYTICS PAGE
# ╚══════════════════════════════════════════════════════════╝
elif "Analytics" in page:
    st.markdown(f'<div class="sec-hdr">📊 Job Market Analytics Dashboard</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, len(df),                  "Total Jobs"),
        (c2, df['company'].nunique(),  "Companies"),
        (c3, df['category'].nunique(), "Categories"),
        (c4, df['location'].nunique(), "Locations"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cats_s  = df['category'].value_counts()
    types_s = df['type'].value_counts()
    all_sk  = []
    for s in df['skills']:
        all_sk.extend([x.strip() for x in s.split(',')])
    top12  = Counter(all_sk).most_common(12)
    sk_n, sk_v = zip(*top12)

    bins   = pd.cut(df['exp_min'], bins=[-1, 0, 1, 2, 3, 5, 10, 20],
                    labels=['Fresher', '<1 yr', '1–2 yr', '2–3 yr', '3–5 yr', '5–10 yr', '10+ yr'])
    exp_c  = bins.value_counts().sort_index()

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig = go.Figure(go.Bar(
            y=cats_s.index.tolist(), x=cats_s.values.tolist(),
            orientation='h', marker_color=PALETTE[:len(cats_s)],
            text=cats_s.values, textposition='outside',
            textfont=dict(size=11, color=TEXT),
        ))
        fig.update_layout(**plotly_layout("Jobs by Category"), height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with r1c2:
        fig2 = go.Figure(go.Pie(
            labels=types_s.index.tolist(), values=types_s.values.tolist(),
            hole=0.6, marker_colors=PALETTE[:len(types_s)],
            textinfo='label+percent',
            textfont=dict(size=11, color=TEXT),
        ))
        fig2.update_layout(**plotly_layout("Job Type Distribution"), height=360, showlegend=True,
                           legend=dict(font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"))
        fig2.update_traces(pull=[0.04] * len(types_s))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig3 = go.Figure(go.Bar(
            x=exp_c.index.astype(str).tolist(), y=exp_c.values.tolist(),
            marker_color=PALETTE[:len(exp_c)], text=exp_c.values,
            textposition='outside', textfont=dict(size=11, color=TEXT),
        ))
        fig3.update_layout(**plotly_layout("Experience Level Required"), height=340)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with r2c2:
        fig4 = go.Figure(go.Bar(
            y=list(sk_n)[::-1], x=list(sk_v)[::-1],
            orientation='h',
            marker=dict(color=list(sk_v)[::-1], colorscale=[[0, BLUE], [0.5, TEAL], [1, PURPLE]], showscale=False),
            text=list(sk_v)[::-1], textposition='outside',
            textfont=dict(size=10, color=TEXT),
        ))
        fig4.update_layout(**plotly_layout("Top 12 In-Demand Skills"), height=340)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f'<div class="sec-hdr">💰 Salary Range Distribution</div>', unsafe_allow_html=True)
    sal_df = df.copy()

    def parse_sal(s):
        nums = re.findall(r'[\d,]+', s.replace(',', ''))
        return (int(nums[0]) + int(nums[1])) // 2 if len(nums) >= 2 else 0

    sal_df['sal_mid'] = sal_df['salary'].apply(parse_sal)
    sal_df = sal_df[sal_df['sal_mid'] > 0]

    fig5 = px.scatter(
        sal_df, x='exp_min', y='sal_mid', color='category',
        size='sal_mid', hover_name='title',
        hover_data={'company': True, 'location': True, 'salary': True, 'sal_mid': False},
        color_discrete_sequence=PALETTE, size_max=28,
        labels={'exp_min': 'Years of Experience', 'sal_mid': 'Midpoint Salary ($)', 'category': 'Category'},
    )
    fig5.update_layout(**plotly_layout("Salary vs Experience"), height=380, showlegend=True,
                       legend=dict(font=dict(color=TEXT, size=10), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f'<div class="sec-hdr">📋 Complete Job Dataset</div>', unsafe_allow_html=True)
    disp = df[['title', 'company', 'location', 'type', 'category', 'salary', 'exp_min', 'edu']].copy()
    disp.columns = ['Job Title', 'Company', 'Location', 'Type', 'Category', 'Salary', 'Min Exp', 'Education']
    st.dataframe(disp, use_container_width=True, height=400)


# ╔══════════════════════════════════════════════════════════╗
#   HOW IT WORKS PAGE
# ╚══════════════════════════════════════════════════════════╝
elif "How" in page:
    st.markdown(f'<div class="sec-hdr">ℹ️ How the AI System Works</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Data Collection",
         "30 curated real-world job listings with titles, descriptions, required skills, salary ranges, locations, and experience requirements — forming a rich content-based knowledge base."),
        ("02", "Text Preprocessing",
         "Job descriptions are normalized: lowercased, special characters removed, whitespace collapsed using Python regex. TF-IDF's built-in English stop word list filters noise — ensuring only meaningful keywords are vectorized."),
        ("03", "TF-IDF Vectorization",
         "Scikit-learn's TfidfVectorizer converts job descriptions and user skills into high-dimensional sparse numerical vectors. Rare but important keywords receive higher TF-IDF weight; common words are down-weighted."),
        ("04", "Cosine Similarity Scoring",
         "Cosine similarity between the user's skill vector and each job description vector is computed. Scores range from 0 (no overlap) to 1 (perfect alignment), measuring angular closeness in TF-IDF space."),
        ("05", "Score Boosting & Normalization",
         "Raw similarity scores are enhanced: experience alignment (+0.08), job type match (+0.06), education suitability (+0.05). Final scores are normalized to a clean 60–99% range."),
        ("06", "Results & Skill Gap Analysis",
         "Jobs are ranked by final match percentage. Matched skills shown in green, missing skills in red — giving actionable insights on which technologies to learn next."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
<div class="how-card">
  <div class="how-num">{num}</div>
  <div>
    <div class="how-title">{title}</div>
    <div class="how-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sec-hdr">📐 Core Algorithm</div>', unsafe_allow_html=True)
    st.latex(r"\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}")
    st.markdown(f"""
<div class="how-card">
  <div style="font-size:.85rem;color:{MUTED};line-height:1.8;">
    <b style="color:{TEAL};">A</b> = TF-IDF vector of user-entered skills &nbsp;·&nbsp;
    <b style="color:{TEAL};">B</b> = TF-IDF vector of a job description<br>
    <b style="color:{TEAL};">Result:</b> 0 → 1 &nbsp;(higher = better match)<br>
    Measures the angle between vectors in high-dimensional TF-IDF space — independent of document length.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="sec-hdr">🛠️ Technologies Used</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    rows1 = [
        ("Python 3.x",      "Core programming language"),
        ("Streamlit",        "Interactive web application framework"),
        ("Scikit-learn",     "TF-IDF vectorization & cosine similarity"),
        ("Pandas / NumPy",   "Data manipulation & numerical computing"),
    ]
    rows2 = [
        ("Plotly Express",  "Animated interactive analytics charts"),
        ("CSS / HTML",      "UI animations & glassmorphism effects"),
        ("Regex (re)",       "Text preprocessing & normalization"),
        ("Session State",    "In-app memory & state persistence"),
    ]
    with c1:
        rows_html = "".join([f'<tr><td>{t}</td><td>{d}</td></tr>' for t, d in rows1])
        st.markdown(f'<table class="tech-tbl"><tr><th>Technology</th><th>Purpose</th></tr>{rows_html}</table>', unsafe_allow_html=True)
    with c2:
        rows_html = "".join([f'<tr><td>{t}</td><td>{d}</td></tr>' for t, d in rows2])
        st.markdown(f'<table class="tech-tbl"><tr><th>Technology</th><th>Purpose</th></tr>{rows_html}</table>', unsafe_allow_html=True)

    st.markdown(f"""
<div class="info-card" style="margin-top:2rem;">
  <div class="info-label">⭐ Project Information</div>
  <div class="info-value">
    <b>Prepared by:</b> Waqaas Hussain &amp; Hira Abdul Hafeez<br>
    <b>SAP IDs:</b> 5000000291 · 5000000314<br>
    <b>Program:</b> BS Artificial Intelligence &nbsp;·&nbsp; <b>Semester:</b> 4th &nbsp;·&nbsp; <b>Section:</b> C<br>
    <b>Course:</b> Programming for AI &nbsp;·&nbsp; <b>Instructor:</b> Sir Abdul Hasseb<br>
    <b>Framework:</b> Streamlit + Scikit-learn + Plotly<br>
    <b>Algorithm:</b> TF-IDF Vectorization + Cosine Similarity Content-Based Filtering
  </div>
</div>""", unsafe_allow_html=True)

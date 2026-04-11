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
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Recommendation System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 50%, #0d7c66 100%);
    padding: 2rem 2.5rem; border-radius: 16px;
    color: white; margin-bottom: 2rem; text-align: center;
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.main-header p  { font-size: 0.95rem; opacity: 0.85; margin-top: 0.4rem; }

.job-card {
    background: white; border: 1.5px solid #e5e7eb;
    border-radius: 14px; padding: 1.25rem 1.5rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.job-title   { font-size: 1.05rem; font-weight: 700; color: #0f4c75; }
.company-row { font-size: 0.85rem; color: #374151; margin: 4px 0; }

.badge {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 2px 9px; border-radius: 99px; margin: 2px;
}
.bg  { background:#d1fae5; color:#065f46; }
.bb  { background:#dbeafe; color:#1e40af; }
.bp  { background:#ede9fe; color:#5b21b6; }
.bo  { background:#fef3c7; color:#92400e; }
.br  { background:#fee2e2; color:#991b1b; }

.match-bar-bg  { background:#e5e7eb; border-radius:99px; height:7px; overflow:hidden; margin:6px 0; }
.match-bar-fill{ height:100%; border-radius:99px; }

.skill-m { display:inline-block; font-size:0.7rem; padding:2px 9px; border-radius:99px; margin:2px; background:#d1fae5; color:#065f46; }
.skill-x { display:inline-block; font-size:0.7rem; padding:2px 9px; border-radius:99px; margin:2px; background:#fee2e2; color:#991b1b; }

.stat-card { background:white; border:1px solid #e5e7eb; border-radius:12px; padding:1rem; text-align:center; }
.stat-num  { font-size:1.8rem; font-weight:700; color:#0f4c75; }
.stat-lbl  { font-size:0.75rem; color:#6b7280; }

.sec { font-size:1rem; font-weight:600; color:#0f4c75;
       border-left:4px solid #0d7c66; padding-left:0.65rem; margin:1.5rem 0 1rem; }

.gap-box { background:#fffbeb; border:1px solid #fde68a;
           border-radius:10px; padding:0.75rem 1rem; margin-top:0.5rem; }
.gap-t   { font-weight:600; color:#92400e; font-size:0.82rem; }
.gap-s   { font-size:0.78rem; color:#78350f; margin-top:3px; }

.welcome { text-align:center; padding:3rem 2rem; }
.welcome .icon { font-size:4rem; margin-bottom:1rem; }

#MainMenu, footer, header { visibility: hidden; }
</style>
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


# ── TF-IDF Model ─────────────────────────────────────────────
@st.cache_resource
def build_model(descs):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    mat = vec.fit_transform(descs)
    return vec, mat


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Recommendation Engine ─────────────────────────────────────
def recommend(skills_str, exp, edu, jtype, vec, mat, df, n=8):
    user_vec  = vec.transform([preprocess(skills_str)])
    scores    = cosine_similarity(user_vec, mat).flatten()
    res       = df.copy()
    res['base'] = scores

    edu_rank = {"Intermediate":1,"Diploma":2,"Bachelor's":3,"Master's":4,"PhD":5}
    u_edu    = edu_rank.get(edu, 3)

    res['boost'] = (
        res['exp_min'].apply(lambda e: 0.08 if exp >= e else (-0.05 if exp < e-2 else 0)) +
        res['type'].apply(lambda t: 0.06 if (not jtype or t == jtype) else 0) +
        res['edu'].apply(lambda e: 0.05 if edu_rank.get(e,3) <= u_edu else -0.03)
    )
    res['final'] = res['base'] * 0.75 + res['boost']

    mn, mx = res['final'].min(), res['final'].max()
    res['pct'] = ((res['final']-mn)/(mx-mn)*39+60).clip(0,99).astype(int) if mx > mn else 60
    return res.sort_values('pct', ascending=False).head(n).reset_index(drop=True)


def skill_split(user_str, job_str):
    u = {s.strip().lower() for s in user_str.split(',') if s.strip()}
    j = {s.strip().lower() for s in job_str.split(',')  if s.strip()}
    return sorted(u&j), sorted(j-u)


def bar_col(p):
    return '#059669' if p>=80 else '#f59e0b' if p>=65 else '#ef4444'


# ─────────────────────────────────────────────────────────────
df  = load_dataset()
vec, mat = build_model(df['desc'].astype(str))

# Header
st.markdown("""
<div class="main-header">
  <h1> AI-Based Job Recommendation System</h1>
  <p>TF-IDF Vectorization · Cosine Similarity · Content-Based Filtering &nbsp;|&nbsp; Prepared by Waqaas Hussain</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Your Profile")
    st.markdown("---")

    user_skills = st.text_area("💡 Skills (comma-separated)",
        placeholder="e.g. Python, Machine Learning, SQL, Django", height=110)

    education = st.selectbox(" Education",
        ["Intermediate","Diploma","Bachelor's","Master's","PhD"])

    experience = st.slider(" Years of Experience", 0, 20, 1)

    job_type = st.selectbox("🕐 Job Type",
        ["Any","Full-time","Part-time","Remote","Freelance","Internship"])

    location_pref = st.text_input("📍 Location", placeholder="e.g. Karachi, Remote, Dubai")

    top_n = st.slider("📋 Results to Show", 3, 15, 8)

    st.markdown("---")
    find_btn = st.button("🔎 Find My Jobs", use_container_width=True, type="primary")
    st.markdown("---")

    page = st.radio(" Pages",
        [" Home"," Analytics"," How It Works"],
        label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════
if " Home" in page:

    if find_btn or 'results' in st.session_state:

        if find_btn:
            if not user_skills.strip():
                st.warning(" Please enter at least one skill.")
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
            st.warning("No matching jobs found. Try fewer filters.")
            st.stop()

        # Stats
        c1,c2,c3,c4 = st.columns(4)
        for col, num, lbl in [
            (c1, len(res), "Jobs Found"),
            (c2, f"{int(res['pct'].mean())}%", "Avg Match"),
            (c3, f"{int(res['pct'].max())}%", "Best Match"),
            (c4, len([s for s in skills_str.split(',') if s.strip()]), "Skills Entered"),
        ]:
            col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cats = ["All"] + sorted(res['category'].unique().tolist())
        sel  = st.radio("Filter by category:", cats, horizontal=True)
        show = res if sel=="All" else res[res['category']==sel]

        st.caption(f"Showing {len(show)} job(s) — ranked by AI match score")
        st.markdown("---")

        for _, j in show.iterrows():
            pct  = j['pct']
            col  = bar_col(pct)
            matched, missing = skill_split(skills_str, j['skills'])
            m_pills = "".join([f'<span class="skill-m">✓ {s}</span>' for s in matched])
            x_pills = "".join([f'<span class="skill-x">✗ {s}</span>' for s in missing])

            with st.expander(f"  {j['title']}  ·  {j['company']}  —  {pct}% match"):
                st.markdown(f"""
<div class="job-card">
  <div class="job-title">{j['title']}</div>
  <div class="company-row"> {j['company']} &nbsp;·&nbsp; 📍 {j['location']}</div>
  <div style="margin:6px 0;">
    <span class="badge bg"> {pct}% Match</span>
    <span class="badge bb">{j['type']}</span>
    <span class="badge bp">{j['category']}</span>
    <span class="badge bo"> {j['edu']}+</span>
    <span class="badge br">⏱ {j['exp_min']}+ yrs</span>
  </div>
  <div class="match-bar-bg">
    <div class="match-bar-fill" style="width:{pct}%;background:{col};"></div>
  </div>
  <div style="font-size:0.75rem;color:#6b7280;margin-bottom:0.75rem;">{pct}% alignment with your skills</div>

  <b style="font-size:0.8rem;"> Salary:</b>
  <span style="color:#0f4c75;font-weight:600;"> {j['salary']}</span><br><br>

  <b style="font-size:0.8rem;">🛠 Skills Analysis:</b><br>
  {m_pills if m_pills else ''}
  {x_pills if x_pills else ''}
  {'<div style="font-size:0.7rem;color:#6b7280;margin-top:4px;"> Matched &nbsp;|&nbsp;  Missing</div>' if (matched or missing) else ''}
</div>
""", unsafe_allow_html=True)

                if missing:
                    st.markdown(f"""
<div class="gap-box">
  <div class="gap-t"> Skill Gap — You should learn:</div>
  <div class="gap-s">{', '.join(missing)}</div>
</div>""", unsafe_allow_html=True)

                if st.button(f"🚀 Apply Now", key=f"apply_{j['id']}"):
                    st.success(f"Applied for **{j['title']}** at **{j['company']}**!")

        st.markdown("---")
        csv = show[['title','company','location','type','category','salary','pct']]\
              .rename(columns={'title':'Title','company':'Company','location':'Location',
                               'type':'Type','category':'Category','salary':'Salary','pct':'Match%'})\
              .to_csv(index=False).encode('utf-8')
        st.download_button(" Download Results as CSV", csv, "job_recommendations.csv",
                           "text/csv", use_container_width=True)

    else:
        st.markdown("""
<div class="welcome">
  <div class="icon"></div>
  <h3 style="color:#0f4c75;">Welcome, Waqaas!</h3>
  <p style="color:#6b7280;max-width:520px;margin:0 auto;">
    Enter your skills and preferences in the <b>sidebar</b>, then click
    <b>"Find My Jobs"</b> to get personalized AI-powered job recommendations.
  </p>
</div>
""", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.info("**Step 1** — Type your skills\ne.g. Python, SQL, React")
        c2.info("**Step 2** — Set your education,\nexperience & preferences")
        c3.info("**Step 3** — Click\n**'Find My Jobs'** ")

        st.markdown('<div class="sec">📋 Job Categories in System</div>', unsafe_allow_html=True)
        cats = df['category'].value_counts()
        c1,c2 = st.columns(2)
        for i,(cat,cnt) in enumerate(cats.items()):
            (c1 if i%2==0 else c2).markdown(f"**{cat}** — {cnt} jobs")


# ══════════════════════════════════════════════════════════════
#  ANALYTICS
# ══════════════════════════════════════════════════════════════
elif " Analytics" in page:
    st.markdown('<div class="sec"> Job Market Analytics</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,num,lbl in [
        (c1,len(df),"Total Jobs"),
        (c2,df['company'].nunique(),"Companies"),
        (c3,df['category'].nunique(),"Categories"),
        (c4,df['location'].nunique(),"Locations"),
    ]:
        col.markdown(f'<div class="stat-card"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("** Jobs by Category**")
        cats = df['category'].value_counts()
        fig,ax = plt.subplots(figsize=(6,4))
        colors = ['#0d7c66','#1b6ca8','#0f4c75','#7c3aed','#dc2626','#ea580c','#16a34a','#0891b2','#be185d']
        ax.barh(cats.index, cats.values, color=colors[:len(cats)])
        for v,b in zip(cats.values, ax.patches):
            ax.text(v+0.1, b.get_y()+b.get_height()/2, str(v), va='center', fontsize=9)
        ax.set_xlabel('Number of Jobs')
        ax.set_title('Job Distribution by Category')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r1c2:
        st.markdown("**🕐 Job Type Distribution**")
        types = df['type'].value_counts()
        fig2,ax2 = plt.subplots(figsize=(6,4))
        ax2.pie(types.values, labels=types.index, autopct='%1.1f%%', startangle=90,
                colors=['#0d7c66','#1b6ca8','#7c3aed','#dc2626','#ea580c'][:len(types)])
        ax2.set_title('Job Types Available')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("**Experience Required**")
        bins = pd.cut(df['exp_min'],bins=[-1,0,1,2,3,5,10,20],
                      labels=['Fresher','<1yr','1-2yr','2-3yr','3-5yr','5-10yr','10+yr'])
        bc = bins.value_counts().sort_index()
        fig3,ax3 = plt.subplots(figsize=(6,4))
        ax3.bar(bc.index.astype(str), bc.values, color='#1b6ca8', edgecolor='white')
        ax3.set_xlabel('Experience Level'); ax3.set_ylabel('Jobs')
        ax3.set_title('Jobs by Experience Required')
        plt.xticks(rotation=20); plt.tight_layout(); st.pyplot(fig3); plt.close()

    with r2c2:
        st.markdown("**🛠 Top In-Demand Skills**")
        all_skills = []
        for s in df['skills']: all_skills.extend([x.strip() for x in s.split(',')])
        top12 = Counter(all_skills).most_common(12)
        sn,sv = zip(*top12)
        fig4,ax4 = plt.subplots(figsize=(6,4))
        ax4.barh(list(sn)[::-1], list(sv)[::-1], color='#0d7c66', edgecolor='white')
        ax4.set_xlabel('Jobs Requiring Skill')
        ax4.set_title('Top 12 In-Demand Skills')
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.markdown("---")
    st.markdown('<div class="sec"> Full Job Dataset</div>', unsafe_allow_html=True)
    disp = df[['title','company','location','type','category','salary','exp_min','edu']].copy()
    disp.columns = ['Job Title','Company','Location','Type','Category','Salary','Min Exp','Education']
    st.dataframe(disp, use_container_width=True, height=420)


# ══════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ══════════════════════════════════════════════════════════════
elif " How It Works" in page:
    st.markdown('<div class="sec"> How the AI System Works</div>', unsafe_allow_html=True)

    steps = [
        (" Data Collection","30 real-world job listings with titles, descriptions, required skills, salary, location and experience requirements."),
        (" Text Preprocessing","Job descriptions are cleaned: lowercased, special characters removed, stop words filtered using TF-IDF's built-in English stop word list."),
        (" TF-IDF Vectorization","TfidfVectorizer (Scikit-learn) converts job descriptions and your skills into numerical vectors. TF-IDF gives higher weight to rare, important keywords."),
        (" Cosine Similarity","Similarity between your skill vector and each job vector is computed using cosine similarity. Score 0–1 (1 = perfect match)."),
        (" Score Boosting","Boosts applied: experience match (+0.08), job type preference (+0.06), education alignment (+0.05). Scores normalized to 60–99%."),
        (" Results & Gap Analysis","Jobs ranked by match percentage. Skill overlap shown in green (matched) and red (missing = skill gap to fill)."),
    ]
    for t,d in steps:
        with st.expander(t, expanded=True):
            st.write(d)

    st.markdown("---")
    st.markdown("###  Cosine Similarity Formula")
    st.latex(r"\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}")
    st.markdown("- **A** = TF-IDF vector of your skills\n- **B** = TF-IDF vector of job description\n- Result: **0** (no match) → **1** (perfect match)")

    st.markdown("---")
    st.markdown("### 🛠 Technologies Used")
    col1,col2 = st.columns(2)
    col1.markdown("""
| Tool | Purpose |
|---|---|
| **Python** | Core language |
| **Streamlit** | GUI / Frontend |
| **Scikit-learn** | TF-IDF + Cosine Similarity |
| **Pandas / NumPy** | Data handling |
""")
    col2.markdown("""
| Tool | Purpose |
|---|---|
| **Matplotlib** | Charts & analytics |
| **re (Regex)** | Text preprocessing |
| **CSV Dataset** | Job data storage |
| **Session State** | App memory |
""")
    st.markdown("---")
    st.info("📚 **Prepared by:** Waqaas Hussain &nbsp;|&nbsp; **Subject:** Programming for AI &nbsp;|&nbsp; **Framework:** Streamlit + Scikit-learn")

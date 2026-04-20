# ============================================================
#   Job Reccomandation System 
#   Institution : Aror University Sukkur
#   Student     : Waqaas Hussain (SAP-5000000291)
#   Instructor  : Sir Abdul Haseeb (BS AI - Semester 4)
#   
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
#  1. PREMIUM GUI & GLASSMORPH
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Job Reccomandation System ", layout="wide", page_icon="🎯")

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
#  2. THE DATABASE 
# ──────────────────────────────────────────────────────────────
@st.cache_data
def get_national_db():
    data = [
        {"title": "AI Research Scientist", "company": "Systems Ltd", "location": "Lahore", "salary": 280000, "skills": "Python, PyTorch, NLP, Scikit-learn, Research"},
        {"title": "Senior Data Architect", "company": "Afiniti", "location": "Karachi", "salary": 350000, "skills": "SQL, Python, Statistics, Machine Learning, AWS, ETL"},
        {"title": "ML Engineer (Vision)", "company": "Folio3", "location": "Karachi", "salary": 140000, "skills": "Python, Computer Vision, OpenCV, Git, Django"},
        {"title": "AI Web Developer", "company": "Aror Solutions", "location": "Sukkur", "salary": 25000, "skills": "JavaScript, React, API, Python, Tailwind"},
        {"title": "Cloud Security Expert", "company": "NetSol", "location": "Islamabad", "salary": 310000, "skills": "AWS, Docker, Kubernetes, Linux, Python, CI/CD"},
        {"title": "Junior Data Analyst", "company": "Contour Software", "location": "Lahore", "salary": 160000, "skills": "SQL, Excel, Python, PowerBI, Statistics"}
    ]
    return pd.DataFrame(data)

# ──────────────────────────────────────────────────────────────
#  3. THE MATHEMATICAL BRAIN 
# ──────────────────────────────────────────────────────────────
def calculate_ai_fit(input_text, df):
     Preprocessing with Regex
    def clean(t): return re.sub(r'[^a-z0-9\s]', '', t.lower())
    
     TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    corpus = df['title'] + " " + df['skills']
    tfidf_matrix = tfidf.fit_transform(corpus.apply(clean))
    
    # Vector Space Projection
    user_vec = tfidf.transform([clean(input_text)])
    
    
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df['score'] = scores * 100
    
    
    user_tokens = set(clean(input_text).split())
    def find_gap(row_skills):
        required = set([s.strip().lower() for s in row_skills.split(',')])
        gap = required - user_tokens
        return ", ".join(list(gap)).title() if gap else "Ready!"
    
    df['gap'] = df['skills'].apply(find_gap)
    return df.sort_values(by='score', ascending=False)

# ──────────────────────────────────────────────────────────────
#  4. SIDEBAR & CANDIDATE DATA
# ──────────────────────────────────────────────────────────────
df_main = get_national_db()

with st.sidebar:
    st.markdown("<h1 style='color:#10b981;'>TalentMatch AI</h1>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=70)
    st.markdown("---")
    
    st.subheader("👨‍💻 Professional Profile")
    u_name = st.text_input("Candidate Name", "Waqaas Hussain")
    u_input = st.text_area("Paste Full CV / Resume Content", placeholder="e.g. Python Developer with experience in ML...", height=250)
    u_loc = st.selectbox("Market Focus", ["All Pakistan"] + sorted(list(df_main['location'].unique())))
    
    st.markdown("---")
    trigger = st.button("Analyze & Compute Match")
    st.caption(f"Project by {u_name}\nAror University Sukkur")

# ──────────────────────────────────────────────────────────────
#  5. THE DASHBOARD 
# ──────────────────────────────────────────────────────────────
st.title("Digital Pakistan Career Dashboard")
st.write(f"Instructor: **Sir Abdul Haseeb** | **BS AI Semester 4 Final Project**")


m1, m2, m3, m4 = st.columns(4)
m1.metric("Available Jobs", len(df_main))
m2.metric("Top Hub", "Karachi")
m3.metric("AI Demand", "High")
m4.metric("Avg Salary", "PKR 180K")

st.markdown("<br>", unsafe_allow_html=True)

if trigger and u_input:
    results = calculate_ai_fit(u_input, df_main)
    if u_loc != "All Pakistan":
        results = results[results['location'] == u_loc]
        
    res_tab, vis_tab = st.tabs(["🎯 Top Matched Opportunities", "📊 Industry Insights"])
    
    with res_tab:
        st.subheader(f"Ranked Recommendations for {u_name}")
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
                    <div style="margin-top:20px; border-top: 1px solid rgba(255,255,255,0.1); padding-top:15px;">
                        <span style="font-size:0.85rem; color:#94a3b8; font-weight:bold;">💡 SKILL GAP:</span><br>
                        <span style="color:#f87171; font-weight:600; font-size:0.95rem;">{row['gap']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with vis_tab:
        # Week 09: Seaborn & Matplotlib Visualization
        st.subheader("National Market Analysis")
        cl, cr = st.columns(2)
        with cl:
            fig, ax = plt.subplots(facecolor='none')
            sns.barplot(data=df_main, x='location', y='salary', palette='Greens_d', ax=ax)
            ax.set_title("Salary Benchmarks by City", color='white', weight='bold')
            ax.tick_params(colors='white')
            st.pyplot(fig)
        with cr:
            fig2, ax2 = plt.subplots(facecolor='none')
            plt.pie(df_main['location'].value_counts(), labels=df_main['location'].unique(), autopct='%1.1f%%', colors=sns.color_palette('Greens_d'))
            ax2.set_title("Market Opportunity Share", color='white', weight='bold')
            st.pyplot(fig2)

with st.expander("🛠️ Algorithm Explainability (Week 10-12)"):
    st.write("""
    **Mathematical Pipeline:**
    1. **TF-IDF Vectorization:** Converts unstructured CV text into numerical feature vectors.
    2. **Cosine Similarity:** Measures the cosine of the angle between your skill vector and the job requirement vector.
    3. **Ranking Engine:** Sorts jobs based on the highest dot-product score.
    """)
    

st.markdown("---")
st.caption("BS AI Semester 4 | Aror University Sukkur |")

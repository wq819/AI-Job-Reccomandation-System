import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# 1. SETTINGS & UI
st.set_page_config(page_title="ML Job Matcher", layout="wide")
st.title("🤖 ML-Based Job Recommendation")
st.write("Using **K-Nearest Neighbors (KNN)** instead of NLP Vectorization")

# 2. PREPARING ML DATASET
# ML algorithms numbers par kaam karte hain, isliye hum skills ko "Binary Features" banayenge
@st.cache_data
def get_ml_data():
    jobs = [
        {"title": "AI Engineer", "skills_list": ["Python", "ML", "SQL"]},
        {"title": "Web Developer", "skills_list": ["HTML", "CSS", "JS"]},
        {"title": "Data Scientist", "skills_list": ["Python", "R", "ML"]},
        {"title": "DevOps Engineer", "skills_list": ["Docker", "Linux", "Python"]},
        {"title": "Frontend Expert", "skills_list": ["React", "HTML", "CSS"]}
    ]
    df = pd.DataFrame(jobs)
    
    # Encoding: Skills ko 0 aur 1 mein convert karna (ML preprocessing)
    mlb = MultiLabelBinarizer()
    skill_matrix = mlb.fit_transform(df['skills_list'])
    feature_names = mlb.classes_
    
    return df, skill_matrix, mlb, feature_names

df, skill_matrix, mlb, feature_names = get_ml_data()

# 3. TRAINING THE ML MODEL (KNN)
# Hum "Brute" algorithm use kar rahe hain jo distance calculate karta hai
model = NearestNeighbors(n_neighbors=2, algorithm='brute', metric='cosine')
model.fit(skill_matrix)

# 4. SIDEBAR INPUTS
st.sidebar.header("User Profile (ML Features)")
selected_skills = st.sidebar.multiselect("Select Your Skills:", feature_names)

if st.sidebar.button("Predict Best Job"):
    if selected_skills:
        # User input ko machine readable format mein convert karna
        user_input_encoded = mlb.transform([selected_skills])
        
        # ML Prediction (Finding nearest neighbors)
        distances, indices = model.kneighbors(user_input_encoded)
        
        st.subheader("ML Prediction Results:")
        for i in range(len(indices[0])):
            job_idx = indices[0][i]
            # Distance ko similarity percentage mein convert karna
            match_score = round((1 - distances[0][i]) * 100, 2)
            
            with st.container():
                st.markdown(f"""
                <div style="border:1px solid #4CAF50; padding:15px; border-radius:10px; margin-bottom:10px;">
                    <h4>🎯 Job Match: {df.iloc[job_idx]['title']}</h4>
                    <p><b>Confidence Score:</b> {match_score}%</p>
                    <p><b>Required Skills:</b> {", ".join(df.iloc[job_idx]['skills_list'])}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please select at least one skill.")

# 5. ML LOGIC EXPLANATION
with st.expander("How this ML Model works?"):
    st.write("""
    - **One-Hot Encoding**: Humne text ko 0 aur 1 ki matrix mein badal diya.
    - **KNN (K-Nearest Neighbors)**: Ye algorithm user ke data point aur jobs ke data points ke darmiyan 'Euclidean' ya 'Cosine' distance check karta hai.
    - **Classification**: Jo jobs user ke sab se kareeb (nearest) hoti hain, wahi recommend hoti hain.
    """)

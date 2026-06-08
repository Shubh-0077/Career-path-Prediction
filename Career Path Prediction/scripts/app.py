# =========================
# IMPORT LIBRARIES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from career_guidance import career_guidance
# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Career Path Prediction System",
    page_icon="🎯",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("🎯 Career Predictor")

    st.markdown("---")

    st.subheader("📋 Project Overview")

    st.write(
        """
        This application predicts the most suitable career path
        based on a student's skills, interests, certifications,
        workshops, and preferences.
        """
    )

    st.markdown("---")

    st.subheader("🤖 Model Information")

    st.write("Model: Random Forest Classifier")
    st.write("Dataset Size: 6,901 Students")
    st.write("Career Paths: 12")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.write("Shubham Malkar")
    st.write("Machine Learning Intern")

    st.markdown("---")

    st.caption(
        "Built using Python, Scikit-Learn and Streamlit"
    )

# =========================
# LOAD MODEL FILES
# =========================

model = pickle.load(
    open("models/career_prediction_model.pkl", "rb")
)

feature_names = pickle.load(
    open("models/feature_names.pkl", "rb")
)

label_encoder = pickle.load(
    open("models/label_encoder.pkl", "rb")
)

# -------------------------
# HEADER
# -------------------------

st.title("🎯 Career Path Prediction & Guidance System")

st.markdown("""
Predict the most suitable career path based on your
skills, interests, certifications, and preferences.
""")

st.divider()

# -------------------------
# STUDENT INFORMATION
# -------------------------

st.subheader("Student Information")

col1, col2 = st.columns(2)

with col1:

    logical_quotient_rating = st.slider(
        "Logical Quotient Rating",
        1, 10, 5
    )

    hackathons = st.slider(
        "Hackathons Participated",
        0, 10, 0
    )

    coding_skills_rating = st.slider(
        "Coding Skills Rating",
        1, 10, 5
    )

    public_speaking_points = st.slider(
        "Public Speaking Points",
        1, 10, 5
    )

    self_learning = st.selectbox(
        "Self Learning Capability",
        ["yes", "no"]
    )

    extra_courses = st.selectbox(
        "Extra Courses Done",
        ["yes", "no"]
    )

    certifications = st.selectbox(
        "Certification",
        [
            "information security",
            "shell programming",
            "r programming",
            "distro making",
            "machine learning",
            "full stack",
            "hadoop",
            "app development",
            "python"
        ]
    )

with col2:

    workshops = st.selectbox(
        "Workshop",
        [
            "testing",
            "database security",
            "game development",
            "data science",
            "system designing",
            "hacking",
            "cloud computing",
            "web technologies"
        ]
    )

    reading_writing = st.selectbox(
        "Reading & Writing Skills",
        ["poor", "medium", "excellent"]
    )

    memory_capability = st.selectbox(
        "Memory Capability",
        ["poor", "medium", "excellent"]
    )

    interested_subject = st.selectbox(
        "Interested Subject",
        [
            "programming",
            "Management",
            "data engineering",
            "networks",
            "Software Engineering",
            "cloud computing",
            "parallel computing",
            "IOT",
            "Computer Architecture",
            "hacking"
        ]
    )

    interested_career_area = st.selectbox(
        "Interested Career Area",
        [
            "testing",
            "system developer",
            "Business process analyst",
            "security",
            "developer",
            "cloud computing"
        ]
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "BPA",
            "Cloud Services",
            "product development",
            "Testing and Maintainance Services",
            "SAaS services",
            "Web Services",
            "Finance",
            "Sales and Marketing",
            "Product based",
            "Service Based"
        ]
    )

    taken_inputs_from_seniors = st.selectbox(
        "Taken Inputs From Seniors or Elders",
        ["yes", "no"]
    )

    management_or_technical = st.selectbox(
        "Management or Technical",
        ["Management", "Technical"]
    )

    hard_or_smart_worker = st.selectbox(
        "Work Style",
        ["hard worker", "smart worker"]
    )

    worked_in_teams = st.selectbox(
        "Worked In Teams Ever?",
        ["yes", "no"]
    )

    introvert = st.selectbox(
        "Introvert",
        ["yes", "no"]
    )

    interested_type_of_books = st.selectbox(
        "Interested Type Of Books",
        [
            "Series",
            "Autobiographies",
            "Travel",
            "Guide",
            "Health",
            "Journals",
            "Anthology",
            "Dictionaries",
            "Prayer books",
            "Art",
            "Encyclopedias",
            "Religion-Spirituality",
            "Action and Adventure",
            "Comics",
            "Horror",
            "Satire",
            "Self help",
            "History",
            "Cookbooks",
            "Math",
            "Biographies",
            "Drama",
            "Diaries",
            "Science fiction",
            "Poetry",
            "Romance",
            "Science",
            "Trilogy",
            "Fantasy",
            "Childrens",
            "Mystery"
        ]
    )

predict_btn = st.button(
    "Predict Career Path",
    use_container_width=True
)

# =========================
# CREATE INPUT DATAFRAME
# =========================

if predict_btn:

    input_data = pd.DataFrame({
        "logical_quotient_rating": [logical_quotient_rating],
        "hackathons": [hackathons],
        "coding_skills_rating": [coding_skills_rating],
        "public_speaking_points": [public_speaking_points],
        "self_learning_capability": [self_learning],
        "extra_courses": [extra_courses],
        "certifications": [certifications],
        "workshops": [workshops],
        "reading_writing_skills": [reading_writing],
        "memory_capability_score": [memory_capability],
        "interested_subjects": [interested_subject],
        "interested_career_area": [interested_career_area],
        "company_type": [company_type],
        "taken_inputs_from_seniors": [taken_inputs_from_seniors],
        "interested_type_of_books": [interested_type_of_books],
        "management_or_technical": [management_or_technical],
        "hard_or_smart_worker": [hard_or_smart_worker],
        "worked_in_teams": [worked_in_teams],
        "introvert": [introvert]
    })

    
# =========================
# ENCODE USER INPUT
# =========================

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Add missing columns
    for col in feature_names:

        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder columns
    input_encoded = input_encoded[
        feature_names
    ]

    

    # =========================
    # MODEL PREDICTION
    # =========================

    prediction = model.predict(
        input_encoded
    )

    predicted_role = label_encoder.inverse_transform(
        prediction
    )[0]

    
    # =========================
    # CONFIDENCE SCORE
    # =========================

    probabilities = model.predict_proba(
        input_encoded
    )

    confidence = np.max(
        probabilities
    ) * 100

    # =========================
    # PREDICTION RESULT
    # =========================

    st.divider()

    st.subheader("🎯 Career Prediction Result")

    st.success(
        f"Recommended Career Path: {predicted_role}"
    )

    st.metric(
        label="📊 Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    st.warning(
        """
        ⚠️ This recommendation is based on the available dataset and machine learning model.
        It should be used as career guidance rather than a definitive career decision.
        """
    )

    # =========================
    # CAREER GUIDANCE
    # =========================

    if predicted_role in career_guidance:

        guidance = career_guidance[predicted_role]

        st.subheader("🛠 Recommended Skills")

        for skill in guidance["skills"]:
            st.markdown(f"✅ **{skill}**")

        st.subheader("📜 Recommended Certifications")

        for cert in guidance["certifications"]:
            st.markdown(f"📜 **{cert}**")

        st.subheader("🚀 Learning Roadmap")

        for i, step in enumerate(
            guidance["roadmap"],
            start=1
        ):
            st.write(f"{i}. {step}")
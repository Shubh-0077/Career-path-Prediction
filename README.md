# 🎯 Career Path Prediction & Guidance System

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green)
![Dataset](https://img.shields.io/badge/Dataset-6%2C901%20Students-lightgrey)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Executive Summary

This project builds an end-to-end machine learning system that predicts the most suitable career path for students in the technology sector based on their skills, interests, certifications, workshops, and personality attributes. The system covers 12 career categories ranging from Software Engineering to UX Design and is deployed as a Streamlit web application with an integrated career guidance engine.

---
## 📑 Table of Contents

* [📌 Executive Summary](#-executive-summary)
* [💼 Business Problem](#-business-problem)
* [🎯 Project Objectives](#-project-objectives)
* [📊 Dataset Overview](#-dataset-overview)
* [🧹 Data Validation & Cleaning Summary](#-data-validation--cleaning-summary)
* [🔍 Exploratory Data Analysis Summary](#-exploratory-data-analysis-summary)
* [⚙️ Methodology](#️-methodology)
* [🔧 Feature Engineering](#-feature-engineering)
* [📐 Feature Encoding](#-feature-encoding)
* [🤖 Machine Learning Models](#-machine-learning-models)
* [🚀 Streamlit Application](#-streamlit-application)
* [📈 Business Metrics](#-business-metrics)
* [🔑 Key Findings](#-key-findings)
* [💡 Recommendations](#-recommendations)
* [🛠 Technologies Used](#-technologies-used)
* [🗂 Repository Structure](#-repository-structure)
* [📦 Installation Instructions](#-installation-instructions)
* [🔁 How to Reproduce the Project](#-how-to-reproduce-the-project)
* [👨‍💻 Author](#-author)



## 💼 Business Problem

Students entering the technology industry face an overwhelming number of career options, with limited structured guidance on which path aligns best with their individual profile. Poor career choices lead to mismatched skill development, low job satisfaction, and reduced productivity. This project addresses that gap by providing a data-driven, personalized career recommendation system that considers a student's full profile — not just academic grades.

---

## 🎯 Project Objectives

- Build a multi-class classification model to predict suitable career paths from 12 technology roles
- Perform thorough exploratory data analysis to uncover patterns between student attributes and career outcomes
- Evaluate multiple machine learning models (Decision Tree, Random Forest, XGBoost, CatBoost) and select the best performer
- Deploy the model as an interactive Streamlit web application
- Provide role-specific guidance including recommended skills, certifications, and a learning roadmap

---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| Source | PS2 Dataset (Career Path Survey) |
| Total Records | 6,901 students |
| Features | 19 (4 numerical, 15 categorical) |
| Target Variable | Suggested Job Role (12 classes) |
| Duplicates | 0 |
| Missing Values | 0 |

**Numerical Features:** Logical Quotient Rating, Hackathons Participated, Coding Skills Rating, Public Speaking Points

**Categorical Features:** Self-Learning Capability, Extra Courses Done, Certifications, Workshops, Reading/Writing Skills, Memory Capability, Interested Subjects, Interested Career Area, Company Type Preference, Senior Input Taken, Book Interests, Management vs Technical Preference, Work Style, Teamwork Experience, Introvert/Extrovert

**Target Classes (12 Career Roles):**
| Career Role | Count |
|---|---|
| Network Security Engineer | 630 |
| Software Engineer | 590 |
| UX Designer | 589 |
| Software Developer | 587 |
| Database Developer | 581 |
| Software QA / Testing | 571 |
| Web Developer | 570 |
| CRM Technical Developer | 567 |
| Technical Support | 565 |
| Systems Security Administrator | 562 |
| Applications Developer | 551 |
| Mobile Applications Developer | 538 |

---

## 🧹 Data Validation & Cleaning Summary

- **Column Renaming:** All 20 columns renamed to standardized snake_case format for Python compatibility and consistency
- **No missing values** detected across all 19 feature columns and the target column
- **No duplicate records** found in the 6,901-row dataset
- **Categorical consistency** verified across all object-type columns — no unexpected string variants, extra spaces, or inconsistent casing detected
- **Numerical range verification** confirmed: all 4 numeric features fell within expected value ranges (ratings 1–10, hackathons 0–10)
- Cleaned data exported as `career_cleaned.csv` for use in EDA and modeling phases

---

## 🔍 Exploratory Data Analysis Summary

### Target Variable Distribution
The dataset is well-balanced across all 12 career roles. Class frequencies range from 538 (Mobile Applications Developer) to 630 (Network Security Engineer), a difference of just 92 records. This near-perfect balance eliminates class imbalance as a concern and removes the need for techniques like SMOTE.

### Numerical Feature Analysis
Histograms and boxplots confirmed that all 4 numerical features are broadly distributed across their value ranges, with no extreme outliers. A correlation heatmap revealed very low correlation among numerical features (close to 0 across all pairs), indicating each feature captures independent information and multicollinearity is not a concern.

### Career Insights from Categorical Features

| Feature | Key Insight |
|---|---|
| Certifications | R Programming → Database/QA roles; Information Security → Systems Security/Web Dev |
| Workshops | Hacking workshops correlate with Network Security / Systems Security careers |
| Interested Subjects | Data Engineering → Database Developer; Networks → Database/Systems Security |
| Career Area Interest | Security interest spreads across Software Engineering, Technical Support, and Network Security |
| Management vs Technical | Both orientations appear across all 12 career categories, neither is dominant |
| Introvert/Extrovert | Distributed across all roles — personality alone does not determine career outcome |

### Overall EDA Conclusion
No single feature strongly determines career outcome. Career recommendations arise from complex interactions between multiple attributes. This supports the use of ensemble models (Random Forest, XGBoost, CatBoost) that can capture non-linear feature relationships.

---

## ⚙️ Methodology

```
Raw Data → Exploratory Data Analysis → Data Cleaning →
Feature Encoding (One-Hot Encoding) → Model Training →
Model Evaluation → Feature Selection Experiments →
Feature Engineering Experiments → Cross-Validation →
Final Model Selection → Model Saving → Streamlit Deployment
```

---

## 🔧 Feature Engineering

Several composite features were created and evaluated:

| Engineered Feature | Components |
|---|---|
| Technical Strength Score | Logical Quotient + Coding Skills + Hackathons |
| Learning Score | Self-Learning Capability + Extra Courses |
| Teamwork Score | Worked in Teams + Took Senior Input |
| Communication Score | Public Speaking Points |
| Professional Development Score | Technical Strength + Learning Score |

**Result:** Engineered features did not improve model accuracy (7.53% vs baseline 9.56%). The original feature representation was retained for the final model.

---

## 📐 Feature Encoding

All categorical variables were encoded using **One-Hot Encoding** (`pd.get_dummies`, `drop_first=True`), expanding the feature space from 19 to 83 columns. The target variable was encoded using `LabelEncoder` from Scikit-Learn.

---

## 🤖 Machine Learning Models

Four models were trained and evaluated:

| Model | Test Accuracy | CV Mean Accuracy (5-Fold) |
|---|---|---|
| **Random Forest** | **9.56%** | **8.13%** |
| Decision Tree | 8.47% | 8.43% |
| CatBoost | 8.83% | — |
| XGBoost | 8.04% | 7.96% |

**Final Model: Random Forest Classifier**
- `n_estimators = 200`, `random_state = 42`
- Trained on full 83-feature one-hot encoded dataset
- Selected for highest test accuracy and deployment simplicity

> **⚠️ Important Note on Accuracy:** The ~9% test accuracy reflects an extraordinarily difficult classification problem — 12 nearly equally distributed classes where no single feature strongly separates them. Random assignment would achieve ~8.3% (1/12). The model shows marginal but consistent improvement over chance. This is a known characteristic of this dataset (PS2), and the primary value of the project lies in the end-to-end ML pipeline and deployed application rather than raw accuracy.

---

## 🚀 Streamlit Application

The Streamlit app (`app.py`) provides:

- **19-input prediction form** covering all numerical sliders and categorical dropdowns
- **Real-time career prediction** using the trained Random Forest model
- **Confidence score** showing model's prediction probability
- **Career guidance panel** with role-specific recommended skills, certifications, and a 5-step learning roadmap for all 12 career paths

---

## 📈 Business Metrics

| Metric | Value |
|---|---|
| Dataset Size | 6,901 students |
| Career Categories Covered | 12 |
| Model Performance vs Random Baseline | +1.26% improvement |
| Roles with Guidance Content | 12 / 12 (100%) |
| Deployment Format | Streamlit Web Application |

---

## 🔑 Key Findings

1. **The dataset is balanced** — no class imbalance treatment required, all 12 career classes within a 92-record range
2. **No single feature predicts career outcome** — career recommendations require the combination of certifications, workshops, interests, personality, and skills
3. **Numerical features are independent** — near-zero correlation among all 4 numerical features, no multicollinearity
4. **Feature engineering reduced performance** — composite scores added noise rather than signal in this dataset
5. **Ensemble models edge out simpler models** — Random Forest (9.56%) beats Decision Tree (8.47%), but the gap is narrow
6. **Cross-validation confirms consistency** — 5-fold CV showed stable performance across all folds (8.1–8.4% range)

---

## 💡 Recommendations

**For career counselors:** This system works best as a starting-point conversation tool, not a final decision engine. The low accuracy reflects the genuine complexity of career matching — use predictions to initiate discussions about student strengths, not conclude them.

**For dataset improvement:** Collecting more discriminative features (GPA, project portfolio, internship experience, specific technical assessments) would significantly improve model performance.

**For model improvement:** Consider ordinal encoding for `reading_writing_skills` and `memory_capability_score` (poor/medium/excellent) rather than one-hot encoding, which may preserve their inherent ordering better.

---

## 🛠 Technologies Used

| Category | Technology |
|---|---|
| Language | Python 3.13 |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn, XGBoost, CatBoost |
| Model Persistence | Pickle |
| Web Application | Streamlit |
| Development Environment | JupyterLab |

---

## 🗂 Repository Structure

```
career-path-prediction/
│
├── data/
│   ├── PS2_Dataset.csv              # Raw dataset (6,901 students)
│   └── career_cleaned.csv           # Cleaned dataset after preprocessing
│
├── notebooks/
│   └── career_path_prediction.ipynb # Full analysis notebook (7 phases)
│
├── models/
│   ├── career_prediction_model.pkl  # Trained Random Forest model
│   ├── feature_names.pkl            # Feature column names for inference
│   └── label_encoder.pkl            # Label encoder for target decoding
│
├── scripts/
│   ├── app.py                       # Streamlit application
│   └── career_guidance.py           # Career guidance dictionary (12 roles)
│
├── images/
│   └── (EDA charts, app screenshots)
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

**Folder Notes:**
- `data/`: Keep raw data as-is; never overwrite the original. The cleaned CSV is the modeling input.
- `models/`: All three .pkl files must be present for the app to run. Load from here, not the project root.
- `notebooks/`: The notebook is self-contained and runs phases 1–7 sequentially.
- `scripts/`: `career_guidance.py` is imported by `app.py` — both must be in the same directory.

---

## 📦 Installation Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/career-path-prediction.git
cd career-path-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run scripts/app.py
```

---

## 🔁 How to Reproduce the Project

1. Open `notebooks/career_path_prediction.ipynb` in JupyterLab
2. Run Phase 1 (data loading and exploration)
3. Run Phase 2 (data cleaning → generates `career_cleaned.csv`)
4. Run Phase 3 (EDA visualizations)
5. Run Phase 4 (model training — Decision Tree, Random Forest, XGBoost, CatBoost)
6. Run Phase 5A–5E (feature selection, engineering, SMOTE evaluation, cross-validation)
7. Run Phase 7 (final model saving → generates the 3 `.pkl` files)
8. Launch the app with `streamlit run scripts/app.py`

---

## 👨‍💻 Author

**Shubham Malkar** 
| Machine Learning Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](http://www.linkedin.com/in/shubhammalkar)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Shubh-0077)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-orange)](https://shubhammalkar.framer.website/)

---

*Built using Python · Scikit-Learn · Streamlit*

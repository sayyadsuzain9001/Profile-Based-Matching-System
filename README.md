# 🤝 Profile-Based Matching System

> Intelligent Hybrid Recommendation System using NLP, TF-IDF, Cosine Similarity, MBTI Compatibility, Location Matching and Adaptive Feedback Learning.

## 👨‍💻 Student

**Suzain Mudabbir Sayyad**  
Artificial Intelligence & Machine Learning Student  
Domain: Machine Learning

---

## 📌 Project Overview

This project implements a profile-based recommendation system that calculates compatibility between users using a combination of unstructured profile text, MBTI personality compatibility and location matching.

The system follows a hybrid recommendation approach. Textual information from user profiles is processed using NLP and represented using TF-IDF vectors. Cosine Similarity is then used to measure textual similarity. MBTI and location compatibility are calculated separately and combined with text similarity to produce an overall compatibility score.

The project also includes an adaptive feedback component. Accept/Reject feedback is used with Logistic Regression to learn updated matching weights and generate adaptive recommendations.

## 🎯 Problem Statement

The system is designed to:

1. Parse unstructured profile information using NLP.
2. Quantify similarity between user profiles.
3. Combine text similarity, MBTI compatibility and location compatibility.
4. Generate the Top 5 recommended profiles.
5. Learn from Accept/Reject feedback and update matching weights.

## 🔄 Complete Workflow

```text
User Profiles
      ↓
Data Generation / Loading
      ↓
Data Inspection
      ↓
Text Preprocessing
      ↓
Combined Profile Text
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
MBTI Compatibility
      ↓
Location Compatibility
      ↓
Hybrid Compatibility Score
      ↓
Top 5 Recommendations
      ↓
Accept / Reject Feedback
      ↓
Logistic Regression
      ↓
Adaptive Matching Weights
      ↓
Adaptive Recommendations
      ↓
Streamlit Application
```

## 🧠 Matching Components

### 1. NLP Text Similarity

The project combines:

- About Me
- Professional Summary
- Skills
- Interests

into a single profile text representation.

The text is cleaned before being converted into numerical features.

### 2. TF-IDF

TF-IDF converts profile text into numerical vectors and gives importance to words based on their frequency within profiles and across the collection.

### 3. Cosine Similarity

Cosine Similarity measures the similarity between the TF-IDF representations of two profiles.

### 4. MBTI Compatibility

The system uses MBTI personality types as an additional compatibility component.

The notebook represents MBTI compatibility on a 0–1 scale:

- `1.0` → high compatibility
- `0.5` → moderate compatibility
- `0.0` → low compatibility

### 5. Location Compatibility

The current project implementation assigns:

- `1.0` when locations match
- `0.0` when locations differ

### 6. Hybrid Compatibility Score

The initial notebook weights are:

```text
Text Similarity       = 60%
MBTI Compatibility    = 25%
Location Compatibility = 15%
```

The initial score is:

```text
Compatibility Score =
    (Text Similarity × 0.60)
  + (MBTI Compatibility × 0.25)
  + (Location Compatibility × 0.15)
```

After feedback learning, the adaptive weights are used instead of only the initial fixed weights.

## 🔁 Adaptive Learning

The notebook simulates **500 Accept/Reject feedback interactions**.

Feedback is represented as:

```text
1 → Accept
0 → Reject
```

Logistic Regression uses:

- Text Similarity
- MBTI Score
- Location Score

as input features and Feedback as the target.

The learned coefficients are normalized into updated weights. These weights are then used by the adaptive recommendation function.

## 👥 Dataset

The notebook generates **200 synthetic user profiles**.

Each profile includes:

- User ID
- Name
- About Me
- Professional Summary
- Skills
- Professional Goal
- Interests
- MBTI Personality Type
- Location

The project uses synthetic profile data generated within the notebook.

## 🏆 Recommendation Output

For a selected user, the system:

1. Compares the user with other profiles.
2. Calculates compatibility components.
3. Calculates the final compatibility score.
4. Sorts candidates by score.
5. Returns the Top 5 profiles.

The recommendation output includes:

- User ID
- Name
- Professional Goal
- MBTI
- Location
- Compatibility Score

## 🌐 Streamlit Application

The Streamlit application provides:

- Dashboard
- Existing User recommendations
- New Profile recommendations
- Upload Profiles
- Model Analysis
- Top 5 recommendations
- Compatibility scores
- Accept / Reject feedback
- Downloadable uploaded-profile recommendations
- Adaptive-weight visualization

## 🏗️ Application Architecture

```text
                 Saved Matching Components
                           │
                           ▼
                    Streamlit App
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   Existing User      New Profile     Upload Profiles
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  Profile Processing
                           │
                           ▼
                 Similarity + MBTI
                    + Location
                           │
                           ▼
                  Hybrid Score
                           │
                           ▼
                    Top 5 Matches
                           │
                           ▼
                  User Feedback
```

## 📁 Repository Structure

```text
Profile_Based_Matching_System/
│
├── notebook/
│   └── MajorProject1.ipynb
│
├── data/
│   └── README.md
│
├── deployment/
│   ├── app.py
│   ├── matching_system.pkl
│   ├── final_profiles.csv
│   ├── final_feedback_data.csv
│   ├── adaptive_weights.csv
│   ├── top_5_recommendations_U001.csv
│   └── requirements.txt
│
├── results/
│   └── README.md
│
├── documentation/
│   ├── Profile_Based_Matching_System_Professional_Report.docx
│   ├── Profile_Based_Matching_System_1_Page_Summary.pdf
│   ├── README_Notebook.md
│   ├── README_Streamlit.md
│   ├── Matching_System_Deep_Guide.md
│   ├── EXPLANATION_GUIDE.md
│   ├── PROJECT_DIRECTORY.md
│   └── RUNTIME_ARTIFACTS.md
│
├── .gitignore
└── README.md
```

> The saved runtime artifacts listed under `deployment/` are produced by the notebook. They are not fabricated by this documentation package. See `documentation/RUNTIME_ARTIFACTS.md`.

## 📦 Requirements

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
nltk
```

## ▶️ Run the Streamlit Application

Open a terminal in the deployment folder:

```powershell
cd deployment
pip install -r requirements.txt
python -m streamlit run app.py
```

The application expects the saved project artifacts to be present in the same deployment folder.

## 🧪 Notebook Requirements

The notebook uses NLTK for stopword removal and lemmatization.

Run:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

before running the text-preprocessing section if the NLTK resources are not already installed.

## 🧪 Testing Checklist

### Notebook

- [x] User profile generation
- [x] Dataset inspection
- [x] Missing-value check
- [x] Text preprocessing
- [x] TF-IDF
- [x] Cosine Similarity
- [x] MBTI compatibility
- [x] Location compatibility
- [x] Hybrid score
- [x] Top 5 recommendations
- [x] Feedback generation
- [x] Logistic Regression adaptive model
- [x] Adaptive weights
- [x] Adaptive recommendations
- [x] Final result saving
- [x] Streamlit artifact preparation

### Streamlit

- [x] Dashboard
- [x] Existing User
- [x] New Profile
- [x] Upload Profiles
- [x] Model Analysis
- [x] Top 5 recommendations
- [x] Compatibility scores
- [x] Accept / Reject feedback
- [x] Download recommendations

## 📌 Project Status

```text
Profile Generation          ✅
NLP Preprocessing           ✅
TF-IDF                      ✅
Cosine Similarity           ✅
MBTI Compatibility          ✅
Location Compatibility      ✅
Hybrid Recommendation        ✅
Feedback Learning            ✅
Adaptive Weights             ✅
Top 5 Recommendations        ✅
Streamlit Application        ✅
Documentation                ✅
```

## 👨‍💻 Author

**Suzain Mudabbir Sayyad**  
Artificial Intelligence & Machine Learning Student

## ⭐ Keywords

`Python` `Machine Learning` `NLP` `TF-IDF` `Cosine Similarity` `Recommendation System` `MBTI` `Adaptive Learning` `Logistic Regression` `Streamlit`
#   P r o f i l e - B a s e d - M a t c h i n g - S y s t e m  
 
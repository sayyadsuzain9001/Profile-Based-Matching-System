# 🌐 Profile-Based Matching System — Streamlit Documentation

## Overview

The Streamlit application provides an interactive interface for the Profile-Based Matching Algorithm.

The application loads the saved matching components and profile data and provides several user-facing sections.

## Required Files

The application expects these files:

```text
matching_system.pkl
final_profiles.csv
adaptive_weights.csv
```

The application also contains feedback-saving logic that can create:

```text
streamlit_feedback.csv
```

during use.

## Saved Matching Components

The saved `matching_system.pkl` contains:

```text
df
tfidf_vectorizer
similarity_matrix
adaptive_text_weight
adaptive_mbti_weight
adaptive_location_weight
mbti_compatibility
```

## Navigation

### 🏠 Dashboard

Shows:

- Total profiles
- Hybrid matching method
- Top 5 recommendation setting
- Adaptive learning status
- Current adaptive weights
- Matching workflow

### 👤 Existing User

A user can be selected from the available User IDs.

The application calculates compatibility against other users and displays recommended profiles.

### 🆕 New Profile

A new profile can be entered through the application.

The application uses the profile information to calculate recommendations against the saved profile dataset.

### 📂 Upload Profiles

Multiple profiles can be uploaded as CSV.

The required columns are:

```text
Name
About_Me
Professional_Summary
Skills
Professional_Goal
Interests
MBTI
Location
```

The application generates recommendations for uploaded profiles and provides a downloadable CSV.

### 📊 Model Analysis

The application displays:

- number of profiles
- TF-IDF feature count
- similarity matrix shape
- adaptive weights
- matching formula
- dataset preview

## Matching Formula

```text
Compatibility Score =
    (Text Similarity × Adaptive Text Weight)
  + (MBTI Compatibility × Adaptive MBTI Weight)
  + (Location Compatibility × Adaptive Location Weight)
```

## Feedback

The application records Accept/Reject feedback with:

- timestamp
- user ID
- match ID
- feedback

The feedback is saved to:

```text
streamlit_feedback.csv
```

## Run

```powershell
cd deployment
pip install -r requirements.txt
python -m streamlit run app.py
```

## Troubleshooting

### `matching_system.pkl was not found`

Run the final notebook sections that create the saved Streamlit components and place the resulting file in the deployment folder.

### `final_profiles.csv was not found`

Run the notebook result-saving section and place the generated profile file beside `app.py`.

### Uploaded CSV rejected

Check that the CSV contains all required columns listed above.

## Application Flow

```text
Profile
  ↓
NLP / Profile Processing
  ↓
Text Similarity
  ↓
MBTI Compatibility
  ↓
Location Compatibility
  ↓
Adaptive Hybrid Score
  ↓
Top 5 Recommendations
  ↓
Accept / Reject Feedback
```

# 📘 Profile-Based Matching System — Notebook Documentation

## 1. Overview

`MajorProject1.ipynb` contains the complete development workflow for the Profile-Based Matching Algorithm.

The notebook moves from synthetic profile generation to NLP-based similarity, hybrid compatibility scoring, adaptive feedback learning and preparation of Streamlit deployment artifacts.

## 2. Notebook Workflow

```text
Generate Profiles
      ↓
Inspect Dataset
      ↓
Check Missing Values
      ↓
Text Preprocessing
      ↓
Combine Profile Text
      ↓
TF-IDF
      ↓
Cosine Similarity
      ↓
MBTI Rules
      ↓
Location Score
      ↓
Hybrid Score
      ↓
Top 5 Matches
      ↓
Feedback
      ↓
Logistic Regression
      ↓
Adaptive Weights
      ↓
Adaptive Recommendations
      ↓
Save Results
      ↓
Save Streamlit Components
```

## 3. Generated Dataset

The notebook generates 200 user profiles containing:

- User ID
- Name
- About Me
- Professional Summary
- Skills
- Professional Goal
- Interests
- MBTI
- Location

## 4. Text Preprocessing

The notebook uses NLTK to:

- convert text to lowercase
- remove punctuation/special characters
- split text into words
- remove English stopwords
- apply lemmatization

The cleaned text fields are combined into `Combined_Text`.

## 5. TF-IDF

`TfidfVectorizer` converts the combined profile text into a numerical matrix.

## 6. Cosine Similarity

`cosine_similarity()` creates a pairwise similarity matrix between profiles.

The resulting values represent textual similarity between users.

## 7. MBTI Compatibility

MBTI compatibility is represented on a 0–1 scale.

The notebook uses a compatibility dictionary to provide compatibility values between personality types.

## 8. Location Compatibility

The notebook assigns:

```text
Same location    → 1.0
Different        → 0.0
```

## 9. Hybrid Score

Initial weights:

```text
Text Similarity        0.60
MBTI Compatibility     0.25
Location Compatibility 0.15
```

Formula:

```text
Score =
(Text Similarity × 0.60)
+
(MBTI Compatibility × 0.25)
+
(Location Compatibility × 0.15)
```

## 10. Top 5 Recommendations

The selected profile is compared against the remaining profiles. The system sorts the calculated scores and returns the five highest-scoring profiles.

## 11. Feedback Learning

The notebook creates 500 simulated feedback interactions.

```text
Accept = 1
Reject = 0
```

The feedback table contains matching component values and the feedback target.

## 12. Logistic Regression

The adaptive model uses:

```text
Text_Similarity
MBTI_Score
Location_Score
```

to predict:

```text
Feedback
```

The learned coefficients are converted into normalized adaptive weights.

## 13. Adaptive Recommendation

The adaptive recommendation function uses the learned weights rather than relying only on the original fixed weights.

## 14. Saved Results

The notebook is designed to save:

```text
final_profiles.csv
final_feedback_data.csv
top_5_recommendations_U001.csv
adaptive_weights.csv
matching_system.pkl
```

## 15. Streamlit Components

`matching_system.pkl` stores:

```text
df
tfidf_vectorizer
similarity_matrix
adaptive_text_weight
adaptive_mbti_weight
adaptive_location_weight
mbti_compatibility
```

These components are loaded by the Streamlit application.

## 16. Notebook Section Map

| Section | Main Work |
|---|---|
| 1 | Libraries |
| 2 | Profile generation |
| 4–6 | Dataset generation and inspection |
| 7 | Missing values |
| 8–10 | Text preprocessing |
| 11 | TF-IDF |
| 12 | Cosine Similarity |
| 13 | MBTI rules |
| 14 | Location score |
| 15–17 | Hybrid matching and Top 5 |
| 18–23 | Feedback and adaptive learning |
| 24–29 | Feedback analysis |
| 30–34 | Final recommendation functions |
| 35–36 | Adaptive weights and saving results |
| 37–38 | Streamlit preparation |

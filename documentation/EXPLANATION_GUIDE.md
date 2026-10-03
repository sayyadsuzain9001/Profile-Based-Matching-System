# 📖 Profile-Based Matching System — Explanation Guide

## One-Line Explanation

This project is an intelligent hybrid recommendation system that compares user profiles using NLP-based text similarity, MBTI compatibility and location matching, and then adapts the matching weights using user feedback.

## Main Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- Cosine Similarity
- Logistic Regression
- Streamlit
- Joblib

## Easy Explanation

### Step 1 — Profiles

The notebook generates 200 user profiles containing personal, professional, personality and location information.

### Step 2 — Text

Important text fields are cleaned and combined.

### Step 3 — TF-IDF

The text is converted into numerical vectors.

### Step 4 — Similarity

Cosine Similarity measures how similar two profiles are.

### Step 5 — MBTI

The system calculates personality compatibility.

### Step 6 — Location

The system checks whether the two users are from the same location.

### Step 7 — Hybrid Score

The three components are combined into one compatibility score.

### Step 8 — Recommendation

The system sorts users and returns the Top 5 matches.

### Step 9 — Feedback

Users can Accept or Reject recommendations.

### Step 10 — Adaptive Learning

Logistic Regression learns from feedback and updates the importance of the matching components.

## Viva Answer

**Q: What is the main idea of your project?**

The main idea is to build a profile-based recommendation system that does not depend only on fixed demographic filters. It understands profile text using NLP, calculates text similarity using TF-IDF and Cosine Similarity, adds MBTI and location compatibility, and combines them into a hybrid score. The system also uses Accept/Reject feedback with Logistic Regression to update the matching weights.

**Q: Why use TF-IDF?**

TF-IDF converts profile text into numerical vectors while giving more importance to informative words. This allows the system to calculate textual similarity between profiles.

**Q: Why use a hybrid approach?**

Text similarity alone does not capture personality or location. Combining text, MBTI and location gives the system multiple sources of compatibility information.

**Q: How does adaptive learning work?**

The system collects Accept/Reject feedback, trains Logistic Regression using the matching components, normalizes the learned coefficients and uses the resulting weights for adaptive recommendations.

**Q: What does the Streamlit application do?**

It provides an interactive interface for existing users, new profiles and uploaded profiles, displays Top 5 recommendations and compatibility scores, shows model analysis and records feedback.

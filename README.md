# 🤝 Profile-Based Matching System

### Intelligent Hybrid Recommendation System using NLP, TF-IDF, Cosine Similarity, MBTI Compatibility, Location Matching and Adaptive Feedback Learning

<p align="center">
  <b>Machine Learning Major Project 1</b><br>
  Artificial Intelligence & Machine Learning
</p>

---

## 👨‍🎓 Student Information

| Field | Details |
|---|---|
| **Student** | Suzain Mudabbir Sayyad |
| **University** | Sanjay Ghodawat University, Kolhapur |
| **Domain** | Artificial Intelligence & Machine Learning |
| **Project Type** | Machine Learning Major Project |
| **System Type** | Hybrid Recommendation System |

---

## 📌 Project Overview

The **Profile-Based Matching System** is an intelligent recommendation system that calculates compatibility between user profiles using a combination of:

- 🧠 Natural Language Processing (NLP)
- 🔢 TF-IDF vectorization
- 📐 Cosine similarity
- 🧩 MBTI personality compatibility
- 📍 Location compatibility
- 🔄 Adaptive feedback learning
- 🤖 Logistic Regression
- 🌐 Streamlit

Instead of relying only on fixed filters or keyword matching, the system combines profile text similarity with personality and location compatibility to generate a **Top-5 recommendation list**.

The system also includes an adaptive feedback component where **Accept/Reject** interactions are used to learn updated matching weights.

---

## 🎯 Problem Statement

The system is designed to:

1. Parse unstructured profile information using NLP.
2. Quantify semantic similarity between user profiles.
3. Combine text similarity, MBTI compatibility and location compatibility.
4. Generate the Top-5 compatible profiles.
5. Learn from Accept/Reject feedback.
6. Update matching weights for adaptive recommendations.

---

## 🎯 Project Objectives

- Convert unstructured profile information into machine-readable features.
- Measure semantic similarity between users.
- Include personality compatibility in the recommendation process.
- Include location as an additional compatibility factor.
- Produce ranked Top-5 recommendations.
- Introduce feedback-based adaptive learning.
- Provide an interactive Streamlit interface for the matching system.

---

## 🔄 Complete System Workflow

```text
User Profiles
     │
     ▼
Data Generation / Loading
     │
     ▼
Text Preprocessing
     │
     ├── Lowercase
     ├── Remove Punctuation
     ├── Remove Stop Words
     └── Lemmatization
     │
     ▼
Combined Profile Text
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Cosine Similarity
     │
     ├───────────────┐
     ▼               ▼
MBTI Compatibility  Location Compatibility
     │               │
     └───────┬───────┘
             ▼
      Hybrid Compatibility Score
             │
             ▼
       Top-5 Recommendations
             │
             ▼
       Accept / Reject Feedback
             │
             ▼
      Logistic Regression
             │
             ▼
     Adaptive Matching Weights
             │
             ▼
    Adaptive Recommendations
             │
             ▼
      Streamlit Application
```

---

## 📊 Dataset

The notebook generates **200 synthetic user profiles**.

Each profile contains:

| Field | Description |
|---|---|
| `User_ID` | Unique user identifier |
| `Name` | User name |
| `About_Me` | Personal profile description |
| `Professional_Summary` | Professional background |
| `Skills` | Technical/professional skills |
| `Professional_Goal` | Career or professional objective |
| `Interests` | User interests |
| `MBTI` | MBTI personality type |
| `Location` | User location |

> **Note:** The project uses synthetic profiles rather than real personal-user data.

---

## 🧹 1. Text Preprocessing

The system preprocesses profile text before calculating similarity.

The preprocessing pipeline includes:

```text
Raw Profile Text
       ↓
Convert to Lowercase
       ↓
Remove Punctuation / Special Characters
       ↓
Tokenization
       ↓
Remove English Stop Words
       ↓
Lemmatization
       ↓
Clean Profile Text
```

The implementation uses **NLTK** for stop-word removal and lemmatization.

---

## 🔢 2. TF-IDF Vectorization

The cleaned profile text is converted into numerical vectors using **TF-IDF**.

```python
TfidfVectorizer()
```

TF-IDF gives importance to words based on their frequency within profiles and across the complete collection.

The result is a numerical representation of each profile that can be compared mathematically.

---

## 📐 3. Cosine Similarity

Cosine similarity is used to measure semantic similarity between profile vectors.

```python
cosine_similarity(tfidf_matrix)
```

A higher similarity value indicates that the profile texts are more similar according to the TF-IDF representation.

The resulting similarity matrix is used as the main text-based compatibility component.

---

## 🧩 4. MBTI Compatibility

The system adds personality compatibility using MBTI types.

Compatibility is represented on a **0–1 scale**:

| Score | Meaning |
|---:|---|
| `1.0` | High compatibility |
| `0.5` | Moderate compatibility |
| `0.0` | Low compatibility |

The compatibility values are based on the predefined MBTI compatibility rules implemented in the notebook.

---

## 📍 5. Location Compatibility

Location is included as an additional compatibility signal.

```text
Same Location      → 1.0
Different Location → 0.0
```

This allows the recommendation score to consider geographical compatibility alongside profile similarity and personality.

---

## ⚙️ 6. Hybrid Compatibility Score

The initial matching model uses three components.

| Component | Initial Weight |
|---|---:|
| **Text Similarity** | **60%** |
| **MBTI Compatibility** | **25%** |
| **Location Compatibility** | **15%** |

### Formula

```text
Compatibility Score =
    (Text Similarity × 0.60)
  + (MBTI Compatibility × 0.25)
  + (Location Compatibility × 0.15)
```

The candidates are sorted by the final compatibility score and the **Top 5** profiles are returned.

---

## 🔄 7. Adaptive Feedback Learning

The system extends the initial fixed-weight model with adaptive learning.

Feedback is represented as:

```text
Accept → 1
Reject → 0
```

The notebook uses simulated feedback interactions and trains a **Logistic Regression** model using:

```text
Text Similarity
MBTI Score
Location Score
```

as input features.

The learned coefficients are normalized to produce updated adaptive weights.

### Adaptive Process

```text
Initial Recommendations
          ↓
     User Feedback
     Accept / Reject
          ↓
   Feedback Dataset
          ↓
   Logistic Regression
          ↓
   Learned Coefficients
          ↓
   Adaptive Weights
          ↓
Adaptive Recommendations
```

---

## 🏆 8. Top-5 Recommendation Process

For each selected user:

1. Calculate text similarity with other profiles.
2. Calculate MBTI compatibility.
3. Calculate location compatibility.
4. Apply the adaptive matching weights.
5. Calculate the final compatibility score.
6. Sort profiles by score.
7. Exclude the selected user.
8. Return the Top 5 profiles.

The recommendation output includes:

```text
User ID
Name
Professional Goal
MBTI
Location
Text Similarity
MBTI Compatibility
Location Compatibility
Compatibility Score
```

---

## 🌐 9. Streamlit Application

The project includes an interactive Streamlit application.

### Application Navigation

| Page | Purpose |
|---|---|
| 🏠 **Dashboard** | System overview and adaptive weights |
| 👤 **Existing User** | Find compatible users from existing profiles |
| 🆕 **New Profile** | Create a new profile and find recommendations |
| 📂 **Upload Profiles** | Upload profile data and generate recommendations |
| 📊 **Model Analysis** | Inspect model components, weights and dataset |

### Dashboard

The dashboard displays:

- Total profiles
- Matching method
- Top-5 recommendation setting
- Adaptive learning status
- Current adaptive weights
- System workflow

### Model Analysis

The Model Analysis page provides:

- Number of profiles
- TF-IDF feature count
- Similarity matrix dimensions
- Adaptive matching weights
- Matching formula
- Dataset preview

---

## 📁 Repository Structure

```text
Profile-Based-Matching-System/
│
├── 📓 notebook/
│   └── MajorProject1.ipynb
│
├── 🌐 deployment/
│   ├── app.py
│   └── requirements.txt
│
├── 📊 data/
│   └── README.md
│
├── 📈 results/
│   └── README.md
│
├── 📚 documentation/
│   ├── EXPLANATION_GUIDE.md
│   ├── Matching_System_Deep_Guide.md
│   ├── PROJECT_DIRECTORY.md
│   ├── README_Notebook.md
│   ├── README_Streamlit.md
│   ├── RUNTIME_ARTIFACTS.md
│   ├── Profile_Based_Matching_System_Professional_Report.docx
│   └── Profile_Based_Matching_System_1_Page_Summary.pdf
│
├── .gitignore
├── GENERATED_PACKAGE_MANIFEST.md
└── README.md
```

---

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

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sayyadsuzain9001/Profile-Based-Matching-System.git
```

Move into the project:

```bash
cd Profile-Based-Matching-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r deployment/requirements.txt
```

---

## 🧠 NLTK Setup

The notebook requires the NLTK resources used for stop-word removal and lemmatization.

Run:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
```

Then run the notebook from beginning to end to generate the runtime model/data artifacts.

---

## ▶️ Run the Streamlit Application

After generating the required runtime files, place the model/data artifacts beside `app.py` as required by the application.

Then:

```bash
cd deployment
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 💾 Runtime Artifacts

The notebook generates the following important files:

```text
profiles.csv
final_profiles.csv
final_feedback_data.csv
top_5_recommendations_U001.csv
adaptive_weights.csv
matching_system.pkl
```

The Streamlit application requires the saved matching components and profile/weight data described in the project documentation.

> The repository documentation does not fabricate runtime artifacts. They should be generated by executing the notebook in an environment where the required NLTK resources are available.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core implementation |
| **Pandas** | Data handling |
| **NumPy** | Numerical operations |
| **NLTK** | Text preprocessing |
| **Scikit-learn** | TF-IDF, cosine similarity and Logistic Regression |
| **Joblib** | Saving/loading model components |
| **Matplotlib** | Visualization |
| **Streamlit** | Interactive web application |

---

## 📚 Documentation

| Document | Purpose |
|---|---|
| `README.md` | Main GitHub project documentation |
| `README_Notebook.md` | Notebook workflow and section guide |
| `README_Streamlit.md` | Streamlit application guide |
| `Matching_System_Deep_Guide.md` | Detailed technical explanation |
| `EXPLANATION_GUIDE.md` | Simple explanation and viva preparation |
| `PROJECT_DIRECTORY.md` | Repository structure |
| `RUNTIME_ARTIFACTS.md` | Runtime files and generation instructions |
| `Profile_Based_Matching_System_Professional_Report.docx` | Professional project report |
| `Profile_Based_Matching_System_1_Page_Summary.pdf` | One-page project summary |

---

## ⚠️ Limitations

- The profile dataset is synthetic.
- The feedback interactions used by the notebook are simulated.
- MBTI compatibility depends on the predefined compatibility rules.
- Location compatibility currently uses a simple same/different comparison.
- The adaptive model depends on the available feedback data.
- TF-IDF represents lexical/weighted word similarity and does not provide deep contextual understanding like transformer-based language models.
- Runtime model artifacts must be generated before the Streamlit application can operate.

---

## 🔮 Future Scope

Possible extensions include:

- Transformer-based sentence embeddings.
- Semantic similarity using modern language models.
- More detailed geographical distance calculations.
- Real user feedback collection.
- Online/incremental recommendation learning.
- User preference profiles.
- Explainable recommendation reasons.
- Larger real-world datasets.
- Recommendation history and analytics.
- Cloud deployment and database integration.

---

## 🧪 Project Status

| Component | Status |
|---|:---:|
| Profile generation | ✅ |
| Data inspection | ✅ |
| Text preprocessing | ✅ |
| TF-IDF vectorization | ✅ |
| Cosine similarity | ✅ |
| MBTI compatibility | ✅ |
| Location compatibility | ✅ |
| Hybrid scoring | ✅ |
| Top-5 recommendations | ✅ |
| Feedback collection | ✅ |
| Adaptive learning | ✅ |
| Adaptive recommendations | ✅ |
| Streamlit application | ✅ |
| Project documentation | ✅ |

---

## 📌 Conclusion

The **Profile-Based Matching System** demonstrates an end-to-end hybrid recommendation workflow that combines NLP-based profile similarity with MBTI personality compatibility and location matching.

The system goes beyond a fixed scoring formula by introducing **feedback-driven adaptive learning** using Logistic Regression. The resulting recommendation workflow can generate ranked Top-5 matches and expose the matching process through an interactive Streamlit application.

---

## 👨‍💻 Author

### Suzain Mudabbir Sayyad

**Artificial Intelligence & Machine Learning Student**  
Sanjay Ghodawat University, Kolhapur

---

## 🔑 Keywords

`Machine Learning` `Recommendation System` `NLP` `TF-IDF` `Cosine Similarity` `MBTI` `Hybrid Recommendation` `Adaptive Learning` `Logistic Regression` `Python` `Streamlit`

---

⭐ **Profile-Based Matching System — Intelligent Hybrid Recommendation using NLP and Adaptive Feedback**

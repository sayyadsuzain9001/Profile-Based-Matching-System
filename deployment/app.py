import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Profile-Based Matching System",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #111827,
        #1e293b
    );
    color: white;
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-size: 2.6rem;
    margin-bottom: 0.5rem;
}

.hero p {
    font-size: 1.05rem;
    color: #d1d5db;
}

.profile-card {
    padding: 1.3rem;
    border-radius: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
}

.match-card {
    padding: 1.4rem;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.score {
    font-size: 2rem;
    font-weight: 700;
}

.small-label {
    font-size: 0.8rem;
    color: #64748b;
}

.feature-box {
    padding: 1rem;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FILE PATHS
# ============================================================

MODEL_FILE = "matching_system.pkl"
PROFILE_FILE = "final_profiles.csv"
FEEDBACK_FILE = "final_feedback_data.csv"
WEIGHTS_FILE = "adaptive_weights.csv"


# ============================================================
# 4. LOAD SAVED MODEL COMPONENTS
# ============================================================

@st.cache_resource
def load_matching_system():

    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


@st.cache_data
def load_profiles():

    if not os.path.exists(PROFILE_FILE):
        return pd.DataFrame()

    return pd.read_csv(PROFILE_FILE)


@st.cache_data
def load_weights():

    if not os.path.exists(WEIGHTS_FILE):
        return pd.DataFrame()

    return pd.read_csv(WEIGHTS_FILE)


system = load_matching_system()
df = load_profiles()
weights_df = load_weights()


# ============================================================
# 5. CHECK REQUIRED FILES
# ============================================================

if system is None:

    st.error(
        "matching_system.pkl was not found. "
        "Please run the final notebook cell that creates "
        "matching_system.pkl first."
    )

    st.stop()


if df.empty:

    st.error(
        "final_profiles.csv was not found or is empty."
    )

    st.stop()


# ============================================================
# 6. LOAD MODEL COMPONENTS
# ============================================================

similarity_matrix = system["similarity_matrix"]
tfidf_vectorizer = system["tfidf_vectorizer"]

adaptive_text_weight = system["adaptive_text_weight"]
adaptive_mbti_weight = system["adaptive_mbti_weight"]
adaptive_location_weight = system["adaptive_location_weight"]

mbti_compatibility = system["mbti_compatibility"]


# ============================================================
# 7. TEXT PREPROCESSING
# ============================================================

@st.cache_data
def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    words = text.split()

    stop_words = {
        "a", "an", "the", "and", "or", "but",
        "is", "are", "was", "were", "am",
        "i", "me", "my", "we", "our",
        "to", "of", "in", "on", "for",
        "with", "this", "that", "it",
        "as", "be", "have", "has",
        "from", "at", "by", "about"
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# 8. MBTI SCORING
# ============================================================

def calculate_mbti_score(mbti_1, mbti_2):

    if mbti_1 == mbti_2:
        return 0.5

    if mbti_2 in mbti_compatibility.get(
        mbti_1,
        []
    ):
        return 1.0

    return 0.0


# ============================================================
# 9. LOCATION SCORING
# ============================================================

def calculate_location_score(
    location_1,
    location_2
):

    if str(location_1).strip().lower() == \
       str(location_2).strip().lower():

        return 1.0

    return 0.0


# ============================================================
# 10. EXISTING USER COMPATIBILITY
# ============================================================

def calculate_existing_score(
    user_index,
    match_index
):

    text_score = similarity_matrix[
        user_index,
        match_index
    ]

    mbti_score = calculate_mbti_score(
        df.loc[user_index, "MBTI"],
        df.loc[match_index, "MBTI"]
    )

    location_score = calculate_location_score(
        df.loc[user_index, "Location"],
        df.loc[match_index, "Location"]
    )

    final_score = (
        text_score * adaptive_text_weight
        +
        mbti_score * adaptive_mbti_weight
        +
        location_score * adaptive_location_weight
    )

    return {
        "Text Similarity": text_score,
        "MBTI Compatibility": mbti_score,
        "Location Compatibility": location_score,
        "Compatibility Score": final_score * 100
    }


# ============================================================
# 11. GET TOP MATCHES FOR EXISTING USER
# ============================================================

def get_top_matches(
    user_index,
    top_n=5
):

    matches = []

    for i in range(len(df)):

        if i == user_index:
            continue

        scores = calculate_existing_score(
            user_index,
            i
        )

        matches.append({
            "User_ID": df.loc[i, "User_ID"],
            "Name": df.loc[i, "Name"],
            "Professional_Goal":
                df.loc[i, "Professional_Goal"],
            "MBTI":
                df.loc[i, "MBTI"],
            "Location":
                df.loc[i, "Location"],
            "Text Similarity":
                round(
                    scores["Text Similarity"] * 100,
                    2
                ),
            "MBTI Compatibility":
                round(
                    scores["MBTI Compatibility"] * 100,
                    2
                ),
            "Location Compatibility":
                round(
                    scores["Location Compatibility"] * 100,
                    2
                ),
            "Compatibility Score":
                round(
                    scores["Compatibility Score"],
                    2
                )
        })

    result = pd.DataFrame(matches)

    result = result.sort_values(
        by="Compatibility Score",
        ascending=False
    )

    return result.head(top_n).reset_index(drop=True)


# ============================================================
# 12. PREPARE NEW PROFILE TEXT
# ============================================================

def create_profile_text(profile):

    about = preprocess_text(
        profile.get("About_Me", "")
    )

    professional = preprocess_text(
        profile.get(
            "Professional_Summary",
            ""
        )
    )

    skills = preprocess_text(
        profile.get("Skills", "")
    )

    interests = preprocess_text(
        profile.get("Interests", "")
    )

    return (
        about + " "
        + professional + " "
        + skills + " "
        + interests
    ).strip()


# ============================================================
# 13. MATCH NEW PROFILE AGAINST EXISTING USERS
# ============================================================

def recommend_for_new_profile(
    new_profile,
    top_n=5
):

    new_text = create_profile_text(
        new_profile
    )

    new_vector = tfidf_vectorizer.transform(
        [new_text]
    )

    existing_vectors = tfidf_vectorizer.transform(
        df["Combined_Text"]
    )

    from sklearn.metrics.pairwise import cosine_similarity

    text_similarities = cosine_similarity(
        new_vector,
        existing_vectors
    )[0]

    results = []

    for i in range(len(df)):

        text_score = text_similarities[i]

        mbti_score = calculate_mbti_score(
            new_profile["MBTI"],
            df.loc[i, "MBTI"]
        )

        location_score = calculate_location_score(
            new_profile["Location"],
            df.loc[i, "Location"]
        )

        final_score = (
            text_score * adaptive_text_weight
            +
            mbti_score * adaptive_mbti_weight
            +
            location_score *
            adaptive_location_weight
        )

        results.append({
            "User_ID":
                df.loc[i, "User_ID"],

            "Name":
                df.loc[i, "Name"],

            "Professional_Goal":
                df.loc[i, "Professional_Goal"],

            "MBTI":
                df.loc[i, "MBTI"],

            "Location":
                df.loc[i, "Location"],

            "Text Similarity":
                round(text_score * 100, 2),

            "MBTI Compatibility":
                round(mbti_score * 100, 2),

            "Location Compatibility":
                round(location_score * 100, 2),

            "Compatibility Score":
                round(final_score * 100, 2)
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        by="Compatibility Score",
        ascending=False
    )

    return result.head(top_n).reset_index(drop=True)


# ============================================================
# 14. SAVE FEEDBACK
# ============================================================

def save_feedback(
    user_id,
    match_id,
    feedback
):

    record = pd.DataFrame([
        {
            "Timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "User_ID":
                user_id,

            "Match_ID":
                match_id,

            "Feedback":
                feedback
        }
    ])

    feedback_file = "streamlit_feedback.csv"

    if os.path.exists(feedback_file):

        record.to_csv(
            feedback_file,
            mode="a",
            header=False,
            index=False
        )

    else:

        record.to_csv(
            feedback_file,
            index=False
        )


# ============================================================
# 15. SESSION STATE
# ============================================================

if "recommendations" not in st.session_state:

    st.session_state.recommendations = None


if "new_profile_recommendations" not in st.session_state:

    st.session_state.new_profile_recommendations = None


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🤝 Matching System"
    )

    st.markdown(
        "### Navigation"
    )

    page = st.radio(
        "Select Section",
        [
            "🏠 Dashboard",
            "👤 Existing User",
            "🆕 New Profile",
            "📂 Upload Profiles",
            "📊 Model Analysis"
        ]
    )

    st.divider()

    st.markdown(
        "### System Information"
    )

    st.write(
        f"**Profiles:** {len(df)}"
    )

    st.write(
        "**Algorithm:** Hybrid Recommendation"
    )

    st.write(
        "**NLP:** TF-IDF + Cosine Similarity"
    )

    st.write(
        "**Personality:** MBTI"
    )

    st.write(
        "**Adaptive:** Enabled"
    )


# ============================================================
# 17. HERO HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🤝 Profile-Based Matching System</h1>

<p>
Intelligent Hybrid Recommendation System using
NLP, MBTI Personality Compatibility and Location Matching.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 18. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.subheader(
        "System Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Profiles",
            len(df)
        )

    with col2:

        st.metric(
            "Matching Method",
            "Hybrid"
        )

    with col3:

        st.metric(
            "Recommendations",
            "Top 5"
        )

    with col4:

        st.metric(
            "Adaptive Learning",
            "Enabled"
        )

    st.divider()

    st.subheader(
        "How the System Works"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.markdown(
            "### 1️⃣\nProfile"
        )

        st.caption(
            "User information is collected."
        )

    with c2:

        st.markdown(
            "### 2️⃣\nNLP"
        )

        st.caption(
            "Profile text is converted into numerical representations."
        )

    with c3:

        st.markdown(
            "### 3️⃣\nSimilarity"
        )

        st.caption(
            "Cosine similarity measures profile similarity."
        )

    with c4:

        st.markdown(
            "### 4️⃣\nHybrid Score"
        )

        st.caption(
            "NLP, MBTI and location are combined."
        )

    with c5:

        st.markdown(
            "### 5️⃣\nRecommendation"
        )

        st.caption(
            "Top 5 compatible profiles are displayed."
        )

    st.divider()

    st.subheader(
        "Current Adaptive Weights"
    )

    w1, w2, w3 = st.columns(3)

    with w1:

        st.metric(
            "Text Similarity",
            f"{adaptive_text_weight * 100:.2f}%"
        )

    with w2:

        st.metric(
            "MBTI Compatibility",
            f"{adaptive_mbti_weight * 100:.2f}%"
        )

    with w3:

        st.metric(
            "Location Compatibility",
            f"{adaptive_location_weight * 100:.2f}%"
        )

    if not weights_df.empty:

        st.bar_chart(
            weights_df.set_index(
                "Component"
            )["Adaptive_Weight"] * 100
        )


# ============================================================
# 19. EXISTING USER PAGE
# ============================================================

elif page == "👤 Existing User":

    st.subheader(
        "Find Compatible Users"
    )

    user_ids = df["User_ID"].tolist()

    selected_user = st.selectbox(
        "Select User",
        user_ids
    )

    user_rows = df.index[
        df["User_ID"] == selected_user
    ].tolist()

    if user_rows:

        user_index = user_rows[0]

        user = df.loc[user_index]

        st.markdown(
            "### Selected Profile"
        )

        p1, p2, p3 = st.columns(3)

        with p1:

            st.markdown(
                f"""
                **User ID:** {user["User_ID"]}

                **Name:** {user["Name"]}

                **Professional Goal:** {user["Professional_Goal"]}
                """
            )

        with p2:

            st.markdown(
                f"""
                **MBTI:** {user["MBTI"]}

                **Location:** {user["Location"]}

                **Skills:** {user["Skills"]}
                """
            )

        with p3:

            st.markdown(
                f"""
                **About Me**

                {user["About_Me"]}
                """
            )

        st.divider()

        if st.button(
            "🔎 Find Top 5 Matches",
            type="primary",
            use_container_width=True
        ):

            st.session_state.recommendations = \
                get_top_matches(
                    user_index,
                    top_n=5
                )

        recommendations = \
            st.session_state.recommendations

        if recommendations is not None:

            st.subheader(
                "🏆 Top 5 Recommended Matches"
            )

            for index, match in recommendations.iterrows():

                score = match[
                    "Compatibility Score"
                ]

                st.markdown(
                    f"""
                    <div class="match-card">

                    <h3>
                    #{index + 1}
                    &nbsp;
                    {match["Name"]}
                    </h3>

                    <p>
                    <b>User ID:</b> {match["User_ID"]}
                    </p>

                    <p>
                    <b>Professional Goal:</b>
                    {match["Professional_Goal"]}
                    </p>

                    <p>
                    <b>MBTI:</b> {match["MBTI"]}
                    &nbsp;&nbsp;&nbsp;
                    <b>Location:</b> {match["Location"]}
                    </p>

                    <div class="score">
                    {score:.2f}%
                    </div>

                    <p>
                    Overall Compatibility
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                s1, s2, s3, s4 = st.columns(4)

                with s1:

                    st.metric(
                        "Overall",
                        f"{score:.2f}%"
                    )

                with s2:

                    st.metric(
                        "Text Similarity",
                        f'{match["Text Similarity"]:.2f}%'
                    )

                with s3:

                    st.metric(
                        "MBTI",
                        f'{match["MBTI Compatibility"]:.2f}%'
                    )

                with s4:

                    st.metric(
                        "Location",
                        f'{match["Location Compatibility"]:.2f}%'
                    )

                f1, f2 = st.columns(2)

                with f1:

                    if st.button(
                        "👍 Accept",
                        key=f"accept_{selected_user}_{match['User_ID']}"
                    ):

                        save_feedback(
                            selected_user,
                            match["User_ID"],
                            1
                        )

                        st.success(
                            "Feedback recorded: Accepted"
                        )

                with f2:

                    if st.button(
                        "👎 Reject",
                        key=f"reject_{selected_user}_{match['User_ID']}"
                    ):

                        save_feedback(
                            selected_user,
                            match["User_ID"],
                            0
                        )

                        st.warning(
                            "Feedback recorded: Rejected"
                        )

            st.divider()

            csv_data = recommendations.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Recommendations",
                data=csv_data,
                file_name=f"{selected_user}_top_matches.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# 20. NEW PROFILE PAGE
# ============================================================

elif page == "🆕 New Profile":

    st.subheader(
        "Create a New Profile"
    )

    st.write(
        "Enter profile information below and the system "
        "will predict the Top 5 compatible users from "
        "the existing 200 profiles."
    )

    c1, c2 = st.columns(2)

    with c1:

        new_name = st.text_input(
            "Name",
            placeholder="Enter your name"
        )

        new_about = st.text_area(
            "About Me",
            placeholder="Tell us about yourself..."
        )

        new_professional_summary = st.text_area(
            "Professional Summary",
            placeholder="Describe your professional interests..."
        )

        new_skills = st.text_input(
            "Skills",
            placeholder="Python, Machine Learning, SQL..."
        )

    with c2:

        new_goal = st.selectbox(
            "Professional Goal",
            sorted(
                df["Professional_Goal"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        new_interests = st.text_input(
            "Interests",
            placeholder="AI, Robotics, Gaming..."
        )

        new_mbti = st.selectbox(
            "MBTI",
            sorted(
                df["MBTI"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        new_location = st.selectbox(
            "Location",
            sorted(
                df["Location"]
                .dropna()
                .unique()
                .tolist()
            )
        )

    if st.button(
        "🚀 Generate My Top 5 Matches",
        type="primary",
        use_container_width=True
    ):

        if not new_name.strip():

            st.error(
                "Please enter your name."
            )

        elif not new_about.strip():

            st.error(
                "Please enter your About Me information."
            )

        else:

            new_profile = {

                "User_ID": "NEW_USER",

                "Name": new_name,

                "About_Me": new_about,

                "Professional_Summary":
                    new_professional_summary,

                "Skills": new_skills,

                "Professional_Goal": new_goal,

                "Interests": new_interests,

                "MBTI": new_mbti,

                "Location": new_location
            }

            st.session_state.new_profile_recommendations = \
                recommend_for_new_profile(
                    new_profile,
                    top_n=5
                )

            st.success(
                "Profile processed successfully!"
            )

    new_results = \
        st.session_state.new_profile_recommendations

    if new_results is not None:

        st.divider()

        st.subheader(
            "🏆 Predicted Top 5 Matches"
        )

        for index, match in new_results.iterrows():

            st.markdown(
                f"""
                <div class="match-card">

                <h3>
                #{index + 1} {match["Name"]}
                </h3>

                <p>
                <b>User ID:</b> {match["User_ID"]}
                </p>

                <p>
                <b>Professional Goal:</b>
                {match["Professional_Goal"]}
                </p>

                <p>
                <b>MBTI:</b> {match["MBTI"]}
                &nbsp;&nbsp;&nbsp;
                <b>Location:</b> {match["Location"]}
                </p>

                <div class="score">
                {match["Compatibility Score"]:.2f}%
                </div>

                <p>
                Predicted Compatibility
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            a, b, c, d = st.columns(4)

            with a:

                st.metric(
                    "Compatibility",
                    f'{match["Compatibility Score"]:.2f}%'
                )

            with b:

                st.metric(
                    "Text Similarity",
                    f'{match["Text Similarity"]:.2f}%'
                )

            with c:

                st.metric(
                    "MBTI",
                    f'{match["MBTI Compatibility"]:.2f}%'
                )

            with d:

                st.metric(
                    "Location",
                    f'{match["Location Compatibility"]:.2f}%'
                )

        csv_data = new_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Predicted Matches",
            data=csv_data,
            file_name="new_profile_top_matches.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 21. UPLOAD PROFILE PAGE
# ============================================================

elif page == "📂 Upload Profiles":

    st.subheader(
        "Upload New Profile Data"
    )

    st.write(
        "Upload a CSV file containing new user profiles. "
        "The system will process the profiles and generate "
        "Top 5 compatible users for each uploaded profile."
    )

    st.info(
        "Required columns: Name, About_Me, "
        "Professional_Summary, Skills, "
        "Professional_Goal, Interests, MBTI, Location"
    )

    uploaded_file = st.file_uploader(
        "Upload Profile CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"{len(uploaded_df)} profile(s) uploaded successfully."
            )

            st.subheader(
                "Uploaded Data Preview"
            )

            st.dataframe(
                uploaded_df,
                use_container_width=True
            )

            required_columns = [
                "Name",
                "About_Me",
                "Professional_Summary",
                "Skills",
                "Professional_Goal",
                "Interests",
                "MBTI",
                "Location"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in uploaded_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:

                if st.button(
                    "🚀 Generate Recommendations",
                    type="primary",
                    use_container_width=True
                ):

                    all_results = []

                    progress = st.progress(0)

                    for row_index, row in uploaded_df.iterrows():

                        profile = row.to_dict()

                        recommendations = \
                            recommend_for_new_profile(
                                profile,
                                top_n=5
                            )

                        recommendations.insert(
                            0,
                            "Uploaded_Profile",
                            profile.get(
                                "Name",
                                f"Profile_{row_index + 1}"
                            )
                        )

                        all_results.append(
                            recommendations
                        )

                        progress.progress(
                            (row_index + 1)
                            / len(uploaded_df)
                        )

                    final_uploaded_results = \
                        pd.concat(
                            all_results,
                            ignore_index=True
                        )

                    st.session_state[
                        "uploaded_results"
                    ] = final_uploaded_results

                    st.success(
                        "Recommendations generated successfully!"
                    )

        except Exception as error:

            st.error(
                f"Error processing file: {error}"
            )

    if "uploaded_results" in st.session_state:

        st.divider()

        st.subheader(
            "Predicted Recommendations"
        )

        st.dataframe(
            st.session_state[
                "uploaded_results"
            ],
            use_container_width=True
        )

        download_data = \
            st.session_state[
                "uploaded_results"
            ].to_csv(
                index=False
            ).encode("utf-8")

        st.download_button(
            "⬇️ Download All Recommendations",
            data=download_data,
            file_name="uploaded_profile_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# 22. MODEL ANALYSIS PAGE
# ============================================================

elif page == "📊 Model Analysis":

    st.subheader(
        "Model & Matching Analysis"
    )

    st.write(
        "The following information describes the "
        "components used by the hybrid matching system."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Profiles",
            len(df)
        )

    with c2:

        st.metric(
            "TF-IDF Features",
            tfidf_vectorizer.get_feature_names_out().shape[0]
        )

    with c3:

        st.metric(
            "Similarity Matrix",
            f"{similarity_matrix.shape[0]} × "
            f"{similarity_matrix.shape[1]}"
        )

    st.divider()

    st.subheader(
        "Adaptive Matching Weights"
    )

    weight_table = pd.DataFrame({

        "Component": [
            "Text Similarity",
            "MBTI Compatibility",
            "Location Compatibility"
        ],

        "Weight (%)": [
            adaptive_text_weight * 100,
            adaptive_mbti_weight * 100,
            adaptive_location_weight * 100
        ]
    })

    st.dataframe(
        weight_table,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        weight_table.set_index(
            "Component"
        )
    )

    st.divider()

    st.subheader(
        "Matching Formula"
    )

    st.code(
        """
Compatibility Score =
    (Text Similarity × Adaptive Text Weight)
  + (MBTI Compatibility × Adaptive MBTI Weight)
  + (Location Compatibility × Adaptive Location Weight)
        """
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ============================================================
# 23. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    <hr>
    <b>Profile-Based Matching Algorithm</b><br>
    Intelligent Hybrid Recommendation System<br>
    NLP • TF-IDF • Cosine Similarity • MBTI • Adaptive Learning
    </div>
    """,
    unsafe_allow_html=True
)
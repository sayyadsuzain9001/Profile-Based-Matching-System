# ⚠️ Runtime Artifacts

The notebook is designed to generate the following files:

```text
final_profiles.csv
final_feedback_data.csv
top_5_recommendations_U001.csv
adaptive_weights.csv
matching_system.pkl
```

These are runtime/generated artifacts, not documentation files.

## Why they are not included in this documentation package

The current notebook requires NLTK `stopwords` and `wordnet` resources during execution. Those external NLTK resources were not available in the current offline execution environment, so the notebook could not be completed here without changing the project's original preprocessing implementation.

No fake model or fake feedback files have been created.

## Generate the artifacts

Run the notebook from start to finish in Google Colab or another environment with NLTK resources available.

Before the preprocessing section:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
```

After the notebook completes, copy:

```text
final_profiles.csv
final_feedback_data.csv
top_5_recommendations_U001.csv
adaptive_weights.csv
matching_system.pkl
```

into the appropriate `deployment/` and `results/` folders.

## Important

The Streamlit application requires at least:

```text
matching_system.pkl
final_profiles.csv
adaptive_weights.csv
```

in the same directory as `app.py`.

The application itself checks for missing required files and displays an error when they are unavailable.

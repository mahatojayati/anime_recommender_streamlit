# Anime Recommender – Streamlit Front End

A simple Streamlit app that wraps your `recommender_core.py`. If the `get_recommendations(title, top_n)`
function is available, the app uses it; otherwise it falls back to a content-based recommender based on
the title and genre in `anime.csv`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to GitHub.
2. Go to https://share.streamlit.io and create a new app.
3. Set the entry point to `streamlit_app/app.py`.
4. Click Deploy.

## Deploy with Docker

```bash
docker build -t anime-recommender .
docker run -p 8501:8501 anime-recommender
```

Then open http://localhost:8501

## Files

- `streamlit_app/app.py`: Streamlit UI (with a robust fallback recommender)
- `recommender_core.py`: Your core logic (if it exposes `get_recommendations(title, top_n)`, the app will use it)
- `anime.csv`: Dataset (must be present in project root)
- `rating.csv.zip`: Optional ratings data (only needed by your core, if used)
- `requirements.txt`: Python dependencies (includes Streamlit)
- `Dockerfile`: Containerized deployment

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")
st.title("🎌 Anime Recommender System")
st.markdown("Type an anime you like and get similar shows. Uses your core recommender if available, with a smart fallback.")

@st.cache_data
def load_anime():
    df = pd.read_csv("anime.csv")
    # Clean up columns we rely on
    for col in ["name","genre"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df

anime_df = load_anime()

# Try to import get_recommendations from user's module, else fallback
def _import_core():
    try:
        from recommender_core import get_recommendations as core_rec
        return core_rec
    except Exception as e:
        return None

core_get_recs = _import_core()

# Fallback recommender (content-based on title + genre)
def fallback_recommendations(query_title: str, top_n: int = 10):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if anime_df.empty:
        return pd.DataFrame(columns=["name","genre","type","episodes","rating","members"])

    text_series = (anime_df.get("name","") + " " + anime_df.get("genre","")).astype(str)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(text_series)

    # Find best match row for query string
    q_vec = vectorizer.transform([query_title])
    sims_to_query = cosine_similarity(q_vec, tfidf).ravel()
    # Get the index of the most similar existing title to the query
    if np.all(sims_to_query == 0):
        # If no match at all, just return top rated as a reasonable default
        top = anime_df.sort_values(by=["rating","members"], ascending=[False, False]).head(top_n)
        return top

    best_idx = int(np.argmax(sims_to_query))
    # Now compute similarity from that best match to all items (to find "similar to it")
    sims = cosine_similarity(tfidf[best_idx], tfidf).ravel()
    # Exclude itself
    sims[best_idx] = -1.0
    top_idx = sims.argsort()[::-1][:top_n]
    out = anime_df.iloc[top_idx][["name","genre","type","episodes","rating","members"]].copy()
    out.insert(0, "similarity", sims[top_idx])
    return out.reset_index(drop=True)

# --- UI ---
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10, step=1)
    show_debug = st.checkbox("Show debug info", value=False)

query = st.text_input("Enter an anime you like", placeholder="e.g., Fullmetal Alchemist: Brotherhood")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Get recommendations", type="primary"):
        if not query.strip():
            st.warning("Please enter an anime title.")
        else:
            try:
                if core_get_recs is not None:
                    # Use user's recommender_core.get_recommendations
                    results = core_get_recs(query.strip(), top_n=top_n)
                    # Accept DataFrame or list-like
                    if isinstance(results, pd.DataFrame):
                        st.success("Using core recommender ✅")
                        st.dataframe(results, use_container_width=True)
                    else:
                        st.success("Using core recommender ✅")
                        st.write(results)
                else:
                    # Fallback content-based
                    st.info("Core recommender not found or errored. Using fallback content-based recommender.")
                    results = fallback_recommendations(query.strip(), top_n=top_n)
                    st.dataframe(results, use_container_width=True)

                if show_debug:
                    st.caption("Columns available in anime.csv: " + ", ".join(anime_df.columns.astype(str).tolist()))
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(title: str, top_n: int = 10):
    # Load the dataset
    df = pd.read_csv("anime.csv")
    df.fillna("", inplace=True)

    # Combine relevant fields for better context
    text_data = (df["name"] + " " + df["genre"] + " " + df["type"]).astype(str)

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(text_data)

    # Find the index for the input title (or closest match)
    matches = df.index[df["name"] == title]
    if len(matches) > 0:
        idx = matches[0]
    else:
        query_vec = vectorizer.transform([title])
        idx = cosine_similarity(query_vec, tfidf).flatten().argmax()

    # Compute similarities
    sims = cosine_similarity(tfidf[idx], tfidf).flatten()

    # Exclude the item itself
    sims[idx] = -1

    # Get top_n similar items
    similar_idx = sims.argsort()[::-1][:top_n]

    return df.iloc[similar_idx][["name", "genre", "type", "episodes", "rating", "members"]]

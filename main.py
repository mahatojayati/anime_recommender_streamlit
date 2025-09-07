import pandas as pd
#loading data
anime_df = pd.read_csv('/Users/jayatimahato/Documents/anime_recommender/data/anime.csv')
#data checking
print(anime_df.head())
#data info
print(anime_df.info())
#checking no. of null values
print(anime_df.isnull().sum())
#remove missing values
print(anime_df.dropna())
#unique values
anime_df['genre'].value_counts()

# Handle NaN
anime_df['genre'] = anime_df['genre'].fillna('') # Handle NaN
# Example: splitting genres
anime_df['genre_list'] = anime_df['genre'].apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
# Assuming genre is now a comma-separated string from Task 2.3
  # Or you can re-join the list:
  # anime_df['content_features'] = anime_df['genre_list'].apply(lambda x: ' '.join(x))
anime_df['content_features'] = anime_df['genre'].fillna('')
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english') 
# 'english' removes common words like 'the', 'a', etc.

tfidf_matrix = tfidf_vectorizer.fit_transform(anime_df['content_features'])

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#drop duplicates
indices = pd.Series(anime_df.index, index=anime_df['name']).drop_duplicates()

def get_recommendations(title, cosine_sim, df, indices):
      if title not in indices:
          return "Anime not found in the database. Please check the spelling."

      idx = indices[title]
      sim_scores = list(enumerate(cosine_sim[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
      sim_scores = sim_scores[1:11] # Get top 10, excluding itself

      anime_indices = [i[0] for i in sim_scores]
      return df['name'].iloc[anime_indices]

# Example usage
print(get_recommendations('Pokemon', cosine_sim, anime_df, indices))  

#CLI loop for users
# Example CLI loop
while True:
    user_input = input("Enter an anime title (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
          break
    recommendations = get_recommendations(user_input, cosine_sim, anime_df, indices)
    if isinstance(recommendations, str): # Error message
          print(recommendations)
    else:
          print("\nRecommended Anime:")
          for i, rec in enumerate(recommendations):
              print(f"{i+1}. {rec}")
    print("-" * 30)
  
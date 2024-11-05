import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def load_data():
    """Load the cleaned TMDB movies dataset."""
    return pd.read_csv('data/tmdb_movies.csv')

def create_tfidf_matrix(movies):
    """Create TF-IDF matrix for the movie overviews."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'].fillna(''))
    return tfidf_matrix

def calculate_similarity(tfidf_matrix):
    """Calculate the cosine similarity matrix."""
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, movies, cosine_sim):
    """Get movie recommendations based on the provided title."""
    # Get the index of the movie that matches the title
    idx = movies.index[movies['title'] == title].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Skip the first one because it's the movie itself

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies.iloc[movie_indices]

if __name__ == "__main__":
    movies = load_data()
    tfidf_matrix = create_tfidf_matrix(movies)
    cosine_sim = calculate_similarity(tfidf_matrix)

    # Example: Get recommendations for a specific movie title
    movie_title = "Thor"  # Change this to test other titles
    recommendations = get_recommendations(movie_title, movies, cosine_sim)

    print(f"Recommendations for '{movie_title}':")
    print(recommendations[['title', 'release_date', 'vote_average']])

import streamlit as st
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
    try:
        idx = movies.index[movies['title'] == title].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 recommendations
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices]
    except IndexError:
        return None  # Return None if the movie title is not found

# Main Streamlit app
def main():
    st.title("Movie Recommendation System")
    st.write("Enter a movie title to get recommendations.")

    # Load the data and prepare the TF-IDF matrix
    movies = load_data()
    tfidf_matrix = create_tfidf_matrix(movies)
    cosine_sim = calculate_similarity(tfidf_matrix)

    # Input for movie title
    movie_title = st.text_input("Movie Title:")

    if movie_title:
        recommendations = get_recommendations(movie_title, movies, cosine_sim)
        if recommendations is not None:
            st.write(f"Recommendations for '{movie_title}':")
            st.dataframe(recommendations[['title', 'release_date', 'vote_average']])
        else:
            st.write("Movie not found. Please try another title.")

if __name__ == "__main__":
    main()
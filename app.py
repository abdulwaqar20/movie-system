import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Base URL for TMDB images
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

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
    # Perform fuzzy matching for the title
    movie_titles = movies['title'].tolist()
    match = process.extractOne(title, movie_titles)
    
    if match and match[1] > 70:  # You can adjust the threshold (70) based on accuracy
        idx = movies.index[movies['title'] == match[0]].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 recommendations
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices]
    else:
        return None  # Return None if no good match is found
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
            for _, row in recommendations.iterrows():
                st.write(f"**{row['title']}** (Release Date: {row['release_date']}, Rating: {row['vote_average']})")
                # Display the movie poster
                if pd.notnull(row['poster_path']):
                    poster_url = TMDB_IMAGE_URL + row['poster_path']
                    st.image(poster_url, width=200)  # Adjust width as needed
                st.markdown("---")  # Add a separator between movies
        else:
            st.write("Movie not found. Please try another title.")

if __name__ == "__main__":
    main()

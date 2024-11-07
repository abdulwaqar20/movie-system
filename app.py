import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process  # For fuzzy matching
import plotly.express as px  # For enhanced visualization
import ast

# Base URL for TMDB images
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

# Genre ID to Name Mapping (you can extend this based on your dataset)
GENRE_ID_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}

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

def fuzzy_match_title(title, movies):
    """Match movie title with fuzzy matching."""
    movie_titles = movies['title'].tolist()
    matched_title, score = process.extractOne(title, movie_titles)
    return matched_title

def get_genre_names(genre_ids):
    """Convert genre IDs to genre names using the GENRE_ID_MAP."""
    if pd.notnull(genre_ids):
        # Safely parse the genre_ids string to a list if needed
        if isinstance(genre_ids, str):
            try:
                genre_ids = ast.literal_eval(genre_ids)  # Safely convert string to list
            except (ValueError, SyntaxError):
                genre_ids = []  # In case there's a parsing error, fallback to empty list

        # Now genre_ids should be a list
        genre_names = [GENRE_ID_MAP.get(id, "Unknown") for id in genre_ids]
        return ", ".join(genre_names)
    return "Not available"

def get_recommendations(title, movies, cosine_sim):
    """Get movie recommendations based on the provided title."""
    try:
        # Use fuzzy matching to handle partial matches
        matched_title = fuzzy_match_title(title, movies)
        
        if not matched_title:
            return None  # Return None if no match is found

        # Find index of the matched movie
        idx = movies.index[movies['title'] == matched_title].tolist()[0]
        
        # Calculate similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 recommendations
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices]
    except IndexError:
        return None  # Return None if the movie title is not found

# Main Streamlit app
def main():
    st.set_page_config(page_title="Movie Recommendation System", layout="wide")
    
    # Title and description of the app
    st.title("🎬 Movie Recommendation System")
    st.markdown("""
    Welcome to the Movie Recommendation System! 🎥
    Enter a movie title in the sidebar to get personalized movie recommendations based on similar movie overviews.
    The system will also try to find similar movies even if you enter partial or misspelled titles.
    """)
    
    # Sidebar for input
    st.sidebar.header("Movie Recommendation Search")
    
    # Movie title input field
    movie_title = st.sidebar.text_input("Enter a Movie Title:", placeholder="e.g., The Dark Knight")

    # Load the data and prepare the TF-IDF matrix
    movies = load_data()
    tfidf_matrix = create_tfidf_matrix(movies)
    cosine_sim = calculate_similarity(tfidf_matrix)

    # If a title is entered, get recommendations
    if movie_title:
        with st.spinner("Finding movie recommendations..."):
            recommendations = get_recommendations(movie_title, movies, cosine_sim)

        # Display recommendations or error message
        if recommendations is not None:
            st.subheader(f"Recommendations for **{movie_title}**:")
            
            # Display a Plotly bar chart of movie ratings
            fig = px.bar(recommendations, x='title', y='vote_average', color='vote_average', title="Top 10 Movie Ratings", 
                         labels={'vote_average': 'Rating', 'title': 'Movie Title'})
            st.plotly_chart(fig)

            for _, row in recommendations.iterrows():
                # Movie title and basic info
                st.markdown(f"### **{row['title']}**")
                st.markdown(f"**Release Date:** {row['release_date']} | **Rating:** {row['vote_average']} ⭐")

                # Display the movie poster if available
                if pd.notnull(row['poster_path']):
                    poster_url = TMDB_IMAGE_URL + row['poster_path']
                    st.image(poster_url, width=200)  # Adjust width as needed

                st.markdown(f"**Overview:** {row['overview']}")
                st.markdown(f"**Genres:** {get_genre_names(row['genre_ids'])}")
                st.markdown("---")  # Separator between movies
        else:
            st.error(f"Movie titled '{movie_title}' not found. Please try another title or check for typos!")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process
import plotly.express as px
import ast
import numpy as np

# Constants
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
GENRE_ID_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance", 878: "Sci-Fi",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

# Cache data to avoid reloading
@st.cache_data
def load_data():
    return pd.read_csv('data/tmdb_movies.csv')

@st.cache_data
def create_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['overview'].fillna(''))
    return tfidf_matrix

@st.cache_data
def calculate_similarity(_tfidf_matrix):
    """Calculate the cosine similarity matrix."""
    return linear_kernel(_tfidf_matrix, _tfidf_matrix)

def fuzzy_match_title(title, movie_titles):
    matched_title, score = process.extractOne(title, movie_titles, scorer=process.fuzz.token_set_ratio)
    return matched_title if score > 70 else None

def get_genre_names(genre_ids):
    try:
        if isinstance(genre_ids, str):
            genre_ids = ast.literal_eval(genre_ids)
        return ", ".join([GENRE_ID_MAP.get(id, "Unknown") for id in genre_ids])
    except:
        return "Not available"

def get_recommendations(title, movies, cosine_sim, genre_filter=None):
    movie_titles = movies['title'].tolist()
    matched_title = fuzzy_match_title(title, movie_titles)
    if not matched_title:
        return None

    # Get the index of the matched title
    idx = movies.index[movies['title'] == matched_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the similarity scores and get the top 10 most similar movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:16]

    # Filter recommendations to keep unique movie titles using movie ID or title
    seen_titles = set()
    unique_recommendations = []
    
    for i, score in sim_scores:
        movie_title = movies.iloc[i]['title']
        if movie_title not in seen_titles:
            unique_recommendations.append(i)
            seen_titles.add(movie_title)
    
    # Create a DataFrame of the unique recommended movies
    recommendations = movies.iloc[unique_recommendations]
    
    # Apply genre filter if selected
    if genre_filter:
        recommendations = recommendations[recommendations['genre_ids'].apply(lambda x: genre_filter in get_genre_names(x))]
    
    return recommendations

# Main Streamlit app
def main():
    st.set_page_config(page_title="Enhanced Movie Recommender", layout="wide")
    
    # App title
    st.title("🎬 Enhanced Movie Recommendation System")
    st.markdown("Get movie recommendations with filters and visualizations!")

    # Sidebar input
    st.sidebar.header("Search Options")
    movie_title = st.sidebar.text_input("Enter a Movie Title", placeholder="e.g., Inception")
    
    # Genre Filter
    genre_options = list(GENRE_ID_MAP.values())
    selected_genre = st.sidebar.selectbox("Filter by Genre (Optional)", ["All"] + genre_options)
    
    # Load data
    movies = load_data()
    tfidf_matrix = create_tfidf_matrix(movies)
    cosine_sim = calculate_similarity(tfidf_matrix)
    
    if movie_title:
        with st.spinner("Searching for recommendations..."):
            genre_filter = selected_genre if selected_genre != "All" else None
            recommendations = get_recommendations(movie_title, movies, cosine_sim, genre_filter)

        # Display recommendations
        if recommendations is not None and not recommendations.empty:
            st.subheader(f"Top Recommendations for '{movie_title}'")

            fig = px.bar(
                recommendations,
                y='title',
                x='vote_average',
                orientation='h',  # Horizontal bar chart
                color='vote_average',  # Color based on rating
                color_continuous_scale='Viridis',  # Visually appealing gradient color scale
                title="Top 10 Movie Ratings",
                labels={'vote_average': 'Rating', 'title': 'Movie Title'},
                hover_data=['release_date', 'vote_average']
            )

            # Customizing the layout for a better visual appeal
            fig.update_layout(
                xaxis_title="Rating",
                yaxis_title="Movie Title",
                yaxis={'categoryorder': 'total ascending'},  # Sort by rating
                coloraxis_colorbar=dict(
                    title="Rating",
                    ticks="outside",
                    tickcolor='#333333',
                    tickfont=dict(color='#333333')
                ),
                plot_bgcolor='#ffffff',  # White background
                paper_bgcolor='#f0f0f0',  # Light gray background for the paper
                font=dict(color="#333333", size=14),  # Dark gray font for better contrast
                margin=dict(l=100, r=20, t=50, b=50),  # Adding some padding for better readability
            )

            # Display the chart
            st.plotly_chart(fig)

            # Display details of each recommended movie
            for _, row in recommendations.iterrows():
                st.markdown(f"### **{row['title']}**")
                st.markdown(f"**Release Date:** {row['release_date']} | **Rating:** {row['vote_average']} ⭐")
                st.markdown(f"**Genres:** {get_genre_names(row['genre_ids'])}")
                
                if pd.notnull(row['poster_path']):
                    st.image(TMDB_IMAGE_URL + row['poster_path'], width=150)
                
                st.markdown(f"**Overview:** {row['overview']}")
                st.markdown("---")
            
        else:
            st.error(f"No recommendations found for '{movie_title}'. Try another title or adjust filters!")

if __name__ == "__main__":
    main()

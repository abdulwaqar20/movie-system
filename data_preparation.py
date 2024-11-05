import pandas as pd

def load_tmdb_data():
    """Load the TMDB movies dataset from a CSV file."""
    movies = pd.read_csv('data/tmdb_movies.csv')
    return movies

def preprocess_tmdb_data(movies):
    """Clean and prepare the TMDB movies dataset."""
    # Drop duplicates
    movies.drop_duplicates(inplace=True)
    
    # Drop movies with missing titles or ratings
    movies = movies[movies['title'].notnull() & movies['vote_average'].notnull()]
    
    # Convert release_date to datetime
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    
    # Filter out movies with low ratings
    movies = movies[movies['vote_average'] >= 5.0]  # Keep only movies with rating >= 5.0
    
    # Extract relevant columns (you can adjust based on your needs)
    relevant_columns = ['id', 'title', 'release_date', 'vote_average', 'overview']
    movies = movies[relevant_columns]
    
    return movies

if __name__ == "__main__":
    movies = load_tmdb_data()
    movies = preprocess_tmdb_data(movies)
    print(f"Cleaned Movies Data: {movies.shape[0]} movies remaining")
    print(movies.head())

import requests
import pandas as pd

# Replace this with your actual API key
API_KEY = 'd23fe21bb016637e8fd2834184740a39'
BASE_URL = 'https://api.themoviedb.org/3'

def fetch_movies(page=1):
    """Fetch popular movies from the TMDB API."""
    url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()['results']
    else:
        print(f"Error fetching data: {response.status_code}")
        return []

def save_movies_to_csv():
    """Fetch movies and save them to a CSV file."""
    all_movies = []
    for page in range(1, 500):  # Fetch data from the first 5 pages
        movies = fetch_movies(page)
        all_movies.extend(movies)
    
    # Create a DataFrame and save to CSV
    movies_df = pd.DataFrame(all_movies)
    movies_df.to_csv('data/tmdb_movies.csv', index=False)
    print(f"Saved {len(all_movies)} movies to tmdb_movies.csv")

if __name__ == "__main__":
    save_movies_to_csv()

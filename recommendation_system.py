import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_movies(game_id, loaded_vectors, top_n=2):
    # Check if the game_id exists in the loaded_vectors
    if game_id not in loaded_vectors.index:
        print(f"Game ID {game_id} not found.")
        return []
    # Get the vector for the input game ID
    game_vector = loaded_vectors.loc[game_id].values.reshape(1, -1)
    # Calculate cosine similarity with all other movie vectors
    similarities = cosine_similarity(game_vector, loaded_vectors)
    # Create a Pandas Series of similarities
    sim_scores = pd.Series(similarities.flatten(), index=loaded_vectors.index)
    # Remove the game itself from the recommendations
    sim_scores = sim_scores.drop(game_id)
    # Get the top_n most similar movies
    most_similar_movies = sim_scores.nlargest(top_n)
    return most_similar_movies.index.tolist()


movies_metadata_path = r"f:\projects\Movie-Recommender\dataset\movies_metadata.csv"
movies_df = pd.read_csv(movies_metadata_path, low_memory=False)


def get_movie_details(movie_ids, movies_df):
    movie_details = []
    for movie_id in movie_ids:
        movie_id = str(int(movie_id))
        # Check if the movie_id exists in the DataFrame
        movie = movies_df[movies_df['id'] == movie_id]
        if not movie.empty:
            details = {
                'id': movie_id,
                'title': movie['title'].values[0],
                'overview': movie['overview'].values[0]
            }
            movie_details.append(details)
        else:
            movie_details.append({
                'id': movie_id,
                'title': None,
                'overview': None
            })
    return movie_details


def main():
    print("Enter the movie ID to get similar movie IDs. Here are some examples:")
    print("ID 862 : Toy Story")
    print("ID 4584 : Sense and Sensibility")
    print("ID 9603 : Clueless")
    print("ID 807 : Se7en")
    # Get the game ID from user input
    game_id = int(input("Enter the game ID: "))  # Prompting user for game ID
    loaded_vectors = pd.read_csv(
        'movie_vectorization/movie_vectors.csv', index_col='movie_id')
    loaded_vectors = loaded_vectors.fillna(0)
    # Get recommended movie IDs based on the game ID
    recommended_movies = recommend_movies(game_id, loaded_vectors)
    print(f"Recommended movie IDs for game ID {game_id}: {recommended_movies}")
    # Retrieve the movie details for the recommended movie IDs
    # Limit to up to 3 recommendations
    movie_ids = recommended_movies[:3]
    movie_info = get_movie_details(movie_ids, movies_df)

    # Print the movie information
    for movie in movie_info:
        print(
            f"Movie ID: {movie['id']}, Title: {movie['title']}, Overview: {movie['overview']}")


if __name__ == "__main__":
    main()

"""Explainable content-based movie recommender using TF-IDF and cosine similarity."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
MOVIES_PATH = BASE_DIR / "data" / "movies.csv"
RATINGS_PATH = BASE_DIR / "data" / "ratings.csv"


class MovieRecommender:
    """Build a genre-based recommender from the local movie catalog."""

    def __init__(self, movies_path=MOVIES_PATH, ratings_path=RATINGS_PATH):
        if not Path(movies_path).exists():
            raise FileNotFoundError(
                f"Movie data not found at {movies_path}. Run `python prepare_data.py` first "
                "(it downloads IMDb's public datasets and builds the CSV files)."
            )
        self.movies = pd.read_csv(movies_path)
        self.movies["genres"] = self.movies["genres"].fillna("").replace("(no genres listed)", "")

        if Path(ratings_path).exists():
            ratings = pd.read_csv(ratings_path)
            rating_columns = [column for column in ["movieId", "average_rating", "rating_count"] if column in ratings]
            if len(rating_columns) == 3:
                self.movies = self.movies.merge(ratings[rating_columns], on="movieId", how="left")
            elif "movieId" in ratings and "rating" in ratings:
                summary = ratings.groupby("movieId")["rating"].agg(
                    average_rating="mean", rating_count="count"
                ).reset_index()
                self.movies = self.movies.merge(summary, on="movieId", how="left")

        for column, default in [("average_rating", 0.0), ("rating_count", 0)]:
            if column not in self.movies:
                self.movies[column] = default
            self.movies[column] = self.movies[column].fillna(default)
        self.movies["average_rating"] = self.movies["average_rating"].astype(float).round(1)
        self.movies["rating_count"] = self.movies["rating_count"].astype(int)

        # Weighted rating (the Bayesian average IMDb uses for its Top 250): a rating based on few
        # votes is pulled towards the catalog average, so 9.0 from 150 votes does not beat 8.2 from
        # 500,000 votes. It is used to choose between movies that are equally similar by genre.
        votes = self.movies["rating_count"]
        rated = self.movies["rating_count"] > 0
        mean_rating = self.movies.loc[rated, "average_rating"].mean() if rated.any() else 0.0
        min_votes = votes.quantile(0.90)
        denominator = (votes + min_votes).replace(0, 1)
        self.movies["weighted_rating"] = (
            votes / denominator * self.movies["average_rating"] + min_votes / denominator * mean_rating
        )
        self.weighted = self.movies["weighted_rating"].to_numpy()

        genre_text = self.movies["genres"].str.replace("|", " ", regex=False)
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
        self.tfidf_matrix = self.vectorizer.fit_transform(genre_text)
        self.title_to_index = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()
        self.search_titles_cache = self.movies["title"].astype(str).tolist()

    def search_titles(self, query, limit=30):
        """Return titles containing query text, ordered by title."""
        query = (query or "").strip().lower()
        if not query:
            return []
        matches = [title for title in self.search_titles_cache if query in title.lower()]
        return sorted(matches, key=str.lower)[:limit]

    def recommend(self, title, number=10):
        """Return the most genre-similar movies for an exact title."""
        if title not in self.title_to_index:
            return []

        movie_index = int(self.title_to_index[title])
        # Compare only one selected vector with all movies. This avoids a
        # large dense all-by-all matrix when using the larger IMDb catalog.
        scores = cosine_similarity(self.tfidf_matrix[movie_index], self.tfidf_matrix).ravel()
        scores[movie_index] = -1  # never recommend the selected movie itself

        # Genres alone give hundreds of movies the exact same similarity score, so sort by
        # similarity first (rounded, so tiny floating-point differences count as ties) and use the
        # weighted rating to decide between equally similar movies. lexsort uses its LAST key first.
        ranked_indices = np.lexsort((-self.weighted, -np.round(scores, 4)))[:number]

        recommendations = []
        for index in ranked_indices:
            movie = self.movies.iloc[index]
            recommendations.append({
                "title": movie["title"],
                "genres": movie["genres"].replace("|", " • ") if movie["genres"] else "Genre not listed",
                "score": round(float(scores[index]) * 100, 1),
                "average_rating": float(movie["average_rating"]),
                "rating_count": int(movie["rating_count"]),
            })
        return recommendations


if __name__ == "__main__":
    recommender = MovieRecommender()
    print("Example recommendations for The Dark Knight (2008):")
    for movie in recommender.recommend("The Dark Knight (2008)", 5):
        print(f"- {movie['title']} ({movie['score']}% similarity)")

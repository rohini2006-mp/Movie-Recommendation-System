"""Build data/movies.csv and data/ratings.csv from IMDb's public datasets.

Run once before starting the app:   python prepare_data.py

What it does:
  1. Downloads title.basics.tsv.gz (about 200 MB) and title.ratings.tsv.gz from https://datasets.imdbws.com/
     into data/raw/. If the files are already there, it uses them instead of downloading again.
  2. Keeps non-adult feature films that have genres, a release year up to 2026, and at least 100 votes.
  3. Writes the two small CSV files the app reads.

IMDb's datasets are for personal, non-commercial use: https://developer.imdb.com/non-commercial-datasets/
"""
import csv
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"
BASE_URL = "https://datasets.imdbws.com/"
BASICS_FILE = "title.basics.tsv.gz"
RATINGS_FILE = "title.ratings.tsv.gz"
MIN_VOTES = 100
LAST_YEAR = 2026


def download(name: str) -> Path:
    target = RAW_DIR / name
    if target.exists():
        print(f"Using existing {target.relative_to(ROOT)}")
        return target
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} (this can take a few minutes)...")
    request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "movie-recommendation-student-project"})
    with urllib.request.urlopen(request) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out)
    return target


def read_tsv(path: Path, columns: list[str], chunksize: int | None = None):
    # IMDb files are tab-separated, use \N for "missing", and contain stray quote characters.
    return pd.read_csv(
        path, sep="\t", compression="gzip", usecols=columns, dtype=str,
        na_values="\\N", keep_default_na=False, quoting=csv.QUOTE_NONE, chunksize=chunksize,
    )


def build_catalog(basics_path: Path, ratings_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (movies, ratings) tables ready to be saved as CSV."""
    ratings = read_tsv(ratings_path, ["tconst", "averageRating", "numVotes"])
    ratings["averageRating"] = pd.to_numeric(ratings["averageRating"], errors="coerce")
    ratings["numVotes"] = pd.to_numeric(ratings["numVotes"], errors="coerce")
    ratings = ratings.dropna()
    ratings = ratings[ratings["numVotes"] >= MIN_VOTES]
    wanted = set(ratings["tconst"])

    pieces = []
    # The basics file has millions of rows (episodes, shorts...), so filter it piece by piece.
    for chunk in read_tsv(basics_path, ["tconst", "titleType", "primaryTitle", "isAdult", "startYear", "genres"], 500_000):
        chunk = chunk[(chunk["titleType"] == "movie") & (chunk["isAdult"] == "0")]
        chunk = chunk.dropna(subset=["primaryTitle", "startYear", "genres"])
        pieces.append(chunk[chunk["tconst"].isin(wanted)])
    basics = pd.concat(pieces, ignore_index=True)
    basics["startYear"] = pd.to_numeric(basics["startYear"], errors="coerce")
    basics = basics.dropna(subset=["startYear"])
    basics = basics[basics["startYear"] <= LAST_YEAR]

    table = basics.merge(ratings, on="tconst")
    table["title"] = table["primaryTitle"] + " (" + table["startYear"].astype(int).astype(str) + ")"
    # Two different films can share a title and year. Keep the better-known one so every title is unique,
    # because the app finds a movie by its exact title.
    table = table.sort_values("numVotes", ascending=False).drop_duplicates("title").sort_values("title")

    movie_ids = table["tconst"].str[2:].astype(int)
    movies = pd.DataFrame({"movieId": movie_ids, "title": table["title"], "genres": table["genres"].str.replace(",", "|", regex=False)})
    ratings_out = pd.DataFrame({"movieId": movie_ids, "average_rating": table["averageRating"], "rating_count": table["numVotes"].astype(int)})
    return movies, ratings_out


def main() -> None:
    basics_path = download(BASICS_FILE)
    ratings_path = download(RATINGS_FILE)
    print("Building the movie catalog...")
    movies, ratings = build_catalog(basics_path, ratings_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_csv(OUT_DIR / "movies.csv", index=False)
    ratings.to_csv(OUT_DIR / "ratings.csv", index=False)
    print(f"Saved {len(movies):,} movies to data/movies.csv and data/ratings.csv")


if __name__ == "__main__":
    main()

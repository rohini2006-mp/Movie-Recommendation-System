"""Tests for the recommender and the data preparation script. Run from the project folder with:  python -m pytest"""
import gzip
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import build_catalog  # noqa: E402
from recommender import MovieRecommender  # noqa: E402


def make_recommender(tmp_path):
    # A, B, C and D all have the exact same genres. E has only one of them. F is unrelated.
    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "title": ["A (2000)", "B (2001)", "C (2002)", "D (2003)", "E (2004)", "F (2005)"],
        "genres": ["Action|Crime"] * 4 + ["Action", "Comedy"],
    })
    ratings = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "average_rating": [7.0, 9.5, 8.0, 8.0, 9.9, 5.0],
        "rating_count": [1000, 120, 500000, 400000, 900000, 100000],
    })
    movies.to_csv(tmp_path / "movies.csv", index=False)
    ratings.to_csv(tmp_path / "ratings.csv", index=False)
    return MovieRecommender(tmp_path / "movies.csv", tmp_path / "ratings.csv")


def test_selected_movie_is_never_recommended(tmp_path):
    titles = [m["title"] for m in make_recommender(tmp_path).recommend("A (2000)", 5)]
    assert "A (2000)" not in titles


def test_equally_similar_movies_are_ordered_by_weighted_rating(tmp_path):
    titles = [m["title"] for m in make_recommender(tmp_path).recommend("A (2000)", 3)]
    # C (8.0 from 500k votes) and D (8.0 from 400k) beat B (9.5 but only 120 votes)
    assert titles == ["C (2002)", "D (2003)", "B (2001)"]


def test_closer_genre_match_beats_a_better_rating(tmp_path):
    titles = [m["title"] for m in make_recommender(tmp_path).recommend("A (2000)", 5)]
    assert titles.index("B (2001)") < titles.index("E (2004)")  # E has the best rating but fewer shared genres


def test_unknown_title_returns_nothing(tmp_path):
    assert make_recommender(tmp_path).recommend("Not a movie (1999)") == []


def test_missing_data_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        MovieRecommender(tmp_path / "nope.csv", tmp_path / "nope2.csv")


def test_build_catalog_filters_imdb_files(tmp_path):
    basics = "\n".join([
        "tconst\ttitleType\tprimaryTitle\toriginalTitle\tisAdult\tstartYear\tendYear\truntimeMinutes\tgenres",
        'tt0000001\tmovie\tGood Film\tGood Film\t0\t2010\t\\N\t100\tDrama,Romance',
        'tt0000002\ttvEpisode\tAn Episode\tAn Episode\t0\t2010\t\\N\t30\tComedy',
        'tt0000003\tmovie\tAdult Film\tAdult Film\t1\t2010\t\\N\t90\tDrama',
        'tt0000004\tmovie\tNo Genres\tNo Genres\t0\t2010\t\\N\t90\t\\N',
        'tt0000005\tmovie\tFuture Film\tFuture Film\t0\t2031\t\\N\t90\tDrama',
        'tt0000006\tmovie\tFew Votes\tFew Votes\t0\t2010\t\\N\t90\tDrama',
        'tt0000007\tmovie\tSay "Hello"\tSay "Hello"\t0\t2012\t\\N\t90\tComedy',
        'tt0000008\tmovie\tTwin\tTwin\t0\t2015\t\\N\t90\tDrama',
        'tt0000009\tmovie\tTwin\tTwin\t0\t2015\t\\N\t90\tAction',
    ])
    ratings = "\n".join([
        "tconst\taverageRating\tnumVotes",
        "tt0000001\t7.5\t5000", "tt0000002\t8.0\t5000", "tt0000003\t6.0\t5000", "tt0000004\t6.0\t5000",
        "tt0000005\t6.0\t5000", "tt0000006\t9.0\t50", "tt0000007\t6.5\t300",
        "tt0000008\t5.0\t200", "tt0000009\t7.0\t900",
    ])
    for name, text in [("basics.tsv.gz", basics), ("ratings.tsv.gz", ratings)]:
        with gzip.open(tmp_path / name, "wt", encoding="utf-8") as handle:
            handle.write(text + "\n")
    movies, ratings_out = build_catalog(tmp_path / "basics.tsv.gz", tmp_path / "ratings.tsv.gz")
    assert sorted(movies["title"]) == ['Good Film (2010)', 'Say "Hello" (2012)', "Twin (2015)"]
    assert movies.loc[movies["title"] == "Good Film (2010)", "genres"].iloc[0] == "Drama|Romance"
    assert movies.loc[movies["title"] == "Twin (2015)", "genres"].iloc[0] == "Action"  # the better-known twin was kept
    assert list(ratings_out.columns) == ["movieId", "average_rating", "rating_count"]

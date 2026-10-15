# Movie Recommendation System

A beginner-friendly **content-based movie recommendation system** built with Python and Flask. Search for a movie, select it, and receive ten titles with similar genres. Recommendations are calculated from IMDb's public dataset; they are not hardcoded.

## Project objective

This project builds a simple recommendation system using content-based filtering. It shows how movie metadata can be converted into numerical features, compared to find similar movies, and ranked sensibly when many movies look equally similar.

## Dataset used

The project uses IMDb's **non-commercial public datasets**, which IMDb refreshes daily:

- Source: https://datasets.imdbws.com/
- Documentation and terms: https://developer.imdb.com/non-commercial-datasets/

The data files are **not stored in this repository**. `prepare_data.py` downloads `title.basics.tsv.gz` and `title.ratings.tsv.gz` (about 200 MB) and builds two small CSV files:

- `data/movies.csv`: numeric movie ID, title with release year, and pipe-separated genres
- `data/ratings.csv`: IMDb average rating out of 10 and vote count

Filters applied: non-adult feature films, with genres, released up to 2026, with at least 100 votes (roughly 145,000 movies). If two films share the same title and year, the one with more votes is kept so every title is unique.

## How the recommendation works

1. **Load and prepare data:** `recommender.py` reads the two CSV files with pandas and attaches each movie's rating and vote count.
2. **Create movie features:** Each movie's genres are its content features. For example, `Action|Adventure|Sci-Fi` becomes a small text document of those genre words.
3. **TF-IDF:** `TfidfVectorizer` converts the genre text into numerical vectors.
4. **Cosine similarity:** `cosine_similarity` compares the selected movie's vector with every other movie's. A score of 1 means identical genre vectors.
5. **Break ties with a weighted rating:** There are only about 1,000 distinct genre combinations across ~145,000 movies, so hundreds of movies can tie with a perfect genre score. Movies are therefore sorted by similarity first, and ties are broken by a *weighted rating*: `WR = v/(v+m) x R + m/(v+m) x C`, where R is the movie's average rating, v its votes, m the 90th-percentile vote count, and C the catalog's mean rating. This is the Bayesian average IMDb uses for its Top 250: a 9.5 from 120 votes does not outrank an 8.0 from 500,000 votes.
6. **Show results:** The selected movie is excluded and the top 10 are sent to the Flask template.

This is **content-based filtering** because it compares item features (genres) rather than learning from groups of similar users.

### Example

Results from the IMDb snapshot used during development (they shift slightly as IMDb updates):

| Selected movie | Top recommendations |
|---|---|
| Toy Story (1995) | Up (2009), Toy Story 3 (2010), Finding Nemo (2003), Klaus (2019), Monsters, Inc. (2001) |
| Titanic (1997) | Forrest Gump (1994), Cinema Paradiso (1988), Good Will Hunting (1997), Before Sunset (2004), 96 (2018) |

## Screenshots

![Search](docs/screenshots/search.png)
![Recommendations for Toy Story](docs/screenshots/results-toy-story.png)

## Project structure

```text
Movie-Recommendation-System/
├── data/                  (created by prepare_data.py, not committed)
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── tests/
│   └── test_recommender.py
├── app.py
├── recommender.py
├── prepare_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation on Windows

Python 3.9 or newer is recommended. Open the project folder in VS Code and open **Terminal -> New Terminal**.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py
python app.py
```

`prepare_data.py` only needs to run once (the download takes a few minutes). If `python` is not recognized, use `py` instead. On Mac/Linux use `python3` and `source venv/bin/activate`.

Open **http://127.0.0.1:5000/** in a browser. Type a title in the search box, click a matching result, and then click **Recommend**.

The application is intentionally local-only: it binds to `127.0.0.1` and Flask's debug mode is off (set the environment variable `FLASK_DEBUG=1` yourself if you want it while developing).

## Run the tests

```powershell
python -m pytest
```

The tests check that the selected movie is never recommended, that ties are broken by weighted rating, that a closer genre match beats a better rating, and that `prepare_data.py` filters IMDb's files correctly.

## Search and interface

As the user types at least two characters, the `/search` endpoint returns matching movie titles. Selecting a result sets the exact title used by the recommender, which prevents spelling and title-format mistakes.

## Limitations

- **Genres are the only feature.** Two movies with the same genres count as equally similar, even if they have nothing else in common. Adding plot keywords, cast, or director would improve quality.
- **Ratings only break ties.** They are not part of the similarity itself.
- **No accuracy score.** A content-based recommender has no single "right answer" to measure against, so quality is judged by looking at the results. Measuring it properly would need user viewing history.
- **No personalisation.** Everyone gets the same list for the same movie. A collaborative-filtering version based on user ratings would be the next step.

## Files to understand first

- `app.py` creates the Flask app, exposes the search endpoint, handles the form, and renders the page.
- `recommender.py` holds data loading, TF-IDF, cosine similarity, weighted rating, search, and recommendations.
- `prepare_data.py` downloads and filters IMDb's files.
- `templates/index.html` and `static/style.css` hold the interface.

## Example explanation for an internship review

> When a user searches for and selects a movie, the system reads its genre list and compares it with every other movie. TF-IDF changes each genre list into a numeric vector, and cosine similarity measures how close two vectors are. Because many movies share exactly the same genres, ties are broken by a weighted rating that favours well-rated movies with many votes. The ten top-ranked titles are displayed, so every result is generated from IMDb data rather than a manually written list.

## Attribution and terms

Data comes from IMDb's non-commercial datasets and is used here for an educational project. It is subject to IMDb's terms; review the [official documentation](https://developer.imdb.com/non-commercial-datasets/) before using it outside this project.

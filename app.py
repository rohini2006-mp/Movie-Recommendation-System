import os

from flask import Flask, jsonify, render_template, request

from recommender import MovieRecommender

app = Flask(__name__)
recommender = MovieRecommender()


@app.route("/search")
def search():
    query = request.args.get("q", "")
    return jsonify(recommender.search_titles(query))


@app.route("/", methods=["GET", "POST"])
def index():
    selected_movie = request.form.get("movie", "") if request.method == "POST" else ""
    recommendations = []
    error = None

    if request.method == "POST":
        if not selected_movie:
            error = "Search for a movie and select one result first."
        else:
            recommendations = recommender.recommend(selected_movie, number=10)
            if not recommendations:
                error = "That movie was not found. Please choose a result from the search list."

    return render_template(
        "index.html",
        selected_movie=selected_movie,
        recommendations=recommendations,
        error=error,
        movie_count=len(recommender.movies),
        catalog_count=f"{len(recommender.movies):,}",
    )


if __name__ == "__main__":
    # Local-only: bind to 127.0.0.1 so other devices on the network cannot reach the app.
    # Flask's debug mode is off unless you set FLASK_DEBUG=1 yourself while developing.
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")

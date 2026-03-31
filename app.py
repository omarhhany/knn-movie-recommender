import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import requests

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

TMDB_API_KEY = "42a72f47065dde345cbcb5418510ddd8"

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    ratings = pd.read_csv(
        "data/ml-100k/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        "data/ml-100k/u.item",
        sep="|",
        encoding="latin-1",
        names=[
            "movie_id", "title", "release_date", "video_release_date",
            "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
            "Children", "Comedy", "Crime", "Documentary", "Drama",
            "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
    )

    return ratings, movies


ratings, movies = load_data()

# =========================
# USER-MOVIE MATRIX
# =========================

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(user_movie_matrix)

# =========================
# POSTER FUNCTION (FIXED)
# =========================

def get_poster(title):
    try:
        # extract name + year
        if "(" in title:
            name = title.split("(")[0].strip()
            year = title.split("(")[1].replace(")", "")
        else:
            name = title
            year = ""

        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": name,
            "year": year
        }

        data = requests.get(url, params=params).json()

        if data["results"]:
            # sort by popularity (better match)
            results = sorted(
                data["results"],
                key=lambda x: x.get("popularity", 0),
                reverse=True
            )

            poster_path = results[0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except:
        pass

    return None

# =========================
# RECOMMENDER FUNCTION
# =========================

def recommend_movies(user_id, k=5, genre=None):
    user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model.kneighbors(user_vector, n_neighbors=5)

    similar_users = indices.flatten()[1:]
    user_ids = user_movie_matrix.index.tolist()
    similar_user_ids = [user_ids[i] for i in similar_users]

    similarity_scores = 1 - distances.flatten()[1:]

    weighted_sum = np.zeros(user_movie_matrix.shape[1])
    similarity_sum = np.zeros(user_movie_matrix.shape[1])

    for i, sim in zip(similar_user_ids, similarity_scores):
        user_ratings = user_movie_matrix.loc[i].values
        weighted_sum += sim * user_ratings
        similarity_sum += sim

    scores = weighted_sum / (similarity_sum + 1e-8)
    scores_series = pd.Series(scores, index=user_movie_matrix.columns)

    watched_movies = user_movie_matrix.loc[user_id]
    watched_movies = watched_movies[watched_movies > 0].index.tolist()

    scores_series = scores_series.drop(watched_movies)

    # GENRE FILTER
    if genre and genre != "All":
        genre_movies = movies[movies[genre] == 1]["movie_id"]
        scores_series = scores_series[scores_series.index.isin(genre_movies)]

        if len(scores_series) < k:
            st.warning("Not enough movies in this genre. Showing best available.")

    top_movies = scores_series.sort_values(ascending=False).head(k)

    return top_movies, similar_user_ids

# =========================
# UI
# =========================

st.title("🎬 AI Movie Recommender")
st.markdown("Built using KNN collaborative filtering with smart recommendations")

user_id = st.selectbox("Select User ID", user_movie_matrix.index.tolist())

genres = [
    "All", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

selected_genre = st.selectbox("Filter by Genre", genres)

if st.button("Recommend Movies"):

    top_movies, similar_users = recommend_movies(
        user_id,
        k=5,
        genre=selected_genre
    )

    st.subheader("🎯 Your Personalized Recommendations")

    for i, (movie_id, score) in enumerate(top_movies.items(), 1):
        movie = movies[movies["movie_id"] == movie_id]["title"].values[0]

        poster = get_poster(movie)

        col1, col2 = st.columns([1, 4])

        with col1:
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown("🎬 No image")

        with col2:
            st.markdown(f"### {i}. {movie}")
            st.progress(min(score / 5, 1.0))
            st.write(f"⭐ Score: {score:.2f}")

        st.markdown("---")

    with st.expander("👥 See similar users"):
        for i, user in enumerate(similar_users, 1):
            st.write(f"{i}. User {user}")

    st.info("This system recommends movies using KNN based on user similarity.")
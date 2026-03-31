import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

TMDB_API_KEY = "42a72f47065dde345cbcb5418510ddd8"

st.title("🎬 AI Movie Recommender")
st.markdown("KNN-based collaborative filtering with smart recommendations")

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
# MATRIX + MODEL
# =========================

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(user_movie_matrix)

# =========================
# TMDB POSTER FUNCTION
# =========================

@st.cache_data
def get_poster(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        data = requests.get(url).json()

        if data["results"]:
            poster_path = data["results"][0]["poster_path"]
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass

    return None

# =========================
# USER INPUT
# =========================

user_id = st.selectbox("Select User ID", user_movie_matrix.index.tolist())

genre_options = [
    "All", "Action", "Comedy", "Drama",
    "Romance", "Sci-Fi", "Horror", "Thriller"
]

selected_genre = st.selectbox("Filter by Genre", genre_options)

# =========================
# RECOMMENDER
# =========================

def get_recommendations(user_id, k=10):
    user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)

    distances, indices = model.kneighbors(user_vector, n_neighbors=5)

    similar_users = indices.flatten()[1:]
    similarity_scores = 1 - distances.flatten()[1:]

    user_ids = user_movie_matrix.index.tolist()
    similar_user_ids = [user_ids[i] for i in similar_users]

    user_movies = user_movie_matrix.loc[user_id]
    watched_movies = user_movies[user_movies > 0].index.tolist()

    weighted_sum = np.zeros(user_movie_matrix.shape[1])
    similarity_sum = np.zeros(user_movie_matrix.shape[1])

    for i, sim in zip(similar_user_ids, similarity_scores):
        user_ratings = user_movie_matrix.loc[i].values
        weighted_sum += sim * user_ratings
        similarity_sum += sim

    scores = weighted_sum / (similarity_sum + 1e-8)
    scores_series = pd.Series(scores, index=user_movie_matrix.columns)

    scores_series = scores_series.drop(watched_movies)

    top_movies = scores_series.sort_values(ascending=False).head(k)

    return top_movies, similar_user_ids

# =========================
# BUTTON
# =========================

if st.button("🚀 Recommend Movies"):

    top_movies, similar_user_ids = get_recommendations(user_id)

    st.subheader("🎯 Your Personalized Recommendations")

    displayed = 0

    for movie_id, score in top_movies.items():

        movie_row = movies[movies["movie_id"] == movie_id]

        if movie_row.empty:
            continue

        title = movie_row["title"].values[0]

        # =========================
        # GENRE FILTER
        # =========================
        if selected_genre != "All":
            if selected_genre not in movie_row.columns:
                continue
            if movie_row[selected_genre].values[0] != 1:
                continue

        poster = get_poster(title)

        col1, col2 = st.columns([1, 3])

        with col1:
            if poster:
                st.image(poster)
            else:
                st.write("🎬")

        with col2:
            st.markdown(f"### {title}")
            st.progress(float(score) / 5)
            st.caption(
                f"⭐ Score: {score:.2f} | Based on users {similar_user_ids[:2]}"
            )

        st.markdown("---")

        displayed += 1
        if displayed == 5:
            break

    # =========================
    # FALLBACK
    # =========================
    if displayed < 5:
        st.warning("Not enough movies in this genre. Showing best available results.")

    # =========================
    # SIMILAR USERS
    # =========================
    with st.expander("👥 See similar users"):
        for i, uid in enumerate(similar_user_ids, 1):
            st.write(f"{i}. User {uid}")

    st.info("Recommendations based on KNN similarity (cosine distance)")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Movie Recommender", layout="centered")

st.title("🎬 AI Movie Recommender")
st.markdown("KNN-based collaborative filtering using cosine similarity")

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
# CREATE MATRIX
# =========================
user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# =========================
# TRAIN MODEL
# =========================
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(user_movie_matrix)

# =========================
# USER INPUT
# =========================
user_id = st.selectbox(
    "Select User ID",
    user_movie_matrix.index.tolist()
)

# =========================
# RECOMMEND BUTTON
# =========================
if st.button("Recommend Movies"):

    with st.spinner("Finding best movies for you..."):

        # --- Get user vector ---
        user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)

        distances, indices = model.kneighbors(user_vector, n_neighbors=5)

        similar_users = indices.flatten()[1:]
        similarity_scores = 1 - distances.flatten()[1:]

        user_ids = user_movie_matrix.index.tolist()
        similar_user_ids = [user_ids[i] for i in similar_users]

        # --- Movies already watched ---
        user_movies = user_movie_matrix.loc[user_id]
        watched_movies = user_movies[user_movies > 0].index.tolist()

        # --- Weighted recommendation ---
        weighted_sum = np.zeros(user_movie_matrix.shape[1])
        similarity_sum = np.zeros(user_movie_matrix.shape[1])

        for sim_user, sim_score in zip(similar_user_ids, similarity_scores):
            user_ratings = user_movie_matrix.loc[sim_user].values
            weighted_sum += sim_score * user_ratings
            similarity_sum += sim_score

        scores = weighted_sum / (similarity_sum + 1e-8)
        scores_series = pd.Series(scores, index=user_movie_matrix.columns)

        # Remove watched movies
        scores_series = scores_series.drop(watched_movies)

        # Top 5
        top_movies = scores_series.sort_values(ascending=False).head(5)

    # =========================
    # DISPLAY RESULTS
    # =========================
    st.subheader("🎯 Your Personalized Recommendations")

    for i, (movie_id, score) in enumerate(top_movies.items(), 1):
        title = movies[movies["movie_id"] == movie_id]["title"].values[0]
        st.write(f"🎬 **{i}. {title}** ⭐ ({round(score, 2)})")

    st.markdown("---")

    # =========================
    # OPTIONAL: SHOW SIMILAR USERS
    # =========================
    with st.expander("👥 See similar users"):
        for i, uid in enumerate(similar_user_ids, 1):
            st.write(f"{i}. User {uid}")

    # =========================
    # EXPLANATION
    # =========================
    st.caption(
        "This system recommends movies using K-Nearest Neighbors (KNN) "
        "based on user similarity with cosine distance."
    )

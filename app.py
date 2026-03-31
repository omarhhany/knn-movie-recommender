import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

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
# BUILD MODEL
# =========================

@st.cache_data
def build_model(ratings):
    user_movie_matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(user_movie_matrix)

    return user_movie_matrix, model

user_movie_matrix, model = build_model(ratings)

# =========================
# RECOMMEND FUNCTION
# =========================

def recommend_movies(user_id, k=5):
    user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)

    distances, indices = model.kneighbors(user_vector, n_neighbors=5)

    similar_users_idx = indices.flatten()[1:]
    user_ids = user_movie_matrix.index.tolist()
    similar_user_ids = [user_ids[i] for i in similar_users_idx]

    similarity_scores = 1 - distances.flatten()[1:]

    weighted_sum = np.zeros(user_movie_matrix.shape[1])
    similarity_sum = np.zeros(user_movie_matrix.shape[1])

    for i, sim in zip(similar_user_ids, similarity_scores):
        weighted_sum += sim * user_movie_matrix.loc[i].values
        similarity_sum += sim

    scores = weighted_sum / (similarity_sum + 1e-8)
    scores_series = pd.Series(scores, index=user_movie_matrix.columns)

    # remove watched movies
    watched_movies = user_movie_matrix.loc[user_id]
    watched_movies = watched_movies[watched_movies > 0].index.tolist()
    scores_series = scores_series.drop(watched_movies)

    top_movies = scores_series.sort_values(ascending=False).head(k)

    return top_movies, similar_user_ids

# =========================
# UI
# =========================

st.set_page_config(page_title="AI Movie Recommender", layout="centered")

st.title("🎬 AI Movie Recommender")
st.markdown("Built using **KNN collaborative filtering** and **cosine similarity**")

st.divider()

# user selection
user_id = st.selectbox(
    "Select User ID",
    sorted(user_movie_matrix.index.tolist())[:100]
)

# button
if st.button("Recommend Movies"):
    with st.spinner("Finding best movies for you..."):
        recommendations, similar_users = recommend_movies(user_id)

    st.subheader("🎯 Your Personalized Recommendations")

    # show similar users
    st.write("👥 Similar Users:", similar_users)

    st.divider()

    # show recommendations
    for movie_id, score in recommendations.items():
        title = movies[movies["movie_id"] == movie_id]["title"].values[0]
        st.markdown(f"**🎥 {title}**  \n⭐ Rating Score: `{score:.2f}`")

    st.divider()
    st.caption("Built by Omar | AI Engineer Project | KNN Recommender System")
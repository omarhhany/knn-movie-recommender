import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# =========================
# 1. LOAD DATA
# =========================

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

# =========================
# 2. CREATE USER-MOVIE MATRIX
# =========================

user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# =========================
# 3. TRAIN KNN MODEL
# =========================

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(user_movie_matrix)

# =========================
# 4. FIND SIMILAR USERS
# =========================

user_id = 1

user_vector = user_movie_matrix.loc[user_id].values.reshape(1, -1)

distances, indices = model.kneighbors(user_vector, n_neighbors=5)

similar_users = indices.flatten()[1:]

user_ids = user_movie_matrix.index.tolist()
similar_user_ids = [user_ids[i] for i in similar_users]

print("\nSimilar user IDs:", similar_user_ids)

# =========================
# 5. RECOMMEND MOVIES (WEIGHTED)
# =========================

user_movies = user_movie_matrix.loc[user_id]
watched_movies = user_movies[user_movies > 0].index.tolist()

similarity_scores = 1 - distances.flatten()[1:]

weighted_sum = np.zeros(user_movie_matrix.shape[1])
similarity_sum = np.zeros(user_movie_matrix.shape[1])

for i, sim in zip(similar_user_ids, similarity_scores):
    user_ratings = user_movie_matrix.loc[i].values
    weighted_sum += sim * user_ratings
    similarity_sum += sim

scores = weighted_sum / (similarity_sum + 1e-8)

scores_series = pd.Series(scores, index=user_movie_matrix.columns)

scores_series = scores_series.drop(watched_movies)

top_movies = scores_series.sort_values(ascending=False).head(5)

print("\nRecommended movie IDs:")
print(top_movies)

# =========================
# 6. SHOW MOVIE NAMES
# =========================

recommended_movies = movies[movies["movie_id"].isin(top_movies.index)]

print("\nRecommended Movies:")
print(recommended_movies[["movie_id", "title"]])

# =========================
# 7. EVALUATION: PRECISION@K
# =========================

def precision_at_k(user_id, k=5):
    # get user's ratings
    user_data = ratings[ratings["user_id"] == user_id]

    # split user's ratings
    test_movies = user_data.sample(frac=0.2, random_state=42)
    train_movies = user_data.drop(test_movies.index)

    # create FULL dataset but replace this user's data
    train_ratings = ratings.copy()

    # remove original user data
    train_ratings = train_ratings[train_ratings["user_id"] != user_id]

    # add back only TRAIN part
    train_ratings = pd.concat([train_ratings, train_movies])

    # build matrix
    temp_matrix = train_ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

    # train model
    temp_model = NearestNeighbors(metric="cosine", algorithm="brute")
    temp_model.fit(temp_matrix)

    # user vector
    user_vector = temp_matrix.loc[user_id].values.reshape(1, -1)

    # adjust neighbors safely
    n_neighbors = min(5, len(temp_matrix))

    distances, indices = temp_model.kneighbors(user_vector, n_neighbors=n_neighbors)

    similar_users = indices.flatten()[1:]
    user_ids = temp_matrix.index.tolist()
    similar_user_ids = [user_ids[i] for i in similar_users]

    similarity_scores = 1 - distances.flatten()[1:]

    weighted_sum = np.zeros(temp_matrix.shape[1])
    similarity_sum = np.zeros(temp_matrix.shape[1])

    for i, sim in zip(similar_user_ids, similarity_scores):
        weighted_sum += sim * temp_matrix.loc[i].values
        similarity_sum += sim

    scores = weighted_sum / (similarity_sum + 1e-8)
    scores_series = pd.Series(scores, index=temp_matrix.columns)

    # remove watched (TRAIN ONLY)
    watched = train_movies["movie_id"].tolist()
    scores_series = scores_series.drop(watched, errors="ignore")

    recommended = scores_series.sort_values(ascending=False).head(k).index.tolist()
    actual = test_movies["movie_id"].tolist()

    hits = len(set(recommended) & set(actual))
    precision = hits / k

    return precision


precision = precision_at_k(user_id=1, k=5)
print("\nPrecision@5:", precision)
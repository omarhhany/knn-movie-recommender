# 🎬 AI Movie Recommender System

A machine learning-based movie recommendation system using **K-Nearest Neighbors (KNN)** and **collaborative filtering**, deployed as an interactive web app with **Streamlit**.

---

## 🚀 Live Demo
👉 https://knn-movie-recommender-rre6tzw6jczh3kewedmbnf.streamlit.app/

---

## 📌 Features
- Personalized movie recommendations based on user similarity
- KNN model using cosine similarity
- Genre-based filtering
- Movie posters fetched using TMDB API
- Interactive UI built with Streamlit
- Model evaluation using Precision@K

---

## 🧠 How It Works
1. Build a user-movie rating matrix from MovieLens dataset  
2. Use KNN to find similar users  
3. Aggregate weighted ratings  
4. Recommend unseen movies with highest predicted scores  

---

## 🛠 Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn (KNN)  
- Streamlit  
- TMDB API  

---

## 📊 Sample Output
- Top recommended movies with scores  
- Similar users displayed  
- Movie posters for better user experience  

---

## 📁 Dataset
MovieLens 100K dataset:  
https://grouplens.org/datasets/movielens/100k/

---

## ⚙️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
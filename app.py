####################################################
####################################################

################  Getting started ##################

# 1. Install the streamlit library
## pip install streamlit

# 2. Test that the installation worked
## streamlit hello

# 3. Run the app from the command line:
## streamlit run app.py

####################################################
####################################################



### Libraries and data

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")



### Settings

st. set_page_config(layout="wide")
# st.snow() # some nice snow effects



### Layout

# You can introduce a banner here
# st.image("weblink-to-picture-or-banner")


st.sidebar.title("""
Welcome back, `username123`
Looking for something to watch?
 """)

st.write("# Welcome back!")
st.write("#")
st.write("#")

st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")
st.sidebar.write("#")

# Some funny cinema gif
gif_url = "https://media.giphy.com/media/pUeXcg80cO8I8/giphy.gif"
st.sidebar.image(gif_url, use_column_width=False)



### Recommender functions

def recommend_most_popular_movies(
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        n_movies: int):

    # Popularity metrics dataframe
    popularity_metrics = ratings_df.groupby("movieId").agg({
        "rating": "mean",
        "movieId": "count"
    })
    popularity_metrics.columns = ['avg_rating', 'number_of_ratings']

    # Combined metric
    popularity_metrics["combined_metric"] = (
        (2 * popularity_metrics["avg_rating"]) + (1 * popularity_metrics["number_of_ratings"]))

    # Select most popular movies
    top_n_movie_IDs = popularity_metrics.reset_index().nlargest(n_movies, "combined_metric")["movieId"]
    top_n_movies = movies_df.loc[movies_df["movieId"].isin(top_n_movie_IDs), ["title"]].drop_duplicates()
    top_n_movies = top_n_movies.reset_index(drop=True).rename(columns={"title": "Top 10 Movies"})

    return top_n_movies


def recommend_similar_movies(
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        movie_id: int,
        top_n_movies: int,
        no_of_users_threshold: int):

    # User-item matrix
    user_movie_matrix = pd.pivot_table(
        data=ratings_df,
        values='rating',
        index='userId',
        columns='movieId',
        fill_value=0)
    # (Cosine) Similarity matrix
    movie_similarity_matrix = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        columns=user_movie_matrix.columns,
        index=user_movie_matrix.columns)
    
    # Similarities for selected movie
    movie_similarities = pd.DataFrame(movie_similarity_matrix[movie_id])
    movie_similarities = (
        movie_similarities.loc[movie_similarities.index != movie_id, :]
        .rename(columns={movie_id: "movie_similarities"})
        .sort_values("movie_similarities", ascending=False))
    
    # Filter for number of users who rated both movies
    no_of_users_rated_both_movies = [
        sum((user_movie_matrix[movie_id] > 0) & 
            (user_movie_matrix[isbn] > 0)) for isbn in movie_similarities.index]
    movie_similarities["users_who_rated_both_movies"] = no_of_users_rated_both_movies
    movie_similarities = movie_similarities[
        movie_similarities["users_who_rated_both_movies"] > no_of_users_threshold]
    
    # Select top n similar movies
    movie_cols = ["title", "genres"]
    similar_movies = (
        movie_similarities
        .head(top_n_movies)
        .reset_index()
        .merge(movies_df.drop_duplicates(subset='movieId'),
                on='movieId',
                how='left')
        [movie_cols + ["movie_similarities", "users_who_rated_both_movies"]])
    
    similar_movies = similar_movies.rename(columns={"title": "Top 10 Similar Movies"})["Top 10 Similar Movies"]

    return similar_movies


def recommend_what_others_like(
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_id: int, 
        top_n: int):
    
    # Define reader and load data
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    # Filter the testset to include only rows for the specified user
    filtered_testset = [row for row in testset if row[0] == user_id]

    # Create and train SVD model
    model = SVD(
        n_factors=150, n_epochs=30, 
        lr_all=0.01, reg_all=0.1, 
        random_state=42)
    model.fit(trainset)

    # Make predictions on the filtered test set
    predictions = model.test(filtered_testset)
    predictions_df = pd.DataFrame(predictions, columns=['uid', 'iid', 'r_ui', 'est', 'details'])
    # Return the top n recommended movies
    movies = predictions_df.nlargest(top_n, 'est') # 'est' => estimated rating

    # Get the movie names
    movie_IDs = movies["iid"].tolist()
    movie_names = movies_df.loc[
        movies_df["movieId"].isin(movie_IDs), ["movieId", "title"]]
    # Merge the predictions
    movies_final = movies.merge(
        movie_names, 
        how="left", left_on="iid", right_on="movieId")

    return movies_final["title"]


def recommend_random_movies(movies_df: pd.DataFrame, n_movies: int):

    print("Randomly recommended movies:\n")

    # Shuffle movie IDs randomly
    movie_ids = movies_df["movieId"].unique()
    np.random.shuffle(movie_ids)

    # Select top N movies from shuffled list
    top_n_movie_IDs = movie_ids[:n_movies]

    # Retrieve movie details
    movie_cols = ["title", "genres"]
    top_n_movies = movies_df.loc[movies_df["movieId"].isin(top_n_movie_IDs), movie_cols].drop_duplicates()
    top_n_movies.insert(0, column="ranking", value=range(1, n_movies+1, 1))

    return top_n_movies["title"]



### Popularity recommender

st.write("#### Trending...")

# Get popular movies
top10_popular_movies = recommend_most_popular_movies(ratings, movies, 10)

# Prepare some columns
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
columns = [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]

# Print movie posters
col1.image("https://cdn.moviestillsdb.com/storage/posters/a8/112573_100.jpg")
col2.image("https://cdn.moviestillsdb.com/storage/posters/51/76759_100.jpg")
col3.image("https://cdn.moviestillsdb.com/storage/posters/71/110912_100.jpg")
col4.image("https://cdn.moviestillsdb.com/storage/posters/e2/111161_100.jpg")
col5.image("https://cdn.moviestillsdb.com/storage/posters/88/109830_100.jpg")
col6.image("https://cdn.moviestillsdb.com/storage/posters/bc/107290_100.jpg")
col7.image("https://cdn.moviestillsdb.com/storage/posters/a1/108052_100.jpg")
col8.image("https://cdn.moviestillsdb.com/storage/posters/81/103064_100.jpg")
col9.image("https://cdn.moviestillsdb.com/storage/posters/a3/102926_100.jpg")
col10.image("https://cdn.moviestillsdb.com/storage/posters/25/133093_100.jpg")

# Print one movie per column
for count, col in enumerate(columns):
    col.write(top10_popular_movies.iloc[count, 0])


st.write("#")



### Item-based recommender

st.write("#### Because you liked...")

my_expander = st.expander("Tap to Select a Movie")
selected_movie = my_expander.selectbox(" ", movies["title"])
selected_movieId = int(movies.loc[movies["title"] == selected_movie, "movieId"])


if my_expander.button("Recommend"):
    st.text("Here are few Recommendations..")
    st.write("#")

    top10_similar_movies = recommend_similar_movies(
        ratings, movies, 
        movie_id=selected_movieId, 
        top_n_movies=10, 
        no_of_users_threshold=5)

    # Prepare some columns
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
    columns = [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]

    # Print one movie per column
    try:
        for count, col in enumerate(columns):
            col.write(top10_similar_movies.iloc[count])
    except IndexError:
        # Handle out-of-bounds index gracefully
        print("Error: Index out of bounds")


st.write("#")



### User-based recommender

st.write("#### What others like...")


# Recommend what similar users like
what_others_like = recommend_what_others_like(
    ratings, movies, 
    user_id=603, top_n=10)

# Prepare some columns
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
columns = [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10]

# Print movie posters
col1.image("https://cdn.moviestillsdb.com/storage/posters/ba/34583_100.jpg")
col2.image("https://cdn.moviestillsdb.com/storage/posters/6c/112682_100.jpg")
col3.image("https://cdn.moviestillsdb.com/storage/posters/0b/73486_100.jpg")
col4.image("https://cdn.moviestillsdb.com/storage/posters/71/110912_100.jpg")
col5.image("https://cdn.moviestillsdb.com/storage/posters/c5/102494_100.jpg")
col6.image("https://cdn.moviestillsdb.com/storage/posters/bf/95765_100.jpg")
col7.image("https://cdn.moviestillsdb.com/storage/posters/94/93779_100.jpg")
col8.image("https://cdn.moviestillsdb.com/storage/posters/d6/120202_100.jpg")
col9.image("https://cdn.moviestillsdb.com/storage/posters/00/42192_100.jpg")
col10.image("https://cdn.moviestillsdb.com/storage/posters/bd/71562_100.jpg")


# Print one movie per column
for count, col in enumerate(columns):
    col.write(what_others_like.iloc[count])


st.write("#")



### Random recommender

st.write("#### Something random...")
get_random_movies = st.button("Random stuff!")

if get_random_movies:
    # Get movies
    random_movies = recommend_random_movies(movies, n_movies=5)
    random_movies.tolist()
    # Print them in a bullet list
    for movie in random_movies:
        st.write("-", movie)
    # Bullet list style
    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)

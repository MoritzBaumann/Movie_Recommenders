# Movie_Recommenders

Did you ever wonder how the recommendations on Netflix work? 
Let's find out together in this **recommendation system** project. 🎥

I here create three recommendation systems that suggest movies to watch. 
I manually develop both a popularity-based and an item-based recommender, and subsequently use sklearn's `surprise` library to create a user-based recommender. 
These recommenders are then implemented in a `streamlit` app I create.

## Content
- *Movie_Recommenders.ipynb*
- *app.py*

The jupyter notebook contains the code for the three recommenders. 
I then transferred them to the `app.py` module, which creates the streamlit App.
Instructions on how to run the app can be found in the `app.py` module.

## Dataset
The used the movie ratings dataset from [MovieLens](http://movielens.org). It includes 100k 5-star ratings from 610 users of almost 10k different movies. It is publicly available for download at <http://grouplens.org/datasets/>. Credit goes to:
> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

## Context
This is a learning project, in which I practice building recommenders and try to understand their fundamental concepts. 
It was carried out in the context of a 4.5 month-long Data Science bootcamp with WBS Coding School. 
Many thanks to WBS Coding School and to my instructors for the guidance.

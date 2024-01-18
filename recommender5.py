#importing modules
import pandas as pd
import numpy as np
from surprise import KNNBasic, accuracy


# recommender function
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
modelling_data = pd.merge(movies, ratings, on = "movieId")
#dropping timestamp 
modelling_data.drop(columns= "timestamp", axis= 1, inplace= True)




def neural_recommender(user_id, n, svd_model, model):
    # Get movies not rated by the user
    to_recommend = set(movies['movieId'].unique()) - set(movies[movies['userId'] == user_id]['movieId'].unique())

    # Get SVD predictions for movies to recommend
    svd_preds = [svd_model.predict(user_id, movie_id).est for movie_id in to_recommend]

    # Create a DataFrame with SVD predictions
    df_with_svd = pd.DataFrame({'userId': [user_id] * len(to_recommend), 'movieId': list(to_recommend), 'svd_preds': svd_preds})

    # Use the neural network to predict ratings
    nn_preds = model.predict([df_with_svd['userId'], df_with_svd['movieId'], df_with_svd['svd_preds']])

    # Combine movie IDs with their predicted ratings
    recommendations = pd.DataFrame({'movieId': list(to_recommend), 'predicted_rating': nn_preds.flatten()})

    # Get the top N recommendations
    top_recommendations = recommendations.nlargest(n, 'predicted_rating')

    # Merge with movie information to get titles and genres
    top_recommendations = pd.merge(top_recommendations, modelling_data[['movieId', 'title', 'genres']], on='movieId', how='left')

    return top_recommendations[['movieId', 'title', 'genres', 'predicted_rating']]
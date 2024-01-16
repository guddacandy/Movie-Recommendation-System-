#importing modules
import pandas as pd

# recommender function
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
modelling_data = pd.merge(movies, ratings, on = "movieId")
#dropping timestamp 
modelling_data.drop(columns= "timestamp", axis= 1, inplace= True)


def recommender_system(userId, n, df, model):
    #unique user
    user_Id = userId
    #N number of recommendations
    N = n
    #list of movies not rated by the user
    to_recommend  = set(df["movieId"].unique()) - set(df[df["userId"] == user_Id]["movieId"].unique())
    #getting predictions for movies to recommend
    preds_to_user = [model.predict(user_Id, movieId) for  movieId in to_recommend]
    #get top N recommendation
    top_N_recommendations = sorted(preds_to_user, key=lambda x: x.est, reverse=True)[:N]
    # Display the top N recommendations
    for recommendation in top_N_recommendations:
        movie_info = modelling_data[modelling_data['movieId'] == recommendation.iid]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            print(f"MovieId: {recommendation.iid}, Title: {title}, Genres: {genres}, Estimated Rating: {recommendation.est}")


#neural networks recommender function
def neural_recommender(user_id, n, df, svd_model, neural_network_model):
    # Get movies not rated by the user
    to_recommend = set(df['movieId'].unique()) - set(df[df['userId'] == user_id]['movieId'].unique())

    # Get SVD predictions for movies to recommend
    svd_preds = [svd_model.predict(user_id, movie_id).est for movie_id in to_recommend]

    # Create a DataFrame with SVD predictions
    df_with_svd = pd.DataFrame({'userId': [user_id] * len(to_recommend), 'movieId': list(to_recommend), 'svd_preds': svd_preds})

    # Use the neural network to predict ratings
    nn_preds = neural_network_model.predict([df_with_svd['userId'], df_with_svd['movieId'], df_with_svd['svd_preds']])

    # Combine movie IDs with their predicted ratings
    recommendations = pd.DataFrame({'movieId': list(to_recommend), 'predicted_rating': nn_preds.flatten()})

    # Get the top N recommendations
    top_recommendations = recommendations.nlargest(n, 'predicted_rating')

    # Merge with movie information to get titles and genres
    top_recommendations = pd.merge(top_recommendations, df[['movieId', 'title', 'genres']], on='movieId', how='left')

    return top_recommendations[['movieId', 'title', 'genres', 'predicted_rating']]

# Example usage:


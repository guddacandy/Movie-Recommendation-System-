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
            
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


def readable_date(column, df):
    df["timestamp_x"] = pd.to_datetime(df["timestamp_x"])
    df["timestamp_y"] = pd.to_datetime(df["timestamp_y"])
    #using dt.strftime() method since it is pandas timestamp format
    df["rating_timestamp"] = df["timestamp_x"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["tag_timestamp"] = df["timestamp_y"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.drop(columns=["timestamp_x", "timestamp_y"], axis= 1).head()

### EDA
#writting helper function to help us make x-axis countplots in our EDA process
def sns_xcount(column , data):
    sns.countplot(x = column, data = data, hue= column)
    plt.title(f"{column} count in our data set")
    plt.show();

#writting helper function to help us make y-axis countplots in our EDA process
def sns_ycount(column , data):
    sns.countplot(y = column, data = data)
    plt.title(f"{column} count in our data set")
    plt.show();
    
    
def plotting_trends(data, column):
    #set rating_timestamp as the index
    #set rating_timestamp as the index
    trends_data = data.set_index("column")
    #yearly resampling 
    yearly_trend = trends_data.resample("Y").size()
    #plotting the trend
    plt.figure(figsize= (15, 6))
    yearly_trend.plot(marker = "o", linestyle = "--", color = "red")
    plt.title("The trend of ratings yearly over the years")
    plt.xlabel("rating_timestamp")
    plt.ylabel("count")
    plt.show();
    
# recommender function
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
modelling_data = pd.merge(movies, ratings, on = "movieId")
#dropping timestamp 
modelling_data.drop(columns= "timestamp", axis= 1, inplace= True)


# def recommender_system(userId, n, df, model):
#     #unique user
#     user_Id = userId
#     #N number of recommendations
#     N = n
#     #list of movies not rated by the user
#     to_recommend  = set(df["movieId"].unique()) - set(df[df["userId"] == user_Id]["movieId"].unique())
#     #getting predictions for movies to recommend
#     preds_to_user = [model.predict(user_Id, movieId) for  movieId in to_recommend]
#     #get top N recommendation
#     top_N_recommendations = sorted(preds_to_user, key=lambda x: x.est, reverse=True)[:N]
#     # Display the top N recommendations
#     for recommendation in top_N_recommendations:
#         movie_info = modelling_data[modelling_data['movieId'] == recommendation.iid]
#         if not movie_info.empty:
#             title = movie_info['title'].values[0]
#             genres = movie_info['genres'].values[0]
#             print(f"MovieId: {recommendation.iid}, Title: {title}, Genres: {genres}, Estimated Rating: {recommendation.est}")
            
            
            
def recommender_system(user_id, n_recommendations, ratings_df, collaborative_model):
    # List of movies not rated by the user
    to_recommend = set(ratings_df["movieId"].unique()) - set(ratings_df[ratings_df["userId"] == user_id]["movieId"].unique())

    # Getting predictions for movies to recommend
    preds_to_user = [collaborative_model.predict(user_id, movie_id) for movie_id in to_recommend]

    # Get top N recommendations
    top_n_recommendations = sorted(preds_to_user, key=lambda x: x.est, reverse=True)[:n_recommendations]

    # Display the top N recommendations
    for recommendation in top_n_recommendations:
        movie_info = modelling_data[modelling_data['movieId'] == recommendation.iid]
        if not movie_info.empty:
            title = movie_info['title'].values[0]
            genres = movie_info['genres'].values[0]
            print(f"MovieId: {recommendation.iid}, Title: {title}, Genres: {genres}, Estimated Rating: {recommendation.est}")
        else:
            print(f"MovieId: {recommendation.iid}, No information available, Estimated Rating: {recommendation.est}")

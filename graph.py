import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt



tweets = pd.read_csv('scored_tweets_vectorization.csv', lineterminator='\n')
crimes = pd.read_csv('lgbt_crimes.csv')
tweets.dropna(inplace = True)



tweets["sentiment_score"] = np.where(tweets["sentiment"] == "Positive", 1, 0)

avg_sentiment = tweets.groupby(['Date'])['sentiment_score'].mean()

#crimes["avg_sentiment"] = crimes["Date"].apply(get_score)

#print(tweets.groupby(['Date'])['sentiment_score'].mean())


# avg_sentiment.to_csv("avg_sent_per_day.csv")















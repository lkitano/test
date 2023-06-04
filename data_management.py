import pandas as pd
tweets = pd.read_csv('scored_tweets_vectorization.csv', lineterminator='\n')
crimes = pd.read_csv('lgbt_crimes.csv')
crimes_per_day = pd.read_csv("hate_crimes_per_day.csv")
crimes_per_month = pd.read_csv("crimes_per_month.csv")
avg_sentiment = pd.read_csv('avg_sent_per_day.csv')

tweets.dropna(inplace = True)
tweets['year-month'] = tweets.apply(lambda x: x["Date"].split("-")[0] +"-" +
                                              x["Date"].split("-")[1], axis=1)
crimes['year-month'] = crimes.apply(lambda x: x["incident_date"].split("-")[0] +"-" +
                                              x["incident_date"].split("-")[1], axis=1)

# These Lines calcualte the amount of crimes per day and per month, respectively
#crimes_per_day = crimes.groupby(['incident_date'])['incident_date'].count()
#crimes_per_month = crimes.groupby(['year-month'], as_index=False)['year-month'].count()
#crimes_per_month.rename(columns={crimes_per_month.columns[1]: "n" }, inplace = True)
#print(crimes_per_month)

def get_score(date):
    if (avg_sentiment['Date'] == date).any():
        return avg_sentiment.loc[avg_sentiment['Date'] == date, "sentiment_score"].iloc[0]
    else:
        return None

def get_num_crimes_day(date):
    if (crimes_per_day['incident_date'] == date).any():
        return crimes_per_day.loc[crimes_per_day['incident_date'] == date, "n"].iloc[0]
    else:
        return None

def get_num_crimes_month(month):
    if (crimes_per_month['year-month'] == month).any():
        return crimes_per_month.loc[crimes_per_month['year-month'] == month, "n"].iloc[0]
    else:
        return None



# saves as csv files fpr later
#crimes_per_month.to_csv("crimes_per_month.csv")
#crimes_per_day.to_csv("hate_crimes_per_day.csv")

# calculates avg sentiment for each day and save it as a table
#crimes["avg_sentiment"] = crimes["incident_date"].apply(get_score)
#crimes.to_csv("lgbt_crimes_sentiment")

# combine tweets and crime per month tables
#tweets["num_crimes_per_month"] = tweets["year-month"].apply(get_num_crimes_month)
#tweets.to_csv("tweets_num_crimes_per_month.csv")



# combine tweets and crimes per day tables
#tweets["num_crimes_per_day"] = tweets["Date"].apply(get_num_crimes_day)
#tweets.to_csv("tweets_num_crimes.csv")




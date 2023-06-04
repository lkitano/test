# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
df = pd.read_csv('tweetsLGBT.csv')
df["Date"] = df.apply(lambda x: x["create_at"].split("T")[0], axis=1)
df["Year"] = df.apply(lambda x: x["create_at"].split("-")[0], axis=1)

#df["date"] = [df["create_at"].str.split("T")][0]

#print(df)
#df2 = df.groupby('Date').count().sort_values(by = ["Date"], ascending=True)
df = df.dropna()
df2 = df.groupby('Date').apply(lambda x: x.sample(n = min(50, x.shape[0]), random_state = 170).reset_index(drop=True))
#df2.to_csv("sampled_tweets.csv")

#df3 = df.groupby('Year').apply(lambda x: x.sample(n = min(400, x.shape[0])).reset_index(drop=True))
#df3.to_csv("bot_training.csv")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

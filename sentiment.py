import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import seaborn as sns
import nltk
from wordcloud import WordCloud

import matplotlib.pyplot as plt

sample1 = "I really like to eat bean burritos. They are my favorite!"
sample2 = "I hate pineapple on pizza. Whoever eats pineapple on pizza is a loser"

tweets = pd.read_csv("sampled_tweets.csv", lineterminator='\n')
print(tweets.shape)


def removeNoise(text):
    text = re.sub(r'#', '', text) #removes tags
    text = re.sub(r'@[A-Za-z0-9]+', "", text) #removes mentions
    text = re.sub(r'https?:\/\/t.co\/[A-Za-z0-9]+', '', text) #removes links
    return text

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getSentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

tweets["polarity"] = tweets["text"].apply(removeNoise).apply(getPolarity)
tweets["subjectivity"] = tweets["text"].apply(removeNoise).apply(getSubjectivity)
tweets["sentiment"] = tweets["polarity"].apply(getSentiment)

tweets.polarity.plot.kde(bw_method=0.5)
plt.title('polarity')
plt.show()

tweets.to_csv("scored_tweets.csv")


'''
allwords = ' '.join([twt for twt in tweets['text'].apply(removeNoise)])
wc = WordCloud(width = 1000, height = 600, random_state=21).generate(allwords)
plt.imshow(wc, interpolation= 'bilinear')
plt.axis('off')
plt.show()
'''

#tweets.to_csv()






import pandas as pd
import nltk
import ssl
import re
import random
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import classify
from nltk import NaiveBayesClassifier


'''
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download('twitter_samples')
#nltk.download('stopwords')

'''

from nltk.corpus import twitter_samples
from nltk.corpus import stopwords


alphaNum = r'[a-zA-Z0-9]+'
tokenizer = RegexpTokenizer(alphaNum)
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
stop_words = set(stopwords.words('english'))

tweets = pd.read_csv("sampled_tweets.csv", lineterminator='\n')


def removeNoise(text):
    text = re.sub(r'#', '', str(text)) #removes tags
    text = re.sub(r'@[A-Za-z0-9]+', "", str(text)) #removes mentions
    text = re.sub(r'https?:\/\/t.co\/[A-Za-z0-9]+', '', str(text)) #removes links
    return text


def tokenize_tweets(tweets):
    tokenized_tweets = [tokenizer.tokenize(removeNoise(tweet)) for tweet in tweets]
    return tokenized_tweets

def stem_tweets(tweets):
    return [[stemmer.stem(token) for token in tweet] for tweet in tweets]

def lemmatize_tweets(tweets):
    return [[lemmatizer.lemmatize(token) for token in tweet] for tweet in tweets]

def remove_Stopwords(tweets):
    return [[token for token in tweet if token.lower() not in stop_words] for tweet in tweets]


def prepare_tweets(tweets):
    tokenized_tweets = tokenize_tweets(tweets)
    lemmatized_tweets = lemmatize_tweets(tokenized_tweets)
    #lemmatized_tweets = remove_Stopwords(lemmatized_tweets)
    return lemmatized_tweets


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)




prepared_twts = prepare_tweets(text)
prepared_pos_twts = prepare_tweets(pos_tweets)
prepared_neg_twts = prepare_tweets(neg_tweets)


positive_tokens_for_model = get_tweets_for_model(prepared_pos_twts)
negative_tokens_for_model = get_tweets_for_model(prepared_neg_twts)


positive_dataset = [(tweet_dict, "Positive")
                    for tweet_dict in positive_tokens_for_model]

negative_dataset = [(tweet_dict, "Negative")
                    for tweet_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

from nltk.tokenize import word_tokenize

custom_tweets = ["I hate elon musk"]

custom_tokens = prepare_tweets(custom_tweets)[0]
print(custom_tweets, classifier.classify(dict([token, True] for token in custom_tokens)))

def get_sentiment(text):
    text = prepare_tweets([text])[0]
    return classifier.classify(dict([token, True] for token in text))


tweets["sentiment"] = tweets["text"].apply(get_sentiment)

tweets.to_csv("scored_tweets2.csv")









#tokens = tokenizer.tokenize()

#stemmed_tokens = [stemmer.stem(token) for token in tokens]




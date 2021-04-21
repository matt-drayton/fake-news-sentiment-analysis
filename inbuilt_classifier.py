import pickle
from nltk.corpus import twitter_samples
from nltk import pos_tag
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string, random
import json
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle


with open("real1.pkl", "rb") as file:
    real_tweets = pickle.load(file)

with open("fake1.pkl", "rb") as file:
    fake_tweets = pickle.load(file)

classifier = SentimentIntensityAnalyzer()
random.shuffle(real_tweets)
random.shuffle(fake_tweets)

real_output = []
fake_output = []

for tweet in real_tweets:
    probs = classifier.polarity_scores(tweet["tweet"])
    output = {
        "tweet": tweet["tweet"],
        "article_title": tweet["article_title"],
        "positive_score": probs["pos"],
        "negative_score": probs["neg"],
        "neutral_score": probs["neu"],
    }
    real_output.append(output)

for tweet in fake_tweets:
    probs = classifier.polarity_scores(tweet["tweet"])
    classifier.sentiment_valence
    output = {
        "tweet": tweet["tweet"],
        "article_title": tweet["article_title"],
        "positive_score": probs["pos"],
        "negative_score": probs["neg"],
        "neutral_score": probs["neu"],
    }
    fake_output.append(output)

with open("classifications_real.json", "w") as file:
    json.dump(real_output, file)

with open("classifications_fake.json", "w") as file:
    json.dump(fake_output, file)

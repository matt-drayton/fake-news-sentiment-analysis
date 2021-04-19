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
with open("real1.pkl", 'rb') as file:
    tweets = pickle.load(file)


classifier = SentimentIntensityAnalyzer()
random.shuffle(tweets)
output = []

for tweet in tweets:
    probs = classifier.polarity_scores(tweet['tweet'])  
    output = {
        "tweet": tweet['tweet'],
        "article_title": tweet['article_title'],
        "positive_score": probs['pos'],
        "negative_score": probs['neg'],
        "neutral_score": probs['neu'],
    }

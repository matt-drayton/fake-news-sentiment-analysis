import pickle
from nltk.corpus import twitter_samples
from nltk import pos_tag
from nltk import classify
from nltk import NaiveBayesClassifier
import nltk.classify
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string, random
import json
from tqdm import tqdm
import pandas as pd

def preprocess(tweets):
    output = []
    for tweet in tweets:
        new_tweet = tweet
        new_tweet = re.sub(r"http\S+", "", new_tweet) # Remove URLs
        new_tweet = re.sub(r"@[A-Za-z0-9_-]*", "", new_tweet) # Remove Handles
        new_tweet = re.sub(r"RT+", "", new_tweet) # Remove RT
        new_tweet = re.sub(r"\\.+?(\b)", "", new_tweet) # Remove newlines, unicodes, etc
        new_tweet = re.sub(r"\bvia\b", "", new_tweet) # Remove 'via' from tweet
        new_tweet = re.sub(r"#", "", new_tweet) # Remove # signs
        output.append(word_tokenize(new_tweet))
    return output

def load_training_data():
    csv = pd.read_csv("training.1600000.processed.noemoticon.csv")
    csv.columns = ['target', 'id', 'date', 'query', 'user', 'text']
    positives = csv.loc[csv['target'] == 4]['text'].tolist()
    negatives = csv.loc[csv['target'] == 0]['text'].tolist()
    neutrals = csv.loc[csv['target'] == 2]['text'].tolist()
    return preprocess(positives), preprocess(negatives), preprocess(neutrals)



def remove_noise(tweet_tokens):
    stop_words = stopwords.words('english')

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


positive_training_tweets, negative_training_tweets, neutral_training_tweets = load_training_data()


# Clean up training / raw data
positive_training_tweets = [remove_noise(tweet) for tweet in positive_training_tweets]
negative_training_tweets = [remove_noise(tweet) for tweet in negative_training_tweets]
neutral_training_tweets = [remove_noise(tweet) for tweet in neutral_training_tweets]

with open("training_positive.json", "w") as file:
    dump = json.dumps(positive_training_tweets, indent=4)
    file.write(dump)

with open("training_negative.json", "w") as file:
    dump = json.dumps(negative_training_tweets, indent=4)
    file.write(dump)

with open("training_neutral.json", "w") as file:
    dump = json.dumps(neutral_training_tweets, indent=4)
    file.write(dump)
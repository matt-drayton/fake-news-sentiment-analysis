import pickle
from nltk.corpus import twitter_samples
from nltk import pos_tag
from nltk import classify
from nltk import NaiveBayesClassifier
import nltk.classify
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string, random
import json
from tqdm import tqdm
import pandas as pd
import random


def load_training_data():
    with open("training_positive.json", "r") as file:
        positives = json.load(file)
    with open("training_negative.json", "r") as file:
        negatives = json.load(file)
    return positives, negatives


# Load tweet examples to train model
positive_training_tweets, negative_training_tweets = load_training_data()


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


positive_training_tweets = get_tweets_for_model(positive_training_tweets)
negative_training_tweets = get_tweets_for_model(negative_training_tweets)


# Model Training
positive_dataset = [(tweet, "Positive") for tweet in positive_training_tweets]
negative_dataset = [(tweet, "Negative") for tweet in negative_training_tweets]

total_dataset = positive_dataset + negative_dataset
random.shuffle(total_dataset)

# Naive Bayes
naive_train_data = total_dataset[: int(len(total_dataset) * 0.7)]
naive_test_data = total_dataset[int(len(total_dataset) * 0.7) :]

# Logistic
logistic_train_data = total_dataset[:10000]
logistic_test_data = total_dataset[10000:]

# SVC
svc_train_data = total_dataset[:8000]
svc_test_data = total_dataset[8000:]

classifiers = {
    "naive": NaiveBayesClassifier.train(naive_train_data),
    "logistic": classify.SklearnClassifier(LogisticRegression()),
    "svc": classify.SklearnClassifier(SVC(kernel="linear", probability=True)),
}

data = {
    "naive": (naive_train_data, naive_test_data),
    "logistic": (logistic_train_data, logistic_test_data),
    "svc": (svc_train_data, svc_test_data),
}

for name, classifier in classifiers.items():
    print("Training model " + name)
    train, test = data[name]
    if name != "naive":
        classifier.train(train)
    accuracy = classify.accuracy(classifier, test)
    print(f"Exporting classifier {name} with accuracy {accuracy}.")
    with open(f"{name}.pkl", "wb") as file:
        pickle.dump(classifier, file)


print("Finished Executing.")

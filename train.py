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
    with open("training_positive.json", 'r') as file:
        positives = json.load(file) 
    with open("training_negative.json", 'r') as file:
        negatives = json.load(file)
    return positives, negatives



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


with open("real1.pkl", 'rb') as file:
    fake_tweets = pickle.load(file)

# Load tweet examples to train model
positive_training_tweets, negative_training_tweets = load_training_data()


# fake_tweets_unmodified = [tweet['tweet'] for tweet in tqdm(fake_tweets)]
# article_sources = [tweet['article_title'] for tweet in tqdm(fake_tweets)]
# fake_tweets = [word_tokenize(tweet['tweet']) for tweet in tqdm(fake_tweets)]

# # Clean up training / raw data
# fake_tweets = [remove_noise(tweet) for tweet in tqdm(fake_tweets)]

# Load in dict format
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
        
positive_training_tweets = get_tweets_for_model(positive_training_tweets)
negative_training_tweets = get_tweets_for_model(negative_training_tweets)

# fake_tweets = get_tweets_for_model(fake_tweets)


# Model Training
positive_dataset = [(tweet, "Positive") for tweet in positive_training_tweets]
negative_dataset = [(tweet, "Negative") for tweet in negative_training_tweets]

total_dataset = positive_dataset + negative_dataset
random.shuffle(total_dataset)

train_data = total_dataset[:7000]
test_data = total_dataset[7000:]

# classifier = NaiveBayesClassifier.train(train_data)
# classifier = classify.SklearnClassifier(SVC(kernel='linear', probability=True))
classifier = classify.SklearnClassifier(LogisticRegression())
classifier.train(train_data)
# accuracy = classify.accuracy(classifier, test_data)

total_output = []
for fake_tweet, raw_fake_tweet, article_source in tqdm(zip(fake_tweets, fake_tweets_unmodified, article_sources)):
     
    joined_tweet = ' '.join(fake_tweet.keys())
    probs = classifier.prob_classify(fake_tweet)
    output = {
        "tweet": joined_tweet,
        "raw_tweet": raw_fake_tweet,
        "article_title": article_source,
        "classification": classifier.classify(fake_tweet),
        "positive_score": probs.prob("Positive"),
        "negative_score": probs.prob("Negative"),
    }
    total_output.append(output)
        


with open("classifications_real.json", "w") as file:
    dump = json.dumps(total_output, indent=4)
    file.write(dump)

with open("logisticregression.pkl", 'wb') as file:
    pickle.dump(classifier, file)

print("Finished Executing.")
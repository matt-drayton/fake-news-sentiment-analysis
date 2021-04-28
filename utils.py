import re, string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import combinations

def get_substrings(text):
    return [text[x:y] for x, y in combinations(range(len(text)+1), r=2)]

def remove_noise(tweet_tokens):
    stop_words = stopwords.words("english")

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if (
            len(token) > 0
            and token not in string.punctuation
            and token.lower() not in stop_words
        ):
            cleaned_tokens.append(token.lower())
    return cleaned_tokens



def preprocess(tweet):
    new_tweet = tweet
    new_tweet = re.sub(r"http\S+", "", new_tweet)  # Remove URLs
    new_tweet = re.sub(r"@[A-Za-z0-9_-]*", "", new_tweet)  # Remove Handles
    new_tweet = re.sub(r"RT+", "", new_tweet)  # Remove RT
    new_tweet = re.sub(r"\\.+?(\b)", "", new_tweet)  # Remove newlines, unicodes, etc
    new_tweet = new_tweet.encode("ascii", "ignore").decode() # Remove unicode characters missed by regex
    new_tweet = re.sub(r"\bvia\b", "", new_tweet)  # Remove 'via' from tweet
    new_tweet = re.sub(r"#", "", new_tweet)  # Remove # signs
    return new_tweet

def preprocess_bulk(tweets):
    output = []
    for tweet in tweets:
        new_tweet = preprocess(tweet)
        output.append(word_tokenize(new_tweet))
    return output

import re, string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize.casual import casual_tokenize
from itertools import combinations
from datetime import datetime

LOG = True


def log(text):
    """Log text with current timestamp. This method will only print if the LOG boolean, set in the utils
    folder, is True.

    Args:
        text (str): The text to be printed
    """
    if LOG:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f"[{current_time}] {text}")


def get_substrings(text):
    """Returns all possible substrings of a string

    Args:
        text (string)

    Returns:
        List: A list of substrings
    """
    return [text[x:y] for x, y in combinations(range(len(text) + 1), r=2)]


def lemmatize_and_strip(tweet_tokens):
    stop_words = stopwords.words("english")

    cleaned_tokens = []
    # Convert POS tags into feature format
    for token, tag in pos_tag(tweet_tokens):
        if tag.startswith("NN"):
            pos = "n"
        elif tag.startswith("VB"):
            pos = "v"
        else:
            pos = "a"
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        # Exclude tokens that are punctuation, or are stop words
        if (
            len(token) > 0
            # and token not in string.punctuation
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
    new_tweet = new_tweet.encode(
        "ascii", "ignore"
    ).decode()  # Remove unicode characters missed by regex
    new_tweet = re.sub(r"\bvia\b", "", new_tweet)  # Remove 'via' from tweet
    new_tweet = re.sub(r"#", "", new_tweet)  # Remove # signs
    return new_tweet


def preprocess_bulk(tweets):
    output = []
    for tweet in tweets:
        new_tweet = preprocess(tweet)
        output.append(casual_tokenize(new_tweet))
    return output

import pickle
from random import shuffle
import re
from collections import defaultdict
from tqdm import tqdm
from utils import get_substrings, log


def sort_by_substring_percentage(tweet):
    """Finds the percentage of a tweet that is from the article's title

    Args:
        tweet: Tweet JSON containing text and article's title

    Returns:
        float: The percentage of the tweet that is a substring of the article's title
    """
    title = tweet["article_title"]
    text = tweet["tweet"]

    # Create list of all substrings in descending order of length
    substrings = sorted(get_substrings(title), key=lambda x: len(x), reverse=True)
    for substring in substrings:
        if substring in text:
            return len(substring) / len(text)
        else:
            return 0


def remove_article_title(tweet):
    """Removes the largest substring of an article's title from a tweet 

    Args:
        tweet: Tweet JSON containing text and article's title

    Returns:
        string: Tweet text with the article's title removed from the text
    """
    title = tweet["article_title"].lower()
    tweet["tweet"] = tweet["tweet"].lower()
    substrings = sorted(get_substrings(title), key=lambda x: len(x), reverse=True)
    for substring in substrings:
        if substring in tweet["tweet"]:
            return tweet["tweet"].replace(substring, "").strip()


def clean_tweets(tweets):
    """Clean all tweets, removing article's titles from their text

    Args:
        tweets (List): A list of tweet JSONs   

    Returns:
        tweets: A list of cleaned tweet JSONs
    """
    output = []
    sorted_tweets = sorted(tweets, key=sort_by_substring_percentage, reverse=True)
    for tweet in tqdm(sorted_tweets):
        tweet["tweet"] = remove_article_title(tweet)
        try:
            # Only keep tweet if, having removed the title, it has a length of at least 2 (allows for emoticons)
            if len(tweet["tweet"]) > 2 and tweet["tweet"].isascii():
                output.append(tweet)
        except:
            continue
    return output

if __name__ == '__main__':
    log("Loading tweets")
    with open("fake.pkl", "rb") as file:
        fake_tweets = pickle.load(file)

    with open("real.pkl", "rb") as file:
        real_tweets = pickle.load(file)

    log("Cleaning tweets")
    export_fake_tweets = clean_tweets(fake_tweets)

    export_real_tweets = clean_tweets(real_tweets)

    log("Exporting cleaned tweets")
    with open("fake_cleaned.pkl", "wb") as file:
        pickle.dump(export_fake_tweets, file)

    with open("real_cleaned.pkl", "wb") as file:
        pickle.dump(export_real_tweets, file)

    log("Exectuion Complete")
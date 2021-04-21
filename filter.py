import pickle
from random import shuffle
import re
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from difflib import SequenceMatcher


def get_substrings(text):
    return [text[x:y] for x, y in combinations(range(len(text) + 1), r=2)]


def sort_by_substring_percentage(tweet):
    title = tweet["article_title"]
    text = tweet["tweet"]
    substrings = sorted(get_substrings(title), key=lambda x: len(x), reverse=True)
    for substring in substrings:
        if substring in text:
            return len(substring) / len(text)
        else:
            return 0


def remove_article_title(tweet):
    title = tweet["article_title"].lower()
    tweet["tweet"] = tweet["tweet"].lower()
    substrings = sorted(get_substrings(title), key=lambda x: len(x), reverse=True)
    for substring in substrings:
        if substring in tweet["tweet"]:
            return tweet["tweet"].replace(substring, "").strip()


def find_threshold(tweets):
    output = []
    sorted_tweets = sorted(tweets, key=sort_by_substring_percentage, reverse=True)
    for tweet in tqdm(sorted_tweets):
        tweet["tweet"] = remove_article_title(tweet)
        try:
            if len(tweet["tweet"]) > 2:
                output.append(tweet)
        except:
            continue
    return output


with open("fake.pkl", "rb") as file:
    fake_tweets = pickle.load(file)

with open("real.pkl", "rb") as file:
    real_tweets = pickle.load(file)

# previously_seen_tweets = defaultdict(lambda: False)
export_fake_tweets = find_threshold(fake_tweets)

export_real_tweets = find_threshold(real_tweets)


# # Process Tweets
# for tweet in fake_tweets:
#     if previously_seen_tweets[tweet['tweet']] is False:
#         previously_seen_tweets[tweet['tweet']] = True
#         export_fake_tweets.append(tweet)

# for tweet in real_tweets:
#     tweet['tweet'] = preprocess(tweet['tweet'])
#     if previously_seen_tweets[tweet['tweet']] is False:
#         previously_seen_tweets[tweet['tweet']] = True
#         export_real_tweets.append(tweet)

with open("fake1.pkl", "wb") as file:
    pickle.dump(export_fake_tweets, file)

with open("real1.pkl", "wb") as file:
    pickle.dump(export_real_tweets, file)

import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# def preprocess_tweets(tweets):
#     processed_tweets = []
#     for tweet in tweets:
#         # First, do my preprocessing
#         tweet = preprocess_tweet(tweet)

#         # Join as a string again for nltk
#         tweet = ' '.join(tweet)

#         # Tokenize using NLTK
#         tweet = word_tokenize(tweet)

#         # POS Tagging
#         tweet = pos_tag(tweet)

#         processed_tweets.append(tweet)

#     return processed_tweets

# def preprocess_tweet(tweet, exclude_hashtags=False):
#     processed_tweet = []
#     # Can definitely be done better with ReGeX but for now...
#     for token in tweet:
#         if token.startswith("@"):
#             continue
#         if token.startswith("http://") or token.startswith("https://"):
#             continue
#         if token.startswith("#") and exclude_hashtags == True:
#             continue
#         if token == "RT":
#             continue

#         token = token.replace('\n', '')
#         token = token.replace('\xa0', '')
#         token = token.replace('&amp;', 'and')


#         processed_tweet.append(token)

#     return processed_tweet

# with open("out.pkl", 'rb') as file:
#     tweets = pickle.load(file)

# tweets = preprocess_tweets(tweets)

with open("processed_out.pkl", "wb") as file:
    pickle.dump(tweets, file)

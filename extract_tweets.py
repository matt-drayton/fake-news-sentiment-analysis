import os
import json
import nltk
from nltk.corpus import twitter_samples
import pickle
from collections import defaultdict
from tqdm import tqdm
import re
BASEURL = os.getcwd()+"/fakenewsanalysis/fakenewsnet_dataset/politifact/"
LOG = False

def log(str):
    if LOG:
        print(str)


def preprocess(tweet):
    new_tweet = tweet
    new_tweet = re.sub(r"http\S+", "", new_tweet) # Remove URLs
    new_tweet = re.sub(r"@[A-Za-z0-9_-]*", "", new_tweet) # Remove Handles
    new_tweet = re.sub(r"RT+", "", new_tweet) # Remove RT
    new_tweet = re.sub(r"\\.+?(\b)", "", new_tweet) # Remove newlines, unicodes, etc
    new_tweet = re.sub(r"\bvia\b", "", new_tweet) # Remove 'via' from tweet
    new_tweet = re.sub(r"#", "", new_tweet) # Remove # signs
    return new_tweet

def load_from_file(file):
    # Optimisation: Defaultdict rather than list: O(n) -> O(1)
    raw_fake_tweets = defaultdict(lambda:False)
    payload_tweets = []
    directory = BASEURL+file
    for article in tqdm(os.listdir(directory)):
        article_path = directory + f"/{article}"
        
        try:
            with open(article_path+"/news content.json") as article_json:
                data = json.load(article_json)
                current_article = data['meta_data']['twitter']['title']
                log(f"Exploring article: {current_article}")
                raw_fake_tweets[current_article] = True
        except:
            log("Article data empty. Ignoring...")
            continue

        if 'tweets' in os.listdir(article_path):
            log("Found relevant tweets:")
            
            for tweet_json in os.listdir(article_path+"/tweets"):
                path = article_path+"/tweets/"+tweet_json
                with open(path) as json_file:
                    data = json.load(json_file)

                    tweet = preprocess(data['text'])
                    if raw_fake_tweets[tweet] == False:
                        log(tweet)
                        payload = {
                            'tweet': tweet,
                            'article_title': current_article,
                        }
                        raw_fake_tweets[tweet] = True
                        payload_tweets.append(payload)
                    else:
                        log("Duplicate tweet - ignoring.")
            log("\n\n\n")
        else:
            log("No tweets found... \n")

    with open(f"{file}.pkl", 'wb') as file:
        pickle.dump(payload_tweets, file)

    print("Done")

load_from_file("real")
load_from_file("fake")

import os
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
import re
import preprocess
import datetime
from utils import log
BASEURL = os.getcwd() + "/fakenewsanalysis/fakenewsnet_dataset/politifact/"


def load_from_file(file):
    # Optimisation: Defaultdict rather than list: O(n) -> O(1)
    raw_fake_tweets = defaultdict(lambda: False)
    payload_tweets = []
    directory = BASEURL + file
    for article in tqdm(os.listdir(directory)):
        article_path = directory + f"/{article}"

        try:
            with open(article_path + "/news content.json") as article_json:
                data = json.load(article_json)
                current_article = data["meta_data"]["twitter"]["title"]
                log(f"Exploring article: {current_article}")
                raw_fake_tweets[current_article] = True
        except:
            log("Article data empty. Ignoring...")
            continue

        if "tweets" in os.listdir(article_path):
            log("Found relevant tweets:")

            for tweet_json in os.listdir(article_path + "/tweets"):
                path = article_path + "/tweets/" + tweet_json
                with open(path) as json_file:
                    data = json.load(json_file)

                    tweet = preprocess(data["text"])
                    if raw_fake_tweets[tweet] == False:
                        log(tweet)
                        payload = {
                            "tweet": tweet,
                            "article_title": current_article,
                        }
                        raw_fake_tweets[tweet] = True
                        payload_tweets.append(payload)
                    else:
                        log("Duplicate tweet - ignoring.")
            log("\n\n\n")
        else:
            log("No tweets found... \n")

    with open(f"{file}.pkl", "wb") as file:
        pickle.dump(payload_tweets, file)

    print("Done")


load_from_file("real")
load_from_file("fake")
